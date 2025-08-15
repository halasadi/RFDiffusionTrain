import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from rfdiffusion.util import rigid_from_3_points 
from rfdiffusion.kinematics import xyz_to_t2d, xyz_to_c6d, c6d_to_bins2
from pytorch_lightning.utilities import rank_zero_warn


# this is from the RFDiffusion Repo which differs from the supplementary section of the RFDiffusion paper
PARAMS = {
    "DMIN"    : 2.0,
    "DMAX"    : 20.0,
    "DBINS"   : 36,
    "ABINS"   : 36,
}

def compute_dframe_loss_batched_vectorized(xyz_pred, xyz_truth, wtrans, wrot, non_mask = None,
                                             l1_clamp_distance=10, eps=1e-8, gamma=0.99, p_clamp = 0.9):
    """
    Computes the dFrame loss in a fully vectorized manner over I intermediate outputs.
    
    Args:
      xyz_pred: Predicted full-atom coordinates, shape [I, B, L, num_atoms, 3].
      xyz_truth: Ground-truth coordinates, shape [B, L, num_atoms, 3].
      non_mask: Binary mask, shape [B, L] (1 for valid residues, 0 otherwise). Not used in this version.
      wtrans: Weight for translation error.
      wrot: Weight for rotation error.
      l1_clamp_distance: Clamp value for translation error.
      eps: Small constant.
      gamma: Exponential weight factor for intermediate outputs.
      
    Returns:
      total_loss: Scalar tensor (averaged over batch) representing the weighted dFrame loss.
    """
    # Extract dimensions.
    I, B, L, num_atoms, _ = xyz_pred.shape
    device = xyz_pred.device

    # --- Compute predicted rigid frames for each intermediate output ---
    # We use the first three atoms (assumed to be N, Cα, C) from xyz_pred.
    # x_pred has shape [I, B, L, 3, 3].
    x_pred = xyz_pred[:, :, :, :3, :]  
    # Combine intermediate and batch dimensions: shape [I*B, L, 3, 3]
    I_B = I * B
    x_pred_reshaped = x_pred.reshape(I_B, L, 3, 3)
    # Extract the three anchor atoms.
    a1 = x_pred_reshaped[:, :, 0, :].float()  # [I*B, L, 3]
    a2 = x_pred_reshaped[:, :, 1, :].float()  # [I*B, L, 3]
    a3 = x_pred_reshaped[:, :, 2, :].float()  # [I*B, L, 3]
    # Compute predicted rotation matrices and Cα coordinates.
    with torch.autocast(device_type='cuda', enabled=False):
        R_pred_flat, Ca_pred_flat = rigid_from_3_points(a1, a2, a3, non_ideal=False, eps=1e-5)  # Expect shapes: [I*B, L, 3, 3] and [I*B, L, 3]
    # Reshape back to separate intermediate and batch dimensions.
    R_pred = R_pred_flat.reshape(I, B, L, 3, 3)      # [I, B, L, 3, 3]
    Ca_pred = Ca_pred_flat.reshape(I, B, L, 3)         # [I, B, L, 3]
    
    # --- Compute target rigid frames from ground truth ---
    # xyz_truth: [B, L, num_atoms, 3]. Use the first three atoms.
    target_xyz = xyz_truth[:, :, :3, :]  # [B, L, 3, 3]
    t1 = target_xyz[:, :, 0, :].float()
    t2 = target_xyz[:, :, 1, :].float()
    t3 = target_xyz[:, :, 2, :].float()
    with torch.autocast(device_type='cuda', enabled=False):
        R_target, Ca_target = rigid_from_3_points(t1, t2, t3, non_ideal=False, eps=1e-5)  # [B, L, 3, 3] and [B, L, 3]
    # Expand target frames to all intermediate outputs.
    R_target_exp = R_target.unsqueeze(0)  # [1, B, L, 3, 3]
    Ca_target_exp = Ca_target.unsqueeze(0)  # [1, B, L, 3]
    
    # --- Compute translation error (per residue) ---

    # 1) compute L₂ norm per residue (shape [I, B, L])
    dist = torch.norm(Ca_pred - Ca_target_exp, dim=-1).clamp(min=eps)


    # 2) clamp the L₂ norm to l1_clamp_distance
    '''
    # 2) clamp that norm to dclamp
    # dist has shape [I, B, L]
    rv = torch.rand([], device=dist.device)
    if rv < p_clamp:
        # with probability p_clamp clamp *every* entry
        dist_clamped = dist.clamp(max=l1_clamp_distance)
    else:
        # with probability 1-p_clamp leave them all unclamped
        dist_clamped = dist
    '''

    dist_clamped = dist.clamp(max=l1_clamp_distance)
    
    # 3) square once to get the Å² penalty
    trans_error = dist_clamped ** 2  # [I, B, L]
    
    # --- Compute rotation error ---
    R_target_T = R_target_exp.transpose(-2, -1).float()  # [1, B, L, 3, 3]
    R_pred      = R_pred.float()                          # (I, B, L, 3, 3)
    rtx = torch.matmul(R_target_T, R_pred)         # [I, B, L, 3, 3]
    I3 = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(I, B, L, 3, 3)
    rot_error = torch.sum((I3 - rtx) ** 2, dim=(-2, -1))  # [I, B, L]

    # Compute per-intermediate loss: sum over residues (L) for each intermediate and each batch.
    per_int_loss = torch.sum((wtrans * trans_error + wrot * rot_error), dim=-1) / L
    
    # Weight each intermediate output.
    weights = torch.tensor([gamma**(I - 1 - i) for i in range(I)], dtype=per_int_loss.dtype, device=device).view(I, 1)  # [I, 1]
    weighted_loss = torch.sum(weights * per_int_loss, dim=0) / weights.sum()  # [B]
    
    # Finally, average over the batch.
    total_loss = torch.mean(weighted_loss)  # Scalar
    
    return(total_loss)

# this is from the RF2 Repo (also used in RFAntibody)
def calc_c6d_loss(logit_s, label_s, mask_2d, eps=1e-5):
    loss_s = list()
    for i in range(len(logit_s)):
        logits_32 = logit_s[i].float()  # ensure logits are in float3
        loss = nn.CrossEntropyLoss(reduction='none')(logits_32, label_s[...,i]) # (B, L, L)
        loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
        loss_s.append(loss)
    loss_s = torch.stack(loss_s)
    return loss_s

# here I follow the public implementation of RoseTTAFold2 versus the supplementary section of RFDiffusion paper
def compute_c6d_loss(
    logit_s, xyz, params=PARAMS, eps=1e-5
):
   
    # 1) upcast to double for the geometry
    xyz64 = xyz.double()

    # 2) turn off bf16 autocast for the entire geometry→angle step
    #with torch.autocast(device_type='cuda', enabled=False):
        # now xyz_to_c6d will run in FP64
    c6d64, contact_mask = xyz_to_c6d(xyz64, params)
    # c6d64 is float64; cast back to float32/bf16 for downstream
    c6d = c6d64.float()

    # build mask_crds: 1.0 if atom‐i exists, 0.0 if it was NaN
    mask_crds = (~torch.isnan(xyz[:,:, :3, :]).any(dim=-1)).float()   # [B, L, A]

    # then exactly as RF2:
    mask_BB = ~(mask_crds[:, :, :3].sum(dim=-1) < 3.0)    # [B, L]
    mask_2d  = mask_BB[:, None, :] * mask_BB[:, :, None]  # [B, L, L]

    # 2) binning
    label_s = c6d_to_bins2(c6d, same_chain = None, negative=False, params=params)  # [B, L, L, 4]

    # now call the loss exactly as before
    loss_d, loss_o, loss_t, loss_p = calc_c6d_loss(
        logit_s,
        label_s,
        mask_2d.float(),
        eps=eps,
    )
    
    return loss_d, loss_o, loss_t, loss_p
