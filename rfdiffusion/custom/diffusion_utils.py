import torch
import torch.nn as nn
from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfdiffusion.util_module import ComputeAllAtomCoords
from rfdiffusion.kinematics import xyz_to_t2d
import torch.nn.functional as F
import numpy as np
from hydra.core.hydra_config import HydraConfig
from rfdiffusion import util  # For torsion utilities

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles


class RFDiffusion(nn.Module):
    """
    RFDiffusion basically is a wrapper around the RoseTTAFoldModule.
    It takes care of loading the model, and also provides a function to
    preprocess the input data before passing it to the model.
    """
    def __init__(self, conf, ckpt_path):
        """
        Args:
            conf:
                model_config: Dict with parameters to instantiate RoseTTAFoldModule.
                preprocess_config: Object/dict with preprocessing parameters (e.g. d_t1d, d_t2d, T).
            ckpt_path: Path to the checkpoint file.
        """
        super().__init__()

        # Store the provided configuration dictionary.
        self._conf = conf

        # Load the checkpoint and store it.
        self.ckpt = torch.load(ckpt_path)

        # Optionally update the config based on the checkpoint.
        self.assemble_config_from_chk()

        # Store the total number of diffusion steps.
        self.T = conf.diffuser.T


        # Instantiate the RosettaFold model using (possibly updated) configuration.
        self.model = RoseTTAFoldModule(**conf.model,
                                       d_t1d=conf.preprocess.d_t1d,
                                       d_t2d=conf.preprocess.d_t2d,
                                       T=self.T)
        self.model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        
        # Instantiate the all-atom conversion module.
        self.allatom = ComputeAllAtomCoords()

    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
        
        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

        """
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = HydraConfig.get().overrides.task
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        for cat in ['model','diffuser','preprocess']:
            for key in self._conf[cat]:
                try:
                    print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                    self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                except:
                    pass
        
        # add overrides back in again
        for override in overrides:
            if override.split(".")[0] in ['model','diffuser','preprocess']:
                print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
    
    def preprocess_batch(self, seq, xyz_t, t, binderlen, hotspot_res, mask_str, sidechain_input, rf, d_t1d, con_ref_pdb_idx, hal_idx0, repack=False):
        """        
        Function to prepare inputs to diffusion model
        
            seq (B, L,22) one-hot sequence 

            msa_masked (B,1,L,48)

            msa_full (B,1,L,25)
        
            xyz_t (B, L,14,3) template crds (diffused) 

            t1d (B,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)

                MODEL SPECIFIC:
                - contacting residues: for ppi. Target residues in contact with binder (1)
                - empty feature (legacy) (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (B, L, L, 45)
                - last plane is block adjacency
        """

        B, L, _ = seq.shape  # Extract batch size (B) and sequence length (L)
        T = self.T
        
        ##################
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((B, 1, L, 48))
        msa_masked[:, :, :, :22] = seq.unsqueeze(1)
        msa_masked[:, :, :, 22:44] = seq.unsqueeze(1)
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
    
        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((B, 1, L, 25))
        msa_full[:, :, :, :22] = seq.unsqueeze(1)
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
    
    
        ###########
        ### t1d ###
        ########### 
        t1d = torch.zeros((B, 1, L, 21))
        
        # Convert 22-class one-hot to 21-class one-hot
        seqt1d = seq.clone()
        seqt1d[:, :, 20] += seqt1d[:, :, 21]
        seqt1d[:, :, 21] = 0
        t1d[:, :, :, :21] = seqt1d[:, None, :, :21]
        
        # Compute time feature
        # if Batch == 1, comment this out
        #mask_str = mask_str.squeeze()  # (B, L)
        
        timefeature = torch.zeros((B, L)).float()
        timefeature[mask_str] = 1  # Set 1 where diffusion mask is True
        timefeature[~mask_str] = 1 - (t / T)  # Broadcast t over L
        
        timefeature = timefeature[:, None, ..., None]
        
        t1d = torch.cat((t1d, timefeature), dim=-1).float()
        t1d_save = t1d.clone()
    
    
        #############
        ### xyz_t ###
        #############        
        if sidechain_input:
            xyz_t[torch.where(seq == 21)[0], torch.where(seq == 21)[1], 3:, :] = float('nan')
        else:
            xyz_t[torch.where(~mask_str)[0], torch.where(~mask_str)[1], 3:, :] = float('nan')
        
        xyz_t = xyz_t[:, None, :, :, :]  # Expand along new dimensions
        xyz_t = torch.cat((xyz_t, torch.full((B, 1, L, 13, 3), float('nan'), device = xyz_t.device)), dim=3)
    
    
        t2d = xyz_to_t2d(xyz_t)  # Ensure `xyz_to_t2d` supports batched inputs
    
        idx = rf
    
        ###############
        ### alpha_t ###
        ###############
        seq_tmp = t1d_save[..., :-1].argmax(dim=-1).reshape(B, L)

        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(B, L, 27, 3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES
        )
        alpha = alpha.to(alpha_mask.device)
        not_nan_alpha_mask = ~torch.isnan(alpha[..., 0])
        alpha_mask = torch.logical_and(alpha_mask, not_nan_alpha_mask)
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(B, 1, L, 10, 2)
        alpha_mask = alpha_mask.reshape(B, 1, L, 10, 1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, 1, L, 30)
    
        msa_masked = msa_masked.to(xyz_t.device)
        msa_full = msa_full.to(xyz_t.device)
        seq = seq.to(xyz_t.device)
        xyz_t = xyz_t.to(xyz_t.device)
        idx = idx.to(xyz_t.device)
        t1d = t1d.to(xyz_t.device)
        t2d = t2d.to(xyz_t.device)
        alpha_t = alpha_t.to(xyz_t.device)
    

        if d_t1d >= 24: # add hotspot residues
            hotspot_tens = torch.zeros((B, L)).float().to(xyz_t.device)

            # no hotspots
            if len(hotspot_res) == 0:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots.\
                If you're doing monomer diffusion this is fine")
            else:
                #hotspots = [[(item[i][0], int(item[i][1:])) for item in hotspot_res] for i in range(B)]
                hotspots = [[(item[i][0], int(item[i][1:])) for item in hotspot_res if item[i] != ''] for i in range(B)]
                hotspot_idx = []
                for b in range(B):
                    sub_idx = []
                    
                    # Extract the b-th batch of con_ref_pdb_idx
                    #this_batch_con_ref_pdb_idx = [(x[b], y[b].item()) for x, y in con_ref_pdb_idx] 
                    this_batch_con_ref_pdb_idx = [(x[b], y[b].item()) for x, y in con_ref_pdb_idx if (x[b] != '' or torch.isnan(y[b]))]
                
                    # Iterate over residues and check membership
                    for i, res in enumerate(this_batch_con_ref_pdb_idx):
                        if res in hotspots[b]:
                            sub_idx.append(hal_idx0[b, i].item())  # Ensure integer indexing
                
                    hotspot_idx.append(torch.tensor(sub_idx, dtype=torch.long))
                
                for b in range(B):
                    hotspot_tens[b, hotspot_idx[b]] = 1.0  # Assign 1.0 at the indices for batch b
                
            t1d = torch.cat((t1d, torch.zeros_like(t1d[..., :1]).to(xyz_t.device), hotspot_tens[:, None, :, None].to(xyz_t.device)), dim=-1) 

        device = seq.device
        dtype  = seq.dtype  # will be float16 when you're in 16‑bit mode

        msa_masked = msa_masked.to(device=device, dtype=dtype)
        msa_full   = msa_full.to(device=device, dtype=dtype)
        t1d        = t1d.to(device=device, dtype=dtype)
        t2d        = t2d.to(device=device, dtype=dtype)
        xyz_t      = xyz_t.to(device=device, dtype=dtype)
        alpha_t    = alpha_t.to(device=device, dtype=dtype)
    
        return msa_masked, msa_full, seq, torch.squeeze(xyz_t, dim=1), idx, t1d, t2d, xyz_t, alpha_t
    


    def forward(self, msa_masked, msa_full, seq, xt, idx, t1d, t2d, xyz_t, alpha_t, t_gpu, diffusion_mask):
        """
        Forward pass through the model followed by conversion to all-atom coordinates.
        """
        logits, logits_aa, logits_exp, xyz, alpha, plddt = self.model(msa_masked,
                            msa_full,
                            seq,
                            xt,
                            idx,
                            t1d=t1d,
                            t2d=t2d,
                            xyz_t=xyz_t,
                            alpha_t=alpha_t,
                            msa_prev = None,
                            pair_prev = None,
                            state_prev = None,
                            t=t_gpu,
                            return_infer=False,
                            motif_mask = diffusion_mask,
                            use_checkpoint=True)

    
        _, px0 = self.allatom(torch.argmax(seq, dim=-1), xyz[-1], alpha[-1])  # px0: (B, L, N, 3)
        px0 = px0[..., :14, :]  # Keep only the first 14 atoms per residue
        
        return logits, xyz, px0
    

@torch.no_grad()
def forward_noise_batched(x0, t, seq_t, diffuser, atom_mask_mapped, mask_str):
    """
    Simulates the ForwardNoise function using the functions from RFDiffusion package
    """
    t_list = [t]

    fa_stack, xyz_true = diffuser.diffuse_pose_batched(
    x0,
    seq_t,
    atom_mask_mapped,
    diffusion_mask=mask_str,
    t_list=t_list)
    xT = fa_stack.squeeze(1)[:, :, :14, :]
    xt = torch.clone(xT)
    return(xt) 

def reverse_step_batch(xt_batch, px0_batch, t, mask_str_batch, denoiser):
    """
    Batched version of reverse_step. Iterates over the batch dimension
    to compute the reverse step for each item in the batch.
    
    Args:
        xt_batch: Tensor of shape [B, ...] containing xt values for each batch.
        px0_batch: Tensor of shape [B, ...] containing px0 predictions for each batch.
        t_batch: Tensor of shape [B] containing timesteps for each batch.
        diffusion_mask_batch: Tensor of shape [B, ...] containing diffusion masks.
        denoiser: A denoiser object that provides get_next_pose().
    
    Returns:
        x_t_1_batch: Tensor of shape [B, ...] containing the updated xt-1 values.
    """
    batch_size = xt_batch.shape[0]  # Assuming batch dim is first
    x_t_1_list = []

    for i in range(batch_size):
        x_t_1, _ = denoiser.get_next_pose(
            xt=xt_batch[i], 
            px0=px0_batch[i], 
            t=t, 
            diffusion_mask=mask_str_batch[i].squeeze()
        )
        x_t_1 = x_t_1.detach()
        x_t_1_list.append(x_t_1.detach())

    return torch.stack(x_t_1_list, dim=0)  # Stack back into a si
