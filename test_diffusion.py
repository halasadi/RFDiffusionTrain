
"""
Generalized Protein Structure Diffusion Inference Script

This code has been generalized from a TCR-specific implementation to serve as 
a template for various protein structure diffusion tasks. You will need to adapt 
this template to your specific use case.

Adapted from RFDiffusion/scripts/run_inference.py

"""

import os
import sys
import argparse
import torch
import glob as glob
import json

from omegaconf import OmegaConf
import logging
import numpy as np
import random

from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.contigs import ContigMap
from rfdiffusion.inference.utils import Denoise, process_target
from rfdiffusion.diffusion import Diffuser
from rfdiffusion.kinematics import xyz_to_t2d
from rfdiffusion.custom.diffusion_utils import (
    RFDiffusion,
    forward_noise_batched,
    reverse_step_batch,
)


def center_structure_at_fixed_region(xyz: torch.Tensor, diffusion_mask: torch.Tensor) -> torch.Tensor:
    """
    Center structure at fixed region Cα center of mass.
    If no fixed residues, center on entire structure.
    
    Args:
        xyz: Coordinate tensor [L, 27, 3]
        diffusion_mask: Boolean mask indicating which residues are fixed (True = fixed)
    
    Returns:
        Centered coordinate tensor
    """
    if diffusion_mask.sum() != 0:
        # Center at fixed region CA center of mass
        fixed_com = xyz[diffusion_mask, 1, :].mean(0)
        return xyz - fixed_com
    else:
        # Center at overall CA center of mass
        return xyz - xyz[:, 1, :].mean(0)


def build_batch_from_decoy(
    decoy_pdb_path: str,
    # TODO: Replace these parameters with your specific protein system parameters
    # Example parameters - customize for your application:
    region_a_len: int = 0,
    region_b_len: int = 0, 
    fixed_region_len: int = 0,
    contig_config: list = None,
    provide_seq_config: list = None,
    hotspot_residues: list = None,
):
    """
    Build a diffusion batch from a decoy PDB and sequence information.
    
    TODO: Customize this function for your specific protein system by:
    - Updating the parameter list to match your regions
    - Modifying the contig configuration logic
    - Adjusting sequence and structural constraints
    
    Args:
        decoy_pdb_path: Path to the decoy PDB structure
        region_a_len: Length of first diffusion region (customize name/purpose)
        region_b_len: Length of second diffusion region (customize name/purpose) 
        fixed_region_len: Length of fixed structural region (customize name/purpose)
        contig_config: List of contig configuration strings (will auto-generate if None)
        provide_seq_config: List of sequence provision strings (will auto-generate if None)
        hotspot_residues: List of important residues for the design task
    
    Returns:
        Dictionary containing batch data for diffusion
    """
    # 1. Parse decoy structure
    feats = process_target(decoy_pdb_path, parse_hetatom=False, center=False)
    seq_orig = feats['seq']
    seq_orig_raw = seq_orig.clone()
    L_full = len(seq_orig)

    # 2. Calculate total length and validate
    # TODO: Update this calculation for your specific regions
    total_len = region_a_len + region_b_len + fixed_region_len
    assert L_full == total_len, f"PDB length ({L_full}) != expected total length ({total_len})"

    # 3. Create ContigMap with provided or default configuration
    # TODO: Customize these contig configurations for your protein system
    if contig_config is None:
        # Example configuration - modify for your regions and constraints
        fixed_start = total_len - fixed_region_len
        fixed_end = total_len - 1
        contig_config = [
            f"{region_a_len}-{region_a_len}/0 "
            f"{region_b_len}-{region_b_len}/0 "
            f"D{fixed_start}-{fixed_end}"
        ]
    
    if provide_seq_config is None:
        # Example sequence provision - modify for your needs
        provide_seq_config = [f"{region_a_len}-{total_len-1}"]
    
    cm = ContigMap(feats, contig_config, provide_seq=provide_seq_config)

    # 4. Build masks and indices
    mask_seq = torch.from_numpy(cm.inpaint_seq)[None, :]
    mask_str = torch.from_numpy(cm.inpaint_str)[None, :]
    diffusion_mask = mask_str.squeeze(0)
    atom_mask = feats['mask_27']
    rf_feats = cm.rf
    con_ref = cm.con_ref_pdb_idx
    hal_idx = cm.hal_idx0

    # 5. Handle chain mapping
    chain_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    chain_id = [x[0] for x in feats['pdb_idx']]
    numeric_chain_ids = [chain_mapping.get(c, 0) for c in chain_id]
    chain_id_tensor = torch.tensor(numeric_chain_ids)

    # 6. Center coordinates
    xyz_27 = center_structure_at_fixed_region(feats['xyz_27'], diffusion_mask)

    # 7. Prepare sequence with masking
    seq_t = torch.full((1, L_full), 21, dtype=torch.long).squeeze()  # 21 = mask token
    seq_t[hal_idx] = seq_orig[cm.ref_idx0]
    seq_t[mask_seq.squeeze()] = seq_orig[mask_seq.squeeze()]
    seq_t[~mask_seq.squeeze()] = 21
    seq_t = torch.nn.functional.one_hot(seq_t, num_classes=22).float()

    # 8. Build 2D distance representation from decoy
    decoy_slice = xyz_27[:, :14, :].clone()
    decoy_slice[~mask_str.squeeze(), 3:, :] = float('nan')
    decoy_t = decoy_slice[None, None]
    decoy_t = torch.cat((decoy_t, torch.full((1, 1, L_full, 13, 3), float('nan'))), dim=3)
    t2d_decoy = xyz_to_t2d(decoy_t).squeeze(0)

    # 9. Define special structural indices (to be preserved during diffusion)
    # TODO: Implement logic based on your specific structural constraints
    # Example: indices of important structural elements to preserve
    special_indices = []  # Replace with your constraint logic
    # Example logic:
    # special_indices = list(range(region_a_len, region_a_len + region_b_len))  # Keep region B t2d fixed
    
    # 10. Add batch dimensions and prepare outputs
    xyz_mapped = xyz_27.unsqueeze(0)
    atom_mask_mapped = atom_mask.unsqueeze(0)
    seq_t = seq_t.unsqueeze(0)
    diffusion_mask = diffusion_mask.unsqueeze(0)
    t2d_decoy = t2d_decoy.unsqueeze(0)
    chain_id_tensor = chain_id_tensor.unsqueeze(0)
    seq_orig_raw = seq_orig_raw.unsqueeze(0)

    binderlen = len(cm.inpaint)

    # 11. Format RF features and indices
    rf = [torch.tensor(feat, dtype=torch.float32).unsqueeze(0) for feat in rf_feats]
    hal_idx0 = [torch.tensor([idx], dtype=torch.long) for idx in hal_idx]
    con_ref_pdb_idx = [([chain], torch.tensor([idx], dtype=torch.long)) for chain, idx in con_ref]

    # 12. Set default hotspot residues if not provided
    # TODO: Define hotspot residues relevant to your protein system
    if hotspot_residues is None:
        # Example: default to first region residues as hotspots
        hotspot_residues = [f"A{i+1}" for i in range(min(region_a_len, 12))]
        hotspot_residues.extend([''] * (12 - len(hotspot_residues)))

    return {
        'L_mapped': L_full,
        'seq_orig_raw': seq_orig_raw,
        'xyz_mapped': xyz_mapped,
        'atom_mask_mapped': atom_mask_mapped,
        'seq_t': seq_t,
        'diffusion_mask': diffusion_mask,
        'hotspot_res': [hotspot_residues],
        'binderlen': binderlen,
        'mask_str': mask_str,
        'mask_seq': mask_seq,
        'rf': rf,
        'con_ref_pdb_idx': con_ref_pdb_idx,
        'hal_idx0': hal_idx0,
        'special_indices': [special_indices],
        't2d_decoy': t2d_decoy,
        "chain_id": chain_id_tensor
    }


def load_diffuser_and_denoiser(model_name: str, model_dir: str, device: str):
    """Load and configure the diffuser, denoiser, and RF model."""
    
    cache_dir = "./schedules/"
    
    # TODO: Adjust these configurations for your specific model
    d_cfg = OmegaConf.create({
        "T": 50,
        "b_0": 0.01,
        "b_T": 0.07,
        "schedule_type": "linear",
        "so3_type": "igso3",
        "crd_scale": 0.25,  # Updated from 0.1 to match training
        "partial_T": 0,
        "so3_schedule_type": "linear",
        "min_b": 1.5,
        "max_b": 2.5,
        "min_sigma": 0.02,
        "max_sigma": 1.5,
    })

    den_cfg = OmegaConf.create({
        "noise_scale_ca": 1,
        "final_noise_scale_ca": 1,
        "ca_noise_schedule_type": "constant",
        "noise_scale_frame": 1,
        "final_noise_scale_frame": 1,
        "frame_noise_schedule_type": "constant",
    })

    model_config = OmegaConf.create({
        "n_extra_block": 4,
        "n_main_block": 32,
        "n_ref_block": 4,
        "d_msa": 256,
        "d_msa_full": 64,
        "d_pair": 128,
        "d_templ": 64,
        "n_head_msa": 8,
        "n_head_pair": 4,
        "n_head_templ": 4,
        "d_hidden": 32,
        "d_hidden_templ": 32,
        "p_drop": 0.15,
        "SE3_param_full": {
            "num_layers": 1,
            "num_channels": 32,
            "num_degrees": 2,
            "n_heads": 4,
            "div": 4,
            "l0_in_features": 8,
            "l0_out_features": 8,
            "l1_in_features": 3,
            "l1_out_features": 2,
            "num_edge_features": 32,
        },
        "SE3_param_topk": {
            "num_layers": 1,
            "num_channels": 32,
            "num_degrees": 2,
            "n_heads": 4,
            "div": 4,
            "l0_in_features": 64,
            "l0_out_features": 64,
            "l1_in_features": 3,
            "l1_out_features": 2,
            "num_edge_features": 64,
        },
        "freeze_track_motif": False,
        "use_motif_timestep": True,
    })
    
    preprocess = OmegaConf.create({
        "sidechain_input": False,
        "motif_sidechain_input": True,
        "d_t1d": 22,  # Updated to match training script
        "d_t2d": 44,
        "prob_self_cond": 0.5,
        "str_self_cond": True,
        "predict_previous": False,
    })
    
    logging_conf = OmegaConf.create({"inputs": False})

    sampler_config = OmegaConf.create({
        "model": model_config,
        "preprocess": preprocess,
        "diffuser": d_cfg,
        "logging": logging_conf,
        "model_name": model_name,
        "model_directory": model_dir,
    })

    # Initialize diffuser
    diffuser = Diffuser(
        T=d_cfg.T,
        b_0=d_cfg.b_0,
        b_T=d_cfg.b_T,
        min_sigma=d_cfg.min_sigma,
        max_sigma=d_cfg.max_sigma,
        min_b=d_cfg.min_b,
        max_b=d_cfg.max_b,
        schedule_type=d_cfg.schedule_type,
        so3_schedule_type=d_cfg.so3_schedule_type,
        so3_type=d_cfg.so3_type,
        crd_scale=d_cfg.crd_scale,
        cache_dir=cache_dir,
        partial_T=d_cfg.partial_T,
    )
    
    # Initialize denoiser
    denoiser = Denoise(
        T=d_cfg.T,
        diffuser=diffuser,
        b_0=d_cfg.b_0,
        b_T=d_cfg.b_T,
        min_b=d_cfg.min_b,
        max_b=d_cfg.max_b,
        min_sigma=d_cfg.min_sigma,
        max_sigma=d_cfg.max_sigma,
        noise_level=0.5,
        schedule_type=d_cfg.schedule_type,
        so3_schedule_type=d_cfg.so3_schedule_type,
        so3_type=d_cfg.so3_type,
        noise_scale_ca=den_cfg.noise_scale_ca,
        final_noise_scale_ca=den_cfg.final_noise_scale_ca,
        ca_noise_schedule_type=den_cfg.ca_noise_schedule_type,
        noise_scale_frame=den_cfg.noise_scale_frame,
        final_noise_scale_frame=den_cfg.final_noise_scale_frame,
        frame_noise_schedule_type=den_cfg.frame_noise_schedule_type,
        crd_scale=d_cfg.crd_scale,
    )

    # Load RF model
    models_dir = sampler_config.model_directory
    ckpt = f"{models_dir}/{sampler_config.model_name}.pt"
    rf_model = RFDiffusion(conf=sampler_config, ckpt_path=ckpt)
    rf_model = rf_model.to(device)

    return rf_model, diffuser, denoiser, sampler_config


def rosettafold_forward(rf_model, sampler_conf, diffuser, batch, xt_input, x0_prev, t_value):
    """
    Forward pass through RF model with preprocessing and self-conditioning.
    """
    device = xt_input.device
    batch_t2d = batch['t2d_decoy'].to(device)
    batch_special_indices = batch['special_indices']
    batch_rf = torch.stack(batch['rf'], dim=1)
    batch_hal = torch.stack(batch['hal_idx0'], dim=1)
    mask_str = batch['mask_str'].squeeze(1)

    # 1. Preprocessing
    msa_masked, msa_full, seq_in, xt_in, idx, t1d, t2d, xyz_t, alpha_t = \
        rf_model.preprocess_batch(
            seq=batch['seq_t'],
            xyz_t=xt_input,
            t=t_value,
            binderlen=batch['binderlen'],
            hotspot_res=batch['hotspot_res'],
            mask_str=mask_str,
            sidechain_input=sampler_conf.preprocess.sidechain_input,
            rf=batch_rf,
            d_t1d=sampler_conf.preprocess.d_t1d,
            con_ref_pdb_idx=batch['con_ref_pdb_idx'],
            hal_idx0=batch_hal,
        )

    # Move tensors to device
    msa_masked = msa_masked.to(device)
    msa_full = msa_full.to(device)
    seq_in = seq_in.to(device)
    xt_in = xt_in.to(device)
    t1d = t1d.to(device)
    t2d = t2d.to(device)
    xyz_t = xyz_t.to(device)
    alpha_t = alpha_t.to(device)
    
    B, _, L = xyz_t.shape[:3]

    # 2. Self-conditioning
    if t_value < diffuser.T:
        zeros = torch.zeros(B, 1, L, 24, 3, device=x0_prev.device)
        xyz_t = torch.cat((x0_prev.unsqueeze(1), zeros), dim=-2)
        t2d_44 = xyz_to_t2d(xyz_t)
    else:
        xyz_t = torch.zeros_like(xyz_t)
        t2d_44 = torch.zeros_like(t2d[..., :44])
    t2d[..., :44] = t2d_44

    # 3. Restore special structural elements
    for b, special_indices in enumerate(batch_special_indices):
        if len(special_indices) > 0:
            idxs = torch.tensor(special_indices, device=t2d.device)
            rows = idxs.unsqueeze(1).expand(-1, idxs.size(0))
            cols = idxs.unsqueeze(0).expand(idxs.size(0), -1)
            t2d[b, 0, rows, cols, :] = batch_t2d[b, 0, rows, cols, :]

    # 4. Forward pass
    t_gpu = torch.tensor(t_value, device=msa_masked.device)
    with torch.autocast(device_type='cuda'):
        logits, xyz_pred, px0 = rf_model(
            msa_masked, msa_full, seq_in, xt_in,
            idx, t1d, t2d, xyz_t, alpha_t,
            t_gpu, batch['diffusion_mask'],
        )

    return logits, xyz_pred, px0


def run_diffusion(args):
    """Main diffusion inference function."""
    
    # TODO: Replace these with your specific protein system parameters
    # Example parameters - customize for your application:
    region_a_len = getattr(args, 'region_a_len', 50)  # Replace with your region
    region_b_len = getattr(args, 'region_b_len', 100)  # Replace with your region
    fixed_region_len = getattr(args, 'fixed_region_len', 200)  # Replace with your fixed region
    
    # TODO: Define your contig configuration
    # Example contig config - customize for your protein system:
    contig_config = None  # Will use default in build_batch_from_decoy
    provide_seq_config = None  # Will use default in build_batch_from_decoy
    
    # Build batch
    batch = build_batch_from_decoy(
        decoy_pdb_path=args.decoy_pdb,
        region_a_len=region_a_len,
        region_b_len=region_b_len,
        fixed_region_len=fixed_region_len,
        contig_config=contig_config,
        provide_seq_config=provide_seq_config,
        hotspot_residues=None  # Will use default
    )

    # Load models
    rf_model, diffuser, denoiser, sampler_config = load_diffuser_and_denoiser(
        args.model_name, args.model_dir, args.device
    )

    # Initialize with noise
    with torch.no_grad():
        x_t = forward_noise_batched(
            batch['xyz_mapped'], diffuser.T,
            batch['seq_t'], diffuser,
            batch['atom_mask_mapped'], batch['diffusion_mask']
        )
    x0_prev = torch.zeros_like(batch['xyz_mapped'])

    x_t = x_t.to(args.device)
    x0_prev = x0_prev.to(args.device)

    # Run reverse diffusion
    px0_stack = []
    denoised_xyz_stack = []
    
    for t in range(diffuser.T, 0, -1):
        print(f"Running diffusion step t={t}...")
        with torch.no_grad():
            logits, xyz_pred, px0 = rosettafold_forward(
                rf_model, sampler_config, diffuser,
                batch, x_t, x0_prev, t
            )
        
        if t > 1:
            x_t = reverse_step_batch(x_t, px0, t, batch['diffusion_mask'], denoiser)
        else:
            x_t = px0

        x0_prev = px0
        px0_stack.append(px0)
        denoised_xyz_stack.append(x_t)

    # Process results
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, dims=[0])
    px0_stack = torch.stack(px0_stack)
    px0_stack = torch.flip(px0_stack, dims=[0])

    # Remove batch dimension
    denoised_xyz_stack = denoised_xyz_stack.squeeze(1)
    px0_stack = px0_stack.squeeze(1)

    # Prepare final sequence
    final_seq = torch.where(
        torch.argmax(batch["seq_t"], dim=-1) == 21,
        torch.tensor(7),  # 7 is glycine
        torch.argmax(batch["seq_t"], dim=-1),
    )

    # Compute B-factors (0 for diffused positions)
    bfacts = torch.ones_like(final_seq.squeeze(0))
    mask_diff = (torch.argmax(batch["seq_t"], dim=-1).squeeze(0) == 21)
    bfacts[mask_diff] = 0
    binderlen = batch['binderlen']

    # Write final PDB
    if os.path.dirname(args.out_prefix):
        os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    
    final_out_pdb = f"{args.out_prefix}.pdb"
    final_xyz = px0_stack[-1].squeeze(0)
    
    writepdb(
        final_out_pdb,
        final_xyz[:, :4, :], 
        final_seq,
        binderlen,
        bfacts=bfacts,
    )

    print(f"Written {final_out_pdb}")

    # Write trajectory if requested
    if args.write_trajectory:
        traj_dir = os.path.join(os.path.dirname(args.out_prefix), "traj")
        if traj_dir:
            os.makedirs(traj_dir, exist_ok=True)

        traj_prefix = os.path.join(traj_dir, os.path.basename(args.out_prefix))

        out_xt = f"{traj_prefix}_Xt-1_traj.pdb"
        writepdb_multi(
            out_xt,
            denoised_xyz_stack,
            bfacts,
            final_seq.squeeze(),
            use_hydrogens=False,
            backbone_only=False,
        )

        out_px0 = f"{traj_prefix}_pX0_traj.pdb"
        writepdb_multi(
            out_px0,
            px0_stack,
            bfacts,
            final_seq.squeeze(),
            use_hydrogens=False,
            backbone_only=False,
        )

        print("Trajectory files written to:", traj_dir)

    print(f"\nFinal PDB written to: {final_out_pdb}")


def main():
    parser = argparse.ArgumentParser(
        description="Run generalized protein structure diffusion inference"
    )
    
    # Required arguments
    parser.add_argument('--decoy_pdb', required=True,
                       help='Path to decoy PDB structure')
    parser.add_argument('--model_name', required=True,
                       help='Name of the trained model')
    parser.add_argument('--model_dir', required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--out_prefix', required=True,
                       help='Output prefix for generated files')
    
    # TODO: Add your specific protein system parameters here
    # Example parameters - replace with your regions:
    parser.add_argument('--region_a_len', type=int, default=50,
                       help='Length of region A (customize for your system)')
    parser.add_argument('--region_b_len', type=int, default=100,
                       help='Length of region B (customize for your system)')
    parser.add_argument('--fixed_region_len', type=int, default=200,
                       help='Length of fixed region (customize for your system)')
    
    # Optional arguments
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--write_trajectory', action='store_true',
                       help='Write diffusion trajectory PDBs')
    
    args = parser.parse_args()
    run_diffusion(args)


if __name__ == '__main__':
    main()