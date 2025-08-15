"""
Generalized Protein Structure Diffusion Training Script

This code has been generalized from a specific implementation to serve as 
a template for various protein structure diffusion tasks. You will need to adapt 
this template to your specific use case by:

All areas requiring customization are marked with TODO comments throughout the code.

"""

import datetime
import json
import logging
import os
import subprocess
import tempfile
import shutil

import pytorch_lightning as pl
import torch
import torch.optim as optim
import math
from immunodata.kubernetes import PodMapper, shared_memory_volume
from immunodata.refs_azure import ResourceGroupSelector
from immunodata.refs_services import get_refs
from immunodata.workflows import WorkflowsRefs, WeightsAndBiasesRefs
from immunodata.workflows._base import FromRun, GPUSupportCustomization, GPUType
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, Callback

from torch.utils.data import DataLoader

# TODO: Update these imports to match your generalized dataset
from rfdiffusion.custom.dataset import ProteinStructureDataset  # Updated import name
from rfdiffusion.custom.dataset import my_collate
from rfdiffusion.custom.diffusion_utils import (
    RFDiffusion,
    forward_noise_batched,
    reverse_step_batch,
)
from rfdiffusion.custom.loss import compute_dframe_loss_batched_vectorized, compute_c6d_loss
from rfdiffusion.custom.utils import batch_process_pmpnnsample
from rfdiffusion.diffusion import Diffuser
from rfdiffusion.inference.utils import Denoise
from rfdiffusion.kinematics import xyz_to_t2d

logger = logging.getLogger(__name__)

os.environ["WANDB_DEBUG"] = "true"


def azcopy(input_dir: str, output_dir: str, quiet: bool = False):
    azure_credentials = {
        k: os.environ[k]
        for k in ("AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_FEDERATED_TOKEN_FILE")
    }
    os.makedirs(output_dir, exist_ok=True)
    call = f"azcopy copy {input_dir} {output_dir} --recursive=true"
    if quiet:
        call += " --output-level quiet"
    start = datetime.datetime.now()
    subprocess.run(
        call,
        shell=True,
        check=True,
        env={"AZCOPY_AUTO_LOGIN_TYPE": "WORKLOAD", **azure_credentials},
    )
    elapsed = datetime.datetime.now() - start
    print(f"TIME ELAPSED: {elapsed} for {input_dir}")


import threading


class AzureCheckpointUploader(pl.Callback):
    def __init__(self, run_local_dir: str, azure_base_path: str, quiet=True):
        super().__init__()
        self.run_local_dir = run_local_dir
        self.azure_base_path = azure_base_path.rstrip("/")
        self.quiet = quiet
        self._uploaded = set()
        self._threads = []

    def _upload_async(self, src, dest, quiet):
        try:
            azcopy(src, dest, quiet=quiet)
            print(f"[AzureUploader] async upload succeeded: {os.path.basename(src)}")
        except Exception as e:
            print(f"[AzureUploader] async upload failed for {os.path.basename(src)}: {e}")

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return

        epoch_num = trainer.current_epoch + 1
        # TODO: Update model names based on your architecture
        for name, model in [("rosettafold", pl_module.rosettafold.model)]:  # Removed pmpnn reference
            fname = f"{name}_epoch{epoch_num:02d}.pt"
            path = os.path.join(self.run_local_dir, fname)
            if path in self._uploaded:
                continue
            try:
                save_dict = {"model_state_dict": model.state_dict()}
                torch.save(save_dict, path)
                dest = f"{self.azure_base_path}/{fname}"
                thread = threading.Thread(
                    target=self._upload_async, args=(path, dest, self.quiet), daemon=False
                )
                thread.start()
                self._threads.append(thread)
                self._uploaded.add(path)
                print(f"[AzureUploader] scheduled async upload for {fname}")
            except Exception as e:
                print(f"[AzureUploader] failed to save/upload {fname}: {e}")

    def on_train_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        print("[AzureUploader] waiting for pending uploads to finish...")
        for t in self._threads:
            t.join(timeout=300)  # adjust timeout if needed
        print("[AzureUploader] all pending uploads joined")


class ProteinDataModule(pl.LightningDataModule):
    """
    TODO: Customize this data module for your specific protein system:
    - Update directory structure and naming conventions
    - Modify data loading logic if needed
    - Adjust any protein-specific preprocessing
    """
    def __init__(
        self,
        scratch_prefix,
        train_structures,
        train_decoys,
        val_structures,
        val_decoys,
        meta_info_csv,
        base_model_path,
        model_dir,
        output_dir,
        batch_size=1,
        crop_size=384,
    ):
        super().__init__()
        self.scratch_prefix = scratch_prefix
        self.train_structures = train_structures
        self.train_decoys = train_decoys
        self.val_structures = val_structures
        self.val_decoys = val_decoys
        self.meta_info_csv = meta_info_csv
        self.base_model_path = base_model_path
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.crop_size = crop_size
    
    def _remote_to_local(self, remote_path: str) -> str:
        # e.g. "ground_truth_structures/train" → "/home/data/ground_truth_structures/train"
        parts = remote_path.rstrip("/").split("/")
        return os.path.join(self.output_dir, parts[-2], parts[-1])

    @rank_zero_only
    def prepare_data(self):
        # install azcopy if needed
        cmds = [
            "curl https://azcopyvnext-awgzd8g7aagqhzhe.b02.azurefd.net/releases/release-10.27.1-20241113/azcopy_linux_amd64_10.27.1.tar.gz -o azcopy.tar.gz",
            "tar -xvf azcopy.tar.gz",
            "cp azcopy_linux_amd64_10.27.1/azcopy /usr/bin/",
            "rm -rf azcopy*",
        ]

        for c in cmds:
            subprocess.run(c, shell=True, check=True)

        print("Copying data from scratch…")

        # TODO: Update data paths for your specific directory structure
        for path in (
            self.train_structures,
            self.train_decoys,
            self.val_structures,
            self.val_decoys,
        ):
            local_dir = self._remote_to_local(path)
            # copy only the contents of the remote dir, not the dir as a whole
            src = f"{self.scratch_prefix}/{path}/*"
            azcopy(src, local_dir, quiet=True)

        azcopy(f"{self.scratch_prefix}/{self.base_model_path}", self.model_dir, quiet=True)

        # finally copy the single CSV into the root
        azcopy(f"{self.scratch_prefix}/{self.meta_info_csv}",
               self.output_dir, quiet=True)

        # check if the data is copied correctly
        print(os.system(f"ls -l {self.output_dir}"))

    def setup(self, stage=None):
        os.system('ls ' + self.output_dir)

        if stage in (None, "fit", "validate"):
            train_structs = self._remote_to_local(self.train_structures)
            train_decoys = self._remote_to_local(self.train_decoys)
            val_structs = self._remote_to_local(self.val_structures)
            val_decoys = self._remote_to_local(self.val_decoys)
            meta_csv = os.path.join(self.output_dir,
                                         os.path.basename(self.meta_info_csv))

            print("Using:", train_structs, train_decoys, val_structs, val_decoys)
            print("CSV at:", meta_csv)

            # TODO: Update dataset instantiation to match your generalized class
            self.train_dataset = ProteinStructureDataset(
                target_dir=train_structs,
                decoy_dir=train_decoys,
                meta_info_path=meta_csv,
                crop_size=self.crop_size,
                split="train",
            )
            self.val_dataset = ProteinStructureDataset(
                target_dir=val_structs,
                decoy_dir=val_decoys,
                meta_info_path=meta_csv,
                crop_size=self.crop_size,
                split="validation",
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=my_collate, shuffle=True, num_workers=4, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=my_collate, shuffle=False, num_workers=4, persistent_workers=True)

    def on_train_start(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total}")


class DiffusionLightningModel(pl.LightningModule):
    """
    TODO: Customize this model for your specific protein diffusion task:
    - Update the forward pass logic for your protein architecture
    - Modify validation logic to match your evaluation needs
    - Adjust loss functions and metrics for your application
    - Update any protein-specific processing steps
    """
    def __init__(
        self,
        num_epochs,
        diffuser_config,
        denoiser_config,
        sampler_config,
        preprocess,
        logging_conf,
        cache_dir,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cache_dir"])
        self.num_epochs = num_epochs

        # diffuser + denoiser + rosettafold inits
        self._init_diffuser(diffuser_config, cache_dir)
        self._init_denoiser(diffuser_config, denoiser_config)
        self._init_rosettafold(sampler_config)

        self.register_buffer("wtrans", torch.tensor(0.5))
        self.register_buffer("wrot", torch.tensor(1.0))
        print("Initialized DiffusionLightningModel")

    def _init_diffuser(self, cfg, cache_dir):
        self.diffuser_config = cfg
        self.diffuser = Diffuser(
            T=cfg.T,
            b_0=cfg.b_0,
            b_T=cfg.b_T,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            min_b=cfg.min_b,
            max_b=cfg.max_b,
            schedule_type=cfg.schedule_type,
            so3_schedule_type=cfg.so3_schedule_type,
            so3_type=cfg.so3_type,
            crd_scale=cfg.crd_scale,
            cache_dir=cache_dir,
            partial_T=cfg.partial_T,
        )

    def _init_denoiser(self, d_cfg, den_cfg):
        self.denoiser = Denoise(
            T=d_cfg.T,
            diffuser=self.diffuser,
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

    def _init_rosettafold(self, sampler_config):
        models_dir = sampler_config.model_directory
        ckpt = f"{models_dir}/{sampler_config.model_name}.pt"
        self.rosettafold = RFDiffusion(conf=sampler_config, ckpt_path=ckpt)
        self.sampler_config = sampler_config

    def _rosettafold_forward(
        self, batch, xt_input, x0_prev, t_value, batch_t2d, batch_special_indices, batch_rf, batch_hal_idx0, mask
    ):
        """
        Helper method for RosettaFold forward pass.
        
        TODO: Update this method to handle your specific protein regions and constraints:
        - Modify the region-specific processing logic
        - Update how special structural elements are handled
        - Adjust self-conditioning logic if needed
        """

        msa_masked, msa_full, seq_in, xt_in, idx, t1d, t2d, xyz_t, alpha_t = (
            self.rosettafold.preprocess_batch(
                seq=batch["seq_t"],
                xyz_t=xt_input,
                t=t_value,
                binderlen=batch["binderlen"],
                hotspot_res=batch["hotspot_res"],
                mask_str=mask,
                sidechain_input=self.sampler_config.preprocess.sidechain_input,
                rf=batch_rf,
                d_t1d=self.sampler_config.preprocess.d_t1d,
                con_ref_pdb_idx=batch["con_ref_pdb_idx"],
                hal_idx0=batch_hal_idx0,
            )
        )

        B, N, L = xyz_t.shape[:3]

        # self-conditioning logic
        if (t_value < self.diffuser.T):
            zeros = torch.zeros(B, 1, L, 24, 3, device=x0_prev.device, dtype=x0_prev.dtype)
            xyz_t = torch.cat((x0_prev.unsqueeze(1), zeros), dim=-2)  # [B,T,L,27,3]
            t2d_44 = xyz_to_t2d(xyz_t)  # [B,T,L,L,44]
        else:
            xyz_t = torch.zeros_like(xyz_t)
            t2d_44 = torch.zeros_like(t2d[..., :44])
        
        t2d[..., :44] = t2d_44

        # TODO: Update this section to handle your specific structural constraints
        # This code restores pairwise features for conserved structural elements
        for b, special_indices in enumerate(batch_special_indices):
            idxs = torch.tensor(special_indices, dtype=torch.long, device=t2d.device)
            # make a [K,K] index grid so rows[i,j],cols[i,j] enumerate special pairs
            rows = idxs.unsqueeze(1).expand(-1, idxs.size(0))  # [K,K]
            cols = idxs.unsqueeze(0).expand(idxs.size(0), -1)  # [K,K]

            # restore the special structural element pairwise features
            t2d[b,0,rows,cols,:] = batch_t2d[b,0,rows,cols,:]

        t_gpu = torch.tensor(t_value, device=msa_masked.device, dtype=msa_masked.dtype)

        with torch.autocast(device_type="cuda"):
            logits, xyz, px0 = self.rosettafold(
                msa_masked,
                msa_full,
                seq_in,
                xt_in,
                idx,
                t1d,
                t2d,
                xyz_t,
                alpha_t,
                t_gpu,
                batch["diffusion_mask"],
            )

        return logits, xyz, px0

    def training_step(self, batch, batch_idx):
        """
        Training step for protein structure diffusion.
        
        TODO: Customize this for your specific training objectives:
        - Update region-specific processing
        - Modify loss calculations for your application
        - Adjust any protein-specific logic
        """
        # Extract and process batch data
        batch_rf = torch.stack(batch["rf"], dim=1)
        batch_hal_idx0 = torch.stack(batch["hal_idx0"], dim=1)
        x0 = batch["xyz_mapped"][:, :, :14, :]
        mask = batch["mask_str"].squeeze(1)
        T = self.diffuser.T  # total diffusion steps
        t = torch.randint(1, T + 1, (1,)).item()

        # Compute noise and reverse diffusion step
        if torch.rand(1).item() < 0.5 or t == T:
            xt = forward_noise_batched(
                x0,
                t,
                batch["seq_t"],
                self.diffuser,
                batch["atom_mask_mapped"],
                batch["diffusion_mask"],
            )
            x0_prev = torch.zeros_like(x0, device=x0.device, dtype=x0.dtype)
        else:
            xt_plus_1 = forward_noise_batched(
                x0,
                t + 1,
                batch["seq_t"],
                self.diffuser,
                batch["atom_mask_mapped"],
                batch["diffusion_mask"],
            )
            xt = reverse_step_batch(xt_plus_1, x0, t + 1, batch["diffusion_mask"], self.denoiser)

            with torch.no_grad():
                _, x0_prev_temp, _ = self._rosettafold_forward(
                    batch,
                    xt_input=xt_plus_1,
                    x0_prev=torch.zeros_like(x0, device=x0.device, dtype=x0.dtype),
                    t_value=t + 1,
                    batch_t2d=batch["t2d_decoy"],
                    batch_special_indices=batch["special_indices"],  # Updated from tcr_noncdr3_indices
                    batch_rf=batch_rf,
                    batch_hal_idx0=batch_hal_idx0,
                    mask=mask,
                )

                x0_prev = x0_prev_temp[-1]

        # Second diffusion step using xt and self-conditioned x0_prev
        logits, x0_pred, px0 = self._rosettafold_forward(
            batch,
            xt_input=xt,
            x0_prev=x0_prev,
            t_value=t,
            batch_t2d=batch["t2d_decoy"],
            batch_special_indices=batch["special_indices"],  # Updated from tcr_noncdr3_indices
            batch_rf=batch_rf,
            batch_hal_idx0=batch_hal_idx0,
            mask=mask,
        )

        if not torch.isfinite(px0).all():
            raise RuntimeError("px0 contains non-finite values — aborting!")
        
        # Compute backbone losses
        frame_loss = compute_dframe_loss_batched_vectorized(
            x0_pred,
            x0,
            self.wtrans,
            self.wrot,
            non_mask=None,
            l1_clamp_distance=10,
            eps=1e-8,
            gamma=0.99,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            loss_d, loss_o, loss_t, loss_p = compute_c6d_loss(
                logits, x0,
            )

        l2d_loss = loss_d + loss_o + loss_t + loss_p
        total_loss = frame_loss + l2d_loss

        # Log losses
        self.log("train/dframe_loss", frame_loss,
                on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/l2d_loss", l2d_loss,
                on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation step that performs full diffusion from noise to structure.
        
        TODO: Customize validation logic for your specific protein system:
        - Update region-specific evaluation
        - Modify metrics calculation for your application
        - Adjust structural quality assessments
        """
        print(f"→ Running validation_step for batch {batch_idx}")
        
        # Unpack batch data
        x0 = batch["xyz_mapped"][:, :, :14, :]  
        xdecoy = batch["decoy_xyz_mapped"][:, :, :14, :] 
        seq_t = batch["seq_t"]                      
        mask_str = batch["mask_str"].squeeze(1)       
        diff_mask = batch["diffusion_mask"]
        t2d_dec = batch["t2d_decoy"]               
        special_idxs = batch["special_indices"]  # Updated from tcr_noncdr3_indices
        rf_feats = torch.stack(batch["rf"], dim=1)
        hal_idx0 = torch.stack(batch["hal_idx0"], dim=1)
        atom_m = batch["atom_mask_mapped"]

        # Pick starting timestep
        T_int = self.diffuser.T

        # Initial noise
        with torch.no_grad():
            x_t = forward_noise_batched(xdecoy, T_int, seq_t,
                                        self.diffuser, atom_m, diff_mask)
        x0_prev = torch.zeros_like(x0, device=x0.device)

        # Full reverse-diffusion from T → 0
        for t_int in range(T_int, 0, -1):
            with torch.no_grad():
                logits, xyz_pred, px0 = self._rosettafold_forward(
                    batch,
                    xt_input=x_t,
                    x0_prev=x0_prev,
                    t_value=t_int,
                    batch_t2d=t2d_dec,
                    batch_special_indices=special_idxs,  # Updated from tcr_noncdr3_indices
                    batch_rf=rf_feats,
                    batch_hal_idx0=hal_idx0,
                    mask=mask_str,
                )
                
                with torch.autocast(device_type="cuda", enabled=False):
                    loss_d, loss_o, loss_t, loss_p = compute_c6d_loss(logits, x0)

            if t_int > 1:
                x_t = reverse_step_batch(x_t, px0, t_int, diff_mask, self.denoiser)
                x0_prev = px0
            else:
                x_t = px0

        # Compute validation losses
        dframe_loss = compute_dframe_loss_batched_vectorized(
            px0.unsqueeze(0), x0, self.wtrans, self.wrot,
            non_mask=None, l1_clamp_distance=10, eps=1e-8, gamma=0.99,
        )

        l2d_loss = loss_d + loss_o + loss_t + loss_p

        self.log("val/dframe_loss", dframe_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/l2d_loss", l2d_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        total_val_loss = dframe_loss + l2d_loss
        return total_val_loss

    def configure_optimizers(self):
        base_lr = 5.0e-4

        optimizer = optim.Adam(
            list(self.rosettafold.model.parameters()),
            lr=base_lr,
        )
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", base_lr)

        # Use decay schedule from paper: 0.9 every 1000 steps
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1_000,
            gamma=0.9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def train_protein_diffusion(
    path_to_train_structures: str,
    path_to_train_decoys: str,
    path_to_val_structures: str,
    path_to_val_decoys: str,
    path_to_meta_csv: str,
    path_to_base_model: str,
    model_name: str,
    model_dir: str,
    gpus: int = 1,
    batch_size: int = 1,
    crop_size: int = 384,
    max_epochs: int = 10,
    model_output_path: str = None,
    deepspeed_config: str = None,
) -> None:
    """
    Main training function for protein structure diffusion.
    
    TODO: Customize configurations for your specific protein system and training needs.
    """
    # Build configs inline (or load from JSON if you prefer)
    diffuser_config = OmegaConf.create(
        {
            "T": 50,
            "b_0": 0.01,
            "b_T": 0.07,
            "schedule_type": "linear",
            "so3_type": "igso3",
            "crd_scale": 0.25,
            "partial_T": 0,
            "so3_schedule_type": "linear",
            "min_b": 1.5,
            "max_b": 2.5,
            "min_sigma": 0.02,
            "max_sigma": 1.5,
        }
    )

    denoiser_config = OmegaConf.create(
        {
            "noise_scale_ca": 1,
            "final_noise_scale_ca": 1,
            "ca_noise_schedule_type": "constant",
            "noise_scale_frame": 1,
            "final_noise_scale_frame": 1,
            "frame_noise_schedule_type": "constant",
        }
    )

    model_config = OmegaConf.create(
        {
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
        }
    )
    preprocess = OmegaConf.create(
        {
            "sidechain_input": False,
            "motif_sidechain_input": True,
            "d_t1d": 22,  # TODO: Adjust based on your input features
            "d_t2d": 44,
            "prob_self_cond": 0.5,
            "str_self_cond": True,
            "predict_previous": False,
        }
    )
    logging_conf = OmegaConf.create({"inputs": False})

    sampler_config = OmegaConf.create(
        {
            "model": model_config,
            "preprocess": preprocess,
            "diffuser": diffuser_config,
            "logging": logging_conf,
            "model_name": model_name,
            "model_directory": model_dir,
        }
    )

    output_dir = "/home/data/"
    # TODO: Update these paths for your cloud storage setup
    scratch_account = "your_storage_account"
    scratch_container = "your_container"
    scratch_prefix = f"https://{scratch_account}.blob.core.windows.net/{scratch_container}"

    # Data + Model
    dm = ProteinDataModule(
        scratch_prefix=scratch_prefix,
        train_structures=path_to_train_structures,
        train_decoys=path_to_train_decoys,
        val_structures=path_to_val_structures,   
        val_decoys=path_to_val_decoys,     
        meta_info_csv=path_to_meta_csv,
        base_model_path=path_to_base_model,
        model_dir=model_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        crop_size=crop_size,
    )

    dm.prepare_data()

    model = DiffusionLightningModel(
        num_epochs=max_epochs,
        diffuser_config=diffuser_config,
        denoiser_config=denoiser_config,
        sampler_config=sampler_config,
        preprocess=preprocess,
        logging_conf=logging_conf,
        cache_dir="./schedules/",
    )

    # Choose strategy: DeepSpeed if config provided
    strategy = None
    if deepspeed_config:
        strategy = DeepSpeedStrategy(config=deepspeed_config)
        logger.info(f"Using DeepSpeed strategy with config: {deepspeed_config}")

    csv_logger = CSVLogger(
        save_dir=output_dir,
        name=model_output_path,
    )

    get_refs().to(WeightsAndBiasesRefs).ensure_logged_in()

    # TODO: Update project name and entity for your application
    wandb_logger = WandbLogger(
        project="protein_diffusion",  # Update this
        name=model_output_path,
        save_dir=output_dir,
        entity="your_entity",  # Update this
        log_model=False,
    )

    # TODO: Update Azure paths for your project
    scratch_base = f"{scratch_prefix}/your_username/protein_diffusion/trained_models_checkpts"
    azure_run_dir = f"{scratch_base}/{model_output_path}"
    run_local_dir = os.path.join(output_dir, "trained_models", model_output_path)  
    os.makedirs(run_local_dir, exist_ok=True)

    azure_uploader = AzureCheckpointUploader(run_local_dir, azure_run_dir, quiet=True)

    # Trainer setup
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy=strategy,
        max_epochs=max_epochs,
        default_root_dir=output_dir,
        callbacks=[TQDMProgressBar(refresh_rate=1), azure_uploader],
        precision="bf16",
        logger=[csv_logger, wandb_logger],
        num_sanity_val_steps=0,
        limit_train_batches=1.0,    
        check_val_every_n_epoch=1,
        limit_val_batches=0.2,
    )

    trainer.fit(model, datamodule=dm)

    # Clean up wandb
    for thislogger in trainer.loggers:
        if isinstance(thislogger, WandbLogger):
            run = thislogger.experiment
            try:
                run.log({"final/finished": True}, commit=True)
            except Exception:
                pass
            try:
                if hasattr(run, "flush"):
                    run.flush()
            except Exception:
                pass
            try:
                run.finish()
            except Exception as e:
                print(f"W&B finish() warning: {e}")

    # Save final models
    run_local_dir = os.path.join(output_dir, "trained_models", model_output_path)
    os.makedirs(run_local_dir, exist_ok=True)

    # Save Lightning checkpoint
    ckpt_path = os.path.join(run_local_dir, f"{model_output_path}.ckpt")
    trainer.save_checkpoint(ckpt_path)

    # Save RosettaFold state dict
    rf_state_path = os.path.join(run_local_dir, "rosettafold.pt")
    rf_save_dict = {'model_state_dict': model.rosettafold.model.state_dict()}
    torch.save(rf_save_dict, rf_state_path)

    # Copy metrics CSV
    src_csv = os.path.join(csv_logger.log_dir, "metrics.csv")
    dst_csv = os.path.join(run_local_dir, "metrics.csv")
    shutil.copy(src_csv, dst_csv)

    # Upload to Azure
    scratch_base = f"{scratch_prefix}/your_username/protein_diffusion/trained_models"
    azcopy(run_local_dir, f"{scratch_base}/{model_output_path}", quiet=True)

    logger.info(f"All artifacts for run '{model_output_path}' saved under {run_local_dir} and uploaded to {scratch_base}/{model_output_path}")


def workflow_func() -> None:
    """
    Main workflow function that gets executed in the cloud environment.
    
    TODO: Update the parameters and paths for your specific project.
    """
    with tempfile.NamedTemporaryFile() as ds_config_json:
        with open(ds_config_json.name, "w") as ds_config_json:
            json.dump(deepspeed_config, ds_config_json)

        train_protein_diffusion(
            path_to_train_structures=path_to_train_structures,
            path_to_train_decoys=path_to_train_decoys,
            path_to_val_structures=path_to_validation_structures,
            path_to_val_decoys=path_to_validation_decoys,
            path_to_meta_csv=path_to_meta_csv,
            path_to_base_model=path_to_base_model,
            model_name=model_name,
            model_dir=model_dir,
            max_epochs=max_epochs,
            model_output_path=model_output_path,
            gpus=gpus,
            batch_size=batch_size,
            crop_size=crop_size,
            deepspeed_config=ds_config_json.name,
        )


if __name__ == "__main__":
    # TODO: Update all these paths for your specific protein system and data structure
    path_to_train_structures = "your_username/protein_diffusion/data/structures/train/"
    path_to_validation_structures = "your_username/protein_diffusion/data/structures/validation/"
    path_to_meta_csv = "your_username/protein_diffusion/data/metadata.csv"
    path_to_train_decoys = "your_username/protein_diffusion/data/decoys/training/"
    path_to_validation_decoys = "your_username/protein_diffusion/data/decoys/validation/"

    path_to_base_model = "your_username/protein_diffusion/base_models/pretrained_model.pt"
    model_name = "pretrained_model"
    model_dir = "/opt/custom_projects/protein_diffusion/models"
    max_epochs = 10
    gpus = 4
    batch_size = 6
    crop_size = 384
    region = "westus3"
    new_build = True
    keep_pod_on_failure = False
    model_output_path = "protein_diffusion_run_v1"

    job_name = "train-protein-diffusion"

    # TODO: Adjust DeepSpeed configuration for your hardware and model size
    deepspeed_config = {
        "fp16": {"enabled": False, "min_loss_scale": 1},
        "amp": {"enabled": False, "opt_level": "O2"},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "profile": False,
        },
        "gradient_clipping": 0.2,
        "zero_force_ds_cpu_optimizer": False,
    }

    # TODO: Update the workflow execution parameters for your cloud environment
    from_pod: FromRun[PodMapper] = (
        get_refs()
        .to(WorkflowsRefs)
        .run(
            "protein-diffusion-training",  # Updated job name
            workflow_func,
            {"cpu": "22", "memory": "400Gi"},
            environment_name="development",
            resource_group_selector=ResourceGroupSelector(regions=("westus3",)),
            customizations=(
                GPUSupportCustomization(gpus=gpus, gpu_type=GPUType.NVIDIA_A100_80GB_PCIe),
            ),
            volumes=(shared_memory_volume("30Gi"),),
        )
    )
    print(from_pod.logs())
