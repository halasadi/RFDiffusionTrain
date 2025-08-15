import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler

from rfdiffusion.inference.utils import process_target
from rfdiffusion.contigs import ContigMap
from rfdiffusion.kinematics import xyz_to_t2d
import random
import glob as glob
import itertools

from torch.utils.data._utils.collate import default_collate

"""
Generalized Protein Structure Diffusion Dataset Template

This code has been generalized from a specific implementation to serve as 
a template for various protein structure diffusion tasks. You will need to adapt 
this template to your specific use case and test it with your data.

"""


def my_collate(batch):
    """
    Custom collate function for handling batches with non-uniform data.
    Modify this based on your specific data structure needs.
    """
    # batch is a list of dicts; default_collate will stack everything
    # except the keys we pop out first
    out = default_collate(
        [{k: v for k, v in x.items() if k not in ["special_indices"]} for x in batch]
    )
    # put back special_indices as a python list (of tensors or lists)
    out["special_indices"] = [x["special_indices"] for x in batch]
    return out


class ProteinStructureDataset(Dataset):
    """
    Template dataset for protein structure diffusion tasks.
    
    This is a minimal template - you'll need to customize it for your specific use case.
    Key areas to modify are marked with TODO comments.
    
    """
    
    def __init__(self, target_dir, decoy_dir, meta_info_path, crop_size=384, split="train"):
        """
        Args:
            target_dir (str): Directory containing target PDB files
            decoy_dir (str): Directory containing decoy/reference PDB files  
            meta_info_path (str): Path to the metadata CSV file
            crop_size (int): Number of residues to keep after cropping
            split (str): Data split to use ('train', 'val', 'test')
        """
        
        # Load and filter metadata
        df = pd.read_csv(meta_info_path, sep=",")
        df = df[df["split"] == split].reset_index(drop=True)
        self.meta_info_df = df
        
        # TODO: Define your region lengths and parameters
        self.REGION_A_MAX_LEN = 50  # Example: maximum length of region A
        self.REGION_B_MAX_LEN = 100  # Example: maximum length of region B
        self.CROP_SIZE = crop_size
        
        # TODO: Define your chain mapping if using multi-chain structures
        self.chain_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        
        self.target_dir = target_dir
        self.decoy_dir = decoy_dir

    def __len__(self):
        return len(self.meta_info_df)

    def _pad_list(self, lst, target_length, pad_value):
        """Pad list lst to target_length using pad_value."""
        return lst + [pad_value] * (target_length - len(lst))

    def __getitem__(self, idx):
        # 1. Read metadata and build file paths
        row = self.meta_info_df.iloc[idx]
        
        # TODO: Replace with your actual column names
        structure_id = row["structure_id"]
        decoy_prefix = row["decoy_prefix"]  # or however you identify decoy structures
        
        # Build file paths
        target_pdb_path = glob.glob(os.path.join(self.target_dir, f"{structure_id}*.pdb"))[0]
        decoy_pdb_path = glob.glob(os.path.join(self.decoy_dir, f"{decoy_prefix}*.pdb"))[0]
        
        # TODO: Define your sequence regions and lengths
        # Example: replace with your actual sequence columns
        region_a_len = len(row['sequence_region_a'])
        region_b_len = len(row['sequence_region_b'])
        fixed_region_len = len(row['fixed_sequence'])
        total_len = region_a_len + region_b_len + fixed_region_len
        
        # TODO: Define which regions are fixed vs. diffused
        # Example contig configuration:
        # - Regions A and B are diffused (no prefix)
        # - Fixed region is kept constant (D prefix)
        fixed_start = total_len - fixed_region_len
        fixed_end = total_len - 1
        
        contig_conf = [f'{region_a_len}-{region_a_len}/0 {region_b_len}-{region_b_len}/0 D{fixed_start}-{fixed_end}']
        provide_seq = [f'{region_a_len}-{fixed_end}']
        
        # 2. Process target + decoy features and construct contig map
        target_feats = process_target(target_pdb_path, parse_hetatom=False, center=False)
        contig_map = ContigMap(target_feats, contig_conf, provide_seq=provide_seq)
        decoy_feats = process_target(decoy_pdb_path, parse_hetatom=False, center=False)
        
        # 3. Build masks and convert arrays/tensors
        mask_seq = torch.from_numpy(contig_map.inpaint_seq)[None, :]  # Sequence mask
        mask_str = torch.from_numpy(contig_map.inpaint_str)[None, :]  # Structure mask
        diffusion_mask = mask_str.squeeze()
        
        xyz_27 = target_feats['xyz_27']
        seq_orig = target_feats['seq']
        L_mapped = len(seq_orig)
        seq_orig_raw = seq_orig.clone()
        
        # TODO: Define your hotspot residues or important regions
        hotspot_res = [f'A{i+1}' for i in range(region_a_len)]  # Example
        hotspot_res.extend([''] * (self.REGION_A_MAX_LEN - len(hotspot_res)))
        
        rf = contig_map.rf
        con_ref_pdb_idx = contig_map.con_ref_pdb_idx
        hal_idx0 = contig_map.hal_idx0

        # Handle chain IDs if multi-chain
        chain_id = [x[0] for x in target_feats['pdb_idx']]
        numeric_chain_ids = [self.chain_mapping.get(c, 0) for c in chain_id]
        chain_id_tensor = torch.tensor(numeric_chain_ids)

        # 4. Center the structure at your reference point
        xyz_27 = self._center_structure(xyz_27, diffusion_mask)
        L = xyz_27.shape[0]

        # 5. Prepare the sequence
        seq_t = torch.full((1, L_mapped), 21).squeeze()  # 21 is the mask token
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
        seq_t[mask_seq.squeeze()] = seq_orig[mask_seq.squeeze()]  # Unmask sequence
        seq_t[~mask_seq.squeeze()] = 21
        seq_t = torch.nn.functional.one_hot(seq_t, num_classes=22).float()
        seq_orig = torch.nn.functional.one_hot(seq_orig, num_classes=22).float()

        # 6. Prepare 2D distance representation from decoy
        decoy_xyz_27 = decoy_feats['xyz_27']
        mask_27 = decoy_feats['mask_27']
        decoy_xyz_27 = self._center_structure(decoy_xyz_27, diffusion_mask)
        decoy_xyz_t = decoy_xyz_27[:, :14, :].clone()
        decoy_xyz_t[~mask_str.squeeze(), 3:, :] = float('nan')
        decoy_xyz_t = decoy_xyz_t[None, None]
        decoy_xyz_t = torch.cat((decoy_xyz_t, torch.full((1, 1, L, 13, 3), float('nan'))), dim=3)
        t2d_decoy = xyz_to_t2d(decoy_xyz_t)
        t2d_decoy = t2d_decoy.squeeze(0)

        # TODO: Calculate special indices for your application
        # Example: indices of specific structural elements you want to preserve
        special_indices = self._get_special_indices(row, region_a_len, region_b_len)

        # --- CROPPING ---
        # 7. Determine crop indices based on your strategy
        must_keep_indices = self._get_must_keep_indices(
            row, region_a_len, region_b_len, special_indices
        )
        
        # Ensure we don't exceed crop size
        all_indices = set(range(L))
        remaining_indices = list(all_indices - set(must_keep_indices))
        to_sample = self.CROP_SIZE - len(must_keep_indices)
        
        assert to_sample >= 0, f"Must-keep regions ({len(must_keep_indices)}) exceed CROP_SIZE ({self.CROP_SIZE})"
        assert to_sample <= len(remaining_indices), "Not enough remaining residues to reach CROP_SIZE"
        
        chosen_rest = random.sample(remaining_indices, to_sample) if to_sample > 0 else []
        crop_idx = must_keep_indices + chosen_rest
        crop_idx.sort()
        assert len(crop_idx) == self.CROP_SIZE

        # 8. Crop all tensors
        xyz_27 = xyz_27[crop_idx, :, :]
        mask_27 = mask_27[crop_idx, :]
        seq_t = seq_t[crop_idx, :]
        seq_orig = seq_orig[crop_idx, :]
        diffusion_mask = diffusion_mask[crop_idx]
        mask_str = mask_str[:, crop_idx]
        mask_seq = mask_seq.squeeze()[crop_idx]
        seq_orig_raw = seq_orig_raw[crop_idx]
        chain_id_tensor = chain_id_tensor[crop_idx]
        decoy_xyz_27 = decoy_xyz_27[crop_idx, :, :]

        # 9. Build mapping and remap indices
        old_to_new = {old: new for new, old in enumerate(crop_idx)}
        
        hal_idx0_new = [old_to_new[i] for i in hal_idx0 if i in crop_idx]
        hal_idx0_new.sort()
        
        con_ref_pdb_idx_new = []
        for i, old_idx in enumerate(hal_idx0):
            if old_idx in crop_idx:
                new_idx = old_to_new[old_idx]
                chain, old_resnum = con_ref_pdb_idx[i]
                con_ref_pdb_idx_new.append((chain, new_idx))

        # Pad to fixed length
        hal_idx0_new = self._pad_list(hal_idx0_new, self.REGION_B_MAX_LEN, -1)
        con_ref_pdb_idx_new = self._pad_list(con_ref_pdb_idx_new, self.REGION_B_MAX_LEN, ('', -1))

        # Crop 2D matrix
        crop_idx_tensor = torch.tensor(crop_idx)
        t2d_cropped = t2d_decoy[:, crop_idx_tensor.unsqueeze(1), crop_idx_tensor.unsqueeze(0), :]

        # Remap special indices
        new_special_indices = [old_to_new[i] for i in special_indices if i in old_to_new]

        # Update rf
        rf_new = [rf[i] for i in crop_idx]

        return {
            "L_mapped": self.CROP_SIZE,
            "xyz_mapped": xyz_27,
            "decoy_xyz_mapped": decoy_xyz_27,
            "atom_mask_mapped": mask_27,
            "seq_t": seq_t,
            "seq_orig": seq_orig,
            "seq_orig_raw": seq_orig_raw,
            "diffusion_mask": diffusion_mask,
            "hotspot_res": hotspot_res,
            "binderlen": torch.sum(mask_str == False),
            "mask_str": mask_str,
            "mask_seq": mask_seq,
            "rf": rf_new,
            "con_ref_pdb_idx": con_ref_pdb_idx_new,
            "hal_idx0": hal_idx0_new,
            "special_indices": new_special_indices,
            "t2d_decoy": t2d_cropped,
            "chain_id": chain_id_tensor
        }

    def _center_structure(self, xyz, reference_mask):
        """
        Center the structure at a reference point.
        
        Args:
            xyz: Coordinate tensor [L, 27, 3]
            reference_mask: Boolean mask indicating reference region
        
        Returns:
            Centered coordinate tensor
        """
        if torch.sum(reference_mask) != 0:
            # Center at CA atoms of reference region
            reference_com = xyz[reference_mask, 1, :].mean(dim=0)
            xyz = xyz - reference_com
        else:
            # Center at overall CA center of mass
            xyz = xyz - xyz[:, 1, :].mean(dim=0)
        return xyz

    def _get_special_indices(self, row, region_a_len, region_b_len):
        """
        TODO: Implement logic to identify special structural elements.
        
        This should return indices of residues that are important for your 
        specific application (e.g., binding sites, catalytic residues, etc.)
        
        Args:
            row: Metadata row for this structure
            region_a_len: Length of region A
            region_b_len: Length of region B
        
        Returns:
            List of indices for special structural elements
        """
        # Example implementation - replace with your logic
        special_indices = []
        
        # Example: keep important residues from region A
        region_a_start = 0
        region_a_end = region_a_len
        # Add logic to identify important residues...
        
        # Example: keep important residues from region B  
        region_b_start = region_a_len
        region_b_end = region_a_len + region_b_len
        # Add logic to identify important residues...
        
        return special_indices

    def _get_must_keep_indices(self, row, region_a_len, region_b_len, special_indices):
        """
        TODO: Define which residues must be kept during cropping.
        
        This should return indices of residues that are critical for your
        application and should never be cropped out.
        
        Args:
            row: Metadata row for this structure
            region_a_len: Length of region A
            region_b_len: Length of region B
            special_indices: Indices of special structural elements
        
        Returns:
            List of indices that must be preserved during cropping
        """
        must_keep_indices = []
        
        # Example: always keep certain regions
        must_keep_indices.extend(list(range(region_a_len)))  # Keep all of region A
        must_keep_indices.extend(special_indices)  # Keep special elements
        
        # Remove duplicates and sort
        must_keep_indices = sorted(list(set(must_keep_indices)))
        
        return must_keep_indices
