import torch


mapping_rf_to_pmpnn = {
    0: 0,    # ALA -> A
    1: 14,   # ARG -> R
    2: 11,   # ASN -> N
    3: 2,    # ASP -> D
    4: 1,    # CYS -> C
    5: 13,   # GLN -> Q
    6: 3,    # GLU -> E
    7: 5,    # GLY -> G
    8: 6,    # HIS -> H
    9: 7,    # ILE -> I
    10: 9,   # LEU -> L
    11: 8,   # LYS -> K
    12: 10,  # MET -> M
    13: 4,   # PHE -> F
    14: 12,  # PRO -> P
    15: 15,  # SER -> S
    16: 16,  # THR -> T
    17: 18,  # TRP -> W
    18: 19,  # TYR -> Y
    19: 17,  # VAL -> V
    20: 20,  # UNK -> X
    21: 20   # MAS -> X
}

# maps each RFDiffusion idx to the PMPNN index (so the amino-acids match)
# RFDiffusion_alphabet = "ARNDCQEGHILKMFPSTWYVXX"
# pMPNN_alphabet = "ACDEFGHIKLMNPQRSTVWYX"

def batch_process_pmpnnsample(Xpred_batch, batch):
    B = len(batch["seq_orig_raw"])  # number of samples
    X_list = []
    S_final_list = []
    mask_list = []
    chain_m_list = []
    new_residue_idx_list = []
    new_chain_labels_list = []

    device = batch["seq_orig_raw"].device  # Get the device from the input tensor

    # maps each RFDiffusion idx to the PMPNN index (so the amino-acids match)
    # RFDiffusion_alphabet = "ARNDCQEGHILKMFPSTWYVXX"
    # pMPNN_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    pmpnn_mapping_tensor = torch.tensor([ 0, 14, 11,  2,  1, 13,  3,  5,  6,  7,  9,  8, 10,  4, 12, 15, 16, 18,
        19, 17, 20, 20]).to(device)  # Move to the same device as the input tensors
    
    for i in range(B):
        # X: [L, 4, 3]
        X = Xpred_batch[i][:, :4, :]  
        # Original sequence: tensor of shape [L]
        S_orig = batch["seq_orig_raw"][i]
        # Remap each index using the mapping dictionary.
        # (We convert S_orig to a Python list and then back to a tensor.)
        S_new = pmpnn_mapping_tensor[S_orig]
        
        # Get chain IDs for this sample; assumed shape [L]
        chain_ids = batch["chain_id"][i]
        # Get unique chain ids and shuffle them
        unique_chains = torch.unique(chain_ids)
        shuffled_chains = unique_chains[torch.randperm(unique_chains.numel())]
        
        # For each chain in the shuffled order, get the indices where chain_ids equals that chain.
        indices = [torch.nonzero(chain_ids == c, as_tuple=True)[0] for c in shuffled_chains]
        new_order = torch.cat(indices)
        # Reorder the sequence based on the new chain order.
        S_final = S_new[new_order]
        
        # Compute a mask for valid residues from X.
        # X is [L, 4, 3]; we sum over the last two dims and check isfinite.
        mask = torch.isfinite(torch.sum(X, dim=(1, 2))).float().to(device)

        # For chain_m, use the diffusion_mask for this sample, reordering it similarly.
        # Here we assume diffusion_mask is boolean; invert it (~) then convert to float.
        chain_m = (~batch['mask_seq'][i][new_order]).float()
        
        # Now, compute new residue indices and new chain labels.
        new_residue_idx_parts = []
        new_chain_label_parts = []
        for new_chain_id, c in enumerate(shuffled_chains):
            # Get indices for residues belonging to chain "c"
            chain_idx = torch.nonzero(chain_ids == c, as_tuple=True)[0]
            block_length = chain_idx.size(0)
            # For this chain, assign residue indices: 100 * new_chain_id + position
            new_idx_block = 100 * new_chain_id + torch.arange(block_length, device=device)
            new_residue_idx_parts.append(new_idx_block)
            # New chain label: new_chain_id + 1 (labels start at 1)
            new_label_block = torch.full((block_length,), new_chain_id + 1,
                                         dtype=torch.long, device=device)
            new_chain_label_parts.append(new_label_block)
        
        new_residue_idx = torch.cat(new_residue_idx_parts)
        new_chain_labels = torch.cat(new_chain_label_parts)
        
        # Append per-sample results to lists.
        X_list.append(X)
        S_final_list.append(S_final)
        mask_list.append(mask)
        chain_m_list.append(chain_m)
        new_residue_idx_list.append(new_residue_idx)
        new_chain_labels_list.append(new_chain_labels)
    
    # Stack the lists along a new batch dimension.
    # Since each sample has the same number of residues L, this stacking works.
    X_batch = torch.stack(X_list, dim=0)               # [B, L, 4, 3]
    S_final_batch = torch.stack(S_final_list, dim=0)     # [B, L]
    mask_batch = torch.stack(mask_list, dim=0)           # [B, L]
    chain_m_batch = torch.stack(chain_m_list, dim=0)     # [B, L]
    new_residue_idx_batch = torch.stack(new_residue_idx_list, dim=0)  # [B, L]
    new_chain_labels_batch = torch.stack(new_chain_labels_list, dim=0)  # [B, L]
    
    return (X_batch, S_final_batch, mask_batch,
            chain_m_batch, new_residue_idx_batch, new_chain_labels_batch)
