# RFDiffusion Training Implementation

RFDiffusion provides inference scripts but no training code. This repository implements the complete training protocol from the supplement.

## What I Built

- **Implemented training protocol** following the RFDiffusion supplement
- **Loss functions** custom backbone frame loss and L2D distance prediction  
- **PyTorch Lightning wrapper** for clean training loops
- **Weights & Biases integration** for experiment tracking
- **DeepSpeed support** for 16-bit training (vs original 32-bit) - 2x memory reduction
- **Azure Blob Storage integration** for large-scale data handling

## Example Usage: TCR-pMHC Generation

Used this protocol to fine-tune RFDiffusion for TCR-pMHC complex generation. Shown here is a random example from a hold-out set.

**Before Training:**

<img src="gifs/\ basemodel.gif" alt="Before Training" width="400"/>

**After Training:** 

<img src="gifs/\ trainedmodel.gif" alt="After Training" width="400"/>

(Note: the peptide is in red)

## Key Implementation Notes

**Losses:** There are two major losses in RFDiffusion: dFrame and L2D. Followed the supplementary section in coming the dFrame Loss. While for the L2D loss, the supplement seemed to be in-consistent in binning strategy compared to the code (see PARAMS variable), so I followed the original RosettaFold2 repo implementation instead.

**Generalized Templates:** I removed TCR/peptide/MHC specific parts and sketched out generalized code for dataloaders that I have not tested. You'll need to adapt them for your protein system.

**Compute Enviornment** You will need to adapt the code to suit your compute enviornment.

## Files

Require customization for your specific use case
- `dataset.py` - Generalized protein dataset template
- `training_script.py` - Main training pipeline template
- `test_diffusion.py` - Runner script


- `custom` - includes the custom losses and utils

## Where to Start

1. **Customize the templates:**
   - Update metadata column names in `dataset.py`
   - Define your protein regions and diffusion strategy
   - Modify paths and project settings in `training_script.py`

2. **Configure for your cluster:**
   - Update storage paths and authentication
   - Adjust GPU counts and memory settings
   - Set WandB project name and entity
   - Configure DockerFile + pyproject.toml for your enviornment

3. **Launch training:**
   ```bash
   python training_script.py
   ```

## Requirements

- PyTorch Lightning
- Weights & Biases  
- DeepSpeed
- RFDiffusion (original package)

## Origin

Adapted from RFDiffusion by Hussein Al-Asadi @ Adaptive Biotechnologies, with special thanks to Andrew FigPope @ Adaptive Biotechnologies for writing the PodMapper for GPU training, helping with DeepSeed, and setting up the Pixi Enviornment.
