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

**Losses:** RFDiffusion employs two primary loss functions: dFrame and L2D. The dFrame loss was implemented following the methodology described in the supplementary materials. For the L2D loss, inconsistencies were found between the binning strategy outlined in the supplement and the actual code implementation (specifically in the PARAMS variable). To resolve this discrepancy, I opted to follow the implementation from the original RosettaFold2 repository, though this choice may introduce potential inaccuracies given the conflicting documentation.

**Generalized Templates:** I removed TCR/peptide/MHC specific parts and sketched out generalized code for `dataset.py`, `train_rfdiffusion.py`, `test_diffusion.py` that I have not tested. You'll need to adapt them for your protein system.

**Compute Enviornment** This implementation will not build out of the box and requires adaptation to your specific computing environment.

## Files

Require customization for your specific use case
- `dataset.py` - Generalized protein dataset template
- `workflows/train_rfdiffusion.py` - Main training pipeline template
- `test_diffusion.py` - Runner script


- `custom` - includes the custom losses and utils

## A place to Start

1. **Customize the templates:**
   Update `dataset.py` and `workflows/train_rfdiffusion.py` to suit your custom problem
   - E.g. Define your protein regions and diffusion strategy. Modify paths and project settings in `training_script.py`

2. **Configure for your cluster:**
   - Configure DockerFile + pyproject.toml + pixi enviornment
   - Update storage paths and authentication
   - Set-up WandB if desired.

4. **Launch training:**
   ```bash
   python workflows/train_rfdiffusion.py
   ```

## Requirements

- PyTorch Lightning
- Weights & Biases  
- DeepSpeed
- RFDiffusion (original package)

## Origin

Adapted from RFDiffusion by Hussein Al-Asadi @ Adaptive Biotechnologies, with special thanks to Andrew FigPope @ Adaptive Biotechnologies for writing the PodMapper for GPU training, helping with DeepSeed, and setting up the Pixi Enviornment.
