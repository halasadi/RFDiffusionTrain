import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from rfdiffusion.diffusion import Diffuser
from rfdiffusion.chemical import seq2chars
from rfdiffusion.util_module import ComputeAllAtomCoords
from rfdiffusion.contigs import ContigMap
from rfdiffusion.inference import utils as iu, symmetry
from rfdiffusion.potentials.manager import PotentialManager
import logging
#import torch.nn.functional as nn
import torch.nn.functional as F

from rfdiffusion import util
from hydra.core.hydra_config import HydraConfig
import os

from rfdiffusion.model_input_logger import pickle_function_call
import sys

SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles


class Sampler():

    def __init__(self, conf: DictConfig):
        """
        Initialize sampler.
        Args:
            conf: Configuration.
        """
        super().__init__()

        self.initialized = False
        self.initialize(conf)


    def save_model(self, model_name: str):
        """Save the trained RosettaFold model to a checkpoint file."""
    
        if not model_name.endswith(".pt"):
            model_name += ".pt"  # Ensure proper file extension
    
        save_path = f"{SCRIPT_DIR}/../../models/{model_name}"
    
        # Move model to CPU before saving to avoid CUDA dependencies
        model = self.model.to('cpu')
    
        checkpoint = {
            'model_state_dict': model.state_dict(),  # Save model weights
            'config': self._conf,  # Save configuration for reproducibility
        }
    
        torch.save(checkpoint, save_path)
    
        print(f'Model successfully saved at {save_path}')


        
    def initialize(self, conf: DictConfig) -> None:
        """
        Initialize sampler.
        Args:
            conf: Configuration
        
        - Selects appropriate model from input
        - Assembles Config from model checkpoint and command line overrides

        """
        self._log = logging.getLogger(__name__)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using GPU for inference")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for inference")
    
        # Assign config to Sampler
        self._conf = conf 

        model_directory = f"{SCRIPT_DIR}/../../models"

        print(f"Reading models from {model_directory}")

        self.ckpt_path = f'{model_directory}/{conf.model_name}.pt'

        self.load_checkpoint()
        self.assemble_config_from_chk()
        # Now actually load the model weights into RF
        self.model = self.load_model()

        self.initialized=True

        self.diffuser_conf = self._conf.diffuser
        self.allatom = ComputeAllAtomCoords().to(self.device)

        
        
    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        # comment out for PyTorch lightning
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)
        #self.ckpt  = torch.load(self.ckpt_path, map_location='cpu')

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

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        
        # Read input dimensions from checkpoint.
        self.d_t1d=self._conf.preprocess.d_t1d
        self.d_t2d=self._conf.preprocess.d_t2d

        # don't need to the .to(self.device) because of pytorch lightning
        model = RoseTTAFoldModule(**self._conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, T=self._conf.diffuser.T).to(self.device)
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference')
            print(f'pickle_dir: {pickle_dir}')
        
        # don't need this because of pytorch lightning 
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def construct_contig(self, target_feats):
        """
        Construct contig class describing the protein to be generated
        """
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'potential_manager': self.potential_manager,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """
        Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """
        
        #######################
        ### Parse input pdb ###
        #######################

        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)

        ################################
        ### Generate specific contig ###
        ################################

        # Generate a specific contig from the range of possibilities specified at input

        self.contig_map = self.construct_contig(self.target_feats)
        self.mappings = self.contig_map.get_mappings()
        self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None,:]
        self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None,:]
        self.binderlen =  len(self.contig_map.inpaint)     

        ####################
        ### Get Hotspots ###
        ####################

        self.hotspot_0idx=iu.get_idx0_hotspots(self.mappings, self.ppi_conf, self.binderlen)


        #####################################
        ### Initialise Potentials Manager ###
        #####################################

        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.hotspot_0idx,
                                                  self.binderlen)

        ###################################
        ### Initialize other attributes ###
        ###################################

        xyz_27 = self.target_feats['xyz_27']
        mask_27 = self.target_feats['mask_27']
        seq_orig = self.target_feats['seq']
        L_mapped = len(self.contig_map.ref)
        contig_map=self.contig_map

        self.diffusion_mask = self.mask_str
        self.chain_idx=['A' if i < self.binderlen else 'B' for i in range(L_mapped)]
        
        ####################################
        ### Generate initial coordinates ###
        ####################################

        if self.diffuser_conf.partial_T:
            assert xyz_27.shape[0] == L_mapped, f"there must be a coordinate in the input PDB for \
                    each residue implied by the contig string for partial diffusion.  length of \
                    input PDB != length of contig string: {xyz_27.shape[0]} != {L_mapped}"
            assert contig_map.hal_idx0 == contig_map.ref_idx0, f'for partial diffusion there can \
                    be no offset between the index of a residue in the input and the index of the \
                    residue in the output, {contig_map.hal_idx0} != {contig_map.ref_idx0}'
            # Partially diffusing from a known structure
            xyz_mapped=xyz_27
            atom_mask_mapped = mask_27
        else:
            # Fully diffusing from points initialised at the origin
            # adjust size of input xt according to residue map
            xyz_mapped = torch.full((1,1,L_mapped,27,3), np.nan)
            xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0,...]
            xyz_motif_prealign = xyz_mapped.clone()
            motif_prealign_com = xyz_motif_prealign[0,0,:,1].mean(dim=0)
            self.motif_com = xyz_27[contig_map.ref_idx0,1].mean(dim=0)
            xyz_mapped = get_init_xyz(xyz_mapped).squeeze()
            # adjust the size of the input atom map
            atom_mask_mapped = torch.full((L_mapped, 27), False)
            atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T, "Partial_T must be less than T"
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input+1)

        #################################
        ### Generate initial sequence ###
        #################################

        seq_t = torch.full((1,L_mapped), 21).squeeze() # 21 is the mask token
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
        
        # Unmask sequence if desired
        if self._conf.contigmap.provide_seq is not None:
            seq_t[self.mask_seq.squeeze()] = seq_orig[self.mask_seq.squeeze()] 

        seq_t[~self.mask_seq.squeeze()] = 21
        seq_t    = torch.nn.functional.one_hot(seq_t, num_classes=22).float() # [L,22]
        seq_orig = torch.nn.functional.one_hot(seq_orig, num_classes=22).float() # [L,22]

        fa_stack, xyz_true = self.diffuser.diffuse_pose(
            xyz_mapped,
            torch.clone(seq_t),
            atom_mask_mapped.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list)
        xT = fa_stack[-1].squeeze()[:,:14,:]
        xt = torch.clone(xT)

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=self.mask_seq.squeeze())


        self._log.info(f'Sequence init: {seq2chars(torch.argmax(seq_t, dim=-1))}')
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None

        #########################################
        ### Parse ligand for ligand potential ###
        #########################################

        if self.potential_conf.guiding_potentials is not None:
            if any(list(filter(lambda x: "substrate_contacts" in x, self.potential_conf.guiding_potentials))):
                assert len(self.target_feats['xyz_het']) > 0, "If you're using the Substrate Contact potential, \
                        you need to make sure there's a ligand in the input_pdb file!"
                het_names = np.array([i['name'].strip() for i in self.target_feats['info_het']])
                xyz_het = self.target_feats['xyz_het'][het_names == self._conf.potentials.substrate]
                xyz_het = torch.from_numpy(xyz_het)
                assert xyz_het.shape[0] > 0, f'expected >0 heteroatoms from ligand with name {self._conf.potentials.substrate}'
                xyz_motif_prealign = xyz_motif_prealign[0,0][self.diffusion_mask.squeeze()]
                motif_prealign_com = xyz_motif_prealign[:,1].mean(dim=0)
                xyz_het_com = xyz_het.mean(dim=0)
                for pot in self.potential_manager.potentials_to_apply:
                    pot.motif_substrate_atoms = xyz_het
                    pot.diffusion_mask = self.diffusion_mask.squeeze()
                    pot.xyz_motif = xyz_motif_prealign
                    pot.diffuser = self.diffuser
        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t, binderlen, hotspot_res, mask_str, sidechain_input, rf, d_t1d, con_ref_pdb_idx, hal_idx0, repack=False):
        
        """
        Function to prepare inputs to diffusion model
        
            seq (L,22) one-hot sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)
        
            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)

                MODEL SPECIFIC:
                - contacting residues: for ppi. Target residues in contact with binder (1)
                - empty feature (legacy) (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (1, L, L, 45)
                - last plane is block adjacency
    """

        L = seq.shape[0]        
        
        ##################
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,48))
        msa_masked[:,:,:,:22] = seq[None, None]
        msa_masked[:,:,:,22:44] = seq[None, None]
        msa_masked[:,:,0,46] = 1.0
        msa_masked[:,:,-1,47] = 1.0

        ################
        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,25))
        msa_full[:,:,:,:22] = seq[None, None]
        msa_full[:,:,0,23] = 1.0
        msa_full[:,:,-1,24] = 1.0

        ###########
        ### t1d ###
        ########### 

        # Here we need to go from one hot with 22 classes to one hot with 21 classes (last plane is missing token)
        t1d = torch.zeros((1,1,L,21))

        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx,21] == 1:
                seqt1d[idx,20] = 1
                seqt1d[idx,21] = 0
        
        t1d[:,:,:,:21] = seqt1d[None,None,:,:21]
        

        # Set timestep feature to 1 where diffusion mask is True, else 1-t/T
        timefeature = torch.zeros((L)).float()
        timefeature[mask_str.squeeze()] = 1
        timefeature[~mask_str.squeeze()] = 1 - t/self.T
        timefeature = timefeature[None,None,...,None]

        t1d = torch.cat((t1d, timefeature), dim=-1).float()
        
        #############
        ### xyz_t ###
        #############
        if sidechain_input:
            xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
        else:
            xyz_t[~mask_str.squeeze(),3:,:] = float('nan')

        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)

        ###########
        ### t2d ###
        ###########
        t2d = xyz_to_t2d(xyz_t)
        
        ###########      
        ### idx ###
        ###########
        idx = torch.tensor(rf)[None]
        #idx = torch.tensor(args['contig_map'].rf)[None]

        ###############
        ### alpha_t ###
        ###############
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1, L, 27, 3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        #put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        
        ######################
        ### added_features ###
        ######################        
        if d_t1d >= 24: # add hotspot residues
            hotspot_tens = torch.zeros(L).float()
            if len(hotspot_res) == 0:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots.\
                         If you're doing monomer diffusion this is fine")
                hotspot_idx=[]
            else:
                hotspots = [(i[0],int(i[1:])) for i in hotspot_res]
                hotspot_idx=[]
                for i,res in enumerate(con_ref_pdb_idx):
                    if res in hotspots:
                        #hotspot_idx.append(args['contig_map'].hal_idx0[i])
                        hotspot_idx.append(hal_idx0[i])
                hotspot_tens[hotspot_idx] = 1.0

            # Add blank (legacy) feature and hotspot tensor
            t1d=torch.cat((t1d, torch.zeros_like(t1d[...,:1]), hotspot_tens[None,None,...,None].to(self.device)), dim=-1)

        return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        

class SelfConditioning(Sampler):
    """
    Model Runner for self conditioning
    pX0[t+1] is provided as a template input to the model at time t
    """

    def sample_step(self, *, t, x_t, seq_init, final_step, binderlen, 
        hotspot_res, mask_str, sidechain_input, 
        rf, d_t1d, con_ref_pdb_idx, hal_idx0,
        diffusion_mask):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The sequence to the next step (== seq_init)
            plddt: (L, 1) Predicted lDDT of x0.
        '''

        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_init, x_t, t, binderlen, hotspot_res, mask_str, sidechain_input, rf, d_t1d, con_ref_pdb_idx, hal_idx0)

        B,N,L = xyz_t.shape[:3]
        

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):   
            zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
            xyz_t = torch.cat((self.prev_pred.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]
            t2d_44   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
        else:
            xyz_t = torch.zeros_like(xyz_t)
            t2d_44   = torch.zeros_like(t2d[...,:44])
        # No effect if t2d is only dim 44
        t2d[...,:44] = t2d_44

        ## this is where to use the saved t2d.
        ## 06/18/2025 later (Hussein)

        ####################
        ### Forward Pass ###
        ####################

    
        with torch.no_grad():
            msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                msa_full,
                                seq_in,
                                xt_in,
                                idx_pdb,
                                t1d=t1d,
                                t2d=t2d,
                                xyz_t=xyz_t,
                                alpha_t=alpha_t,
                                msa_prev = None,
                                pair_prev = None,
                                state_prev = None,
                                t=torch.tensor(t),
                                return_infer=True,
                                motif_mask=diffusion_mask.unsqueeze(0).to(self.device))   


        self.prev_pred = torch.clone(px0)

        # prediction of X0
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]
        
        ###########################
        ### Generate Next Input ###
        ###########################

        seq_t_1 = torch.clone(seq_init)
        if t > final_step:
            x_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=mask_str.squeeze(),
                align_motif=self.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
            self._log.info(
                    f'Timestep {t}, input to next step: { seq2chars(torch.argmax(seq_t_1, dim=-1).tolist())}')
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            px0 = px0.to(x_t.device)


        return px0, x_t_1, seq_t_1, plddt

