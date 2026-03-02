import os
import sys
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class NSD_Imagery_Dataset(Dataset):
    def __init__(self, args):
        """
        Dataset for NSD imagery experiment data with Schaefer parcellation
        
        Parameters:
        -----------
        args : SimpleNamespace
            Should contain the following:
            - subj: Subject number
            - neuro_dir: Path to neuroimaging data directory
            - behavior_dir: Path to behavioral data directory
            - parcel_dir: Path to parcellation directory
            - parcel_indices: Dictionary with 'lh' and 'rh' keys containing parcel indices
            - task: 'vis' or 'img' to specify which task data to load
            - img_dir: (Optional) Path to image directory for accessing stimulus images
            - topk: (Optional) Number of top parcels to use
            - gen_size: (Optional) Size to resize images to, default 512
        """
        self.subj = args.subj
        
        # Set tasks based on input
        if args.task == 'vis':
            self.tasks = ['visA', 'visB', 'visC']
        elif args.task == 'img':
            self.tasks = ['imgA_1', 'imgB_1', 'imgC_1', 'imgA_2', 'imgB_2', 'imgC_2']
        elif args.task == 'img_1':
            self.tasks = ['imgA_1', 'imgB_1', 'imgC_1'] 
        elif args.task == 'img_2':
            self.tasks = ['imgA_2', 'imgB_2', 'imgC_2']
        else:
            raise ValueError("Task must be 'vis' or 'img'")

        # Set paths and parameters
        self.neuro_dir = Path(args.neuro_dir)
        self.behavior_dir = Path(args.behavior_dir)
        self.topk = getattr(args, 'topk', None)
        self.gen_size = getattr(args, 'gen_size', 512)
        self.selected_parcel_idx = args.parcel_indices
        
        # Image related paths and mapping
        self.img_dir = Path(getattr(args, 'img_dir', '')) if hasattr(args, 'img_dir') else None
        self.has_images = self.img_dir is not None and self.img_dir.exists()
        
        if self.has_images:
            self.stimuli_dir = self.img_dir / "imagery_stimuli"
            
        # Load Schaefer parcels
        self.parcels = {}
        self.max_voxels = 0
        self._load_parcels(args)
        
        # Load and prepare data
        self.neuro_data = self.load_neuro_data()
        self.task2brain_index_dict = self._get_task2brain_index_dict()
        self.integrated_neuro_data = self._merge_neuro_data()
        
        # Setup image mapping and transforms if images are available
        if self.has_images:
            self.condition2imgfile = self._get_condition2imgfile()
            
            # Transforms for image preprocessing
            self.ipadapter_transform = transforms.Compose([
                transforms.Resize(self.gen_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [-1, 1] range
            ])

            self.img_transform = transforms.Compose([
                transforms.Resize(self.gen_size),
                transforms.ToTensor(),
                # No normalization for encoder images, keeping [0, 1] range
            ])

    def _get_task2brain_index_dict(self):
        """Create mapping from task name to brain data indices"""
        task_order = ['visA', 'attA', 'imgA_1', 'visB', 'attB', 'imgB_1', 'visC', 'attC', 'imgC_1', 'imgA_2', 'imgB_2', 'imgC_2']
        num_trials = 48
        task2brain_index_dict = {}
        current_index = 0

        for task in task_order:
            if task.startswith('att'):
                current_index += num_trials * 2  # Skip attention tasks
            else:
                task2brain_index_dict[task] = (current_index, current_index + num_trials)
                current_index += num_trials

        return task2brain_index_dict
    
    def load_neuro_data(self):
        """Load neuroimaging data for both hemispheres"""
        data = {}
        for hemi in ['lh', 'rh']:
            neuro_data_path = self.neuro_dir / f"subj{self.subj:02}/fsaverage/nsdimagerybetas_fithrf_GLMdenoise_RR/{hemi}.betas_nsdimagery.mgh"
            mgh = nib.load(neuro_data_path)
            data[hemi] = mgh.get_fdata().squeeze()
        return data

    def load_behavioral_data(self, task=None):
        """Load behavioral data for a specific task"""
        behavioral_data_path = self.behavior_dir / f"nsdimagery_subj{self.subj:02}_{task}.tsv"
        behavioral_data = pd.read_csv(behavioral_data_path, sep="\t")
        return behavioral_data

    def _merge_neuro_data(self):
        """Merge neuroimaging data across tasks and organize by condition"""
        data = {}
        if len(self.tasks) == 3:
            for task in self.tasks:
                behavioral_data = self.load_behavioral_data(task=task)
                cond_groups = behavioral_data.groupby("CONDITION").indices
                
                for hemi in ['lh', 'rh']:
                    neuro_data_task = self.neuro_data[hemi][:,self.task2brain_index_dict[task][0]:self.task2brain_index_dict[task][1]]
                    
                    for cond, idx in cond_groups.items():
                        neuro_data_cond = neuro_data_task[:, idx]
                        neuro_data_cond = neuro_data_cond.mean(axis=-1)  # Average across repetitions
                        
                        data[cond] = data.get(cond, {})
                        data[cond][hemi] = neuro_data_cond
        elif len(self.tasks) == 6:
            for (session_1, session_2) in zip(['imgA_1', 'imgB_1', 'imgC_1'], ['imgA_2', 'imgB_2', 'imgC_2']):
                behavioral_data_1 = self.load_behavioral_data(task=session_1)
                cond_groups_1 = behavioral_data_1.groupby("CONDITION").indices

                behavioral_data_2 = self.load_behavioral_data(task=session_2)
                cond_groups_2 = behavioral_data_2.groupby("CONDITION").indices
                
                common_conditions = set(cond_groups_1.keys()) & set(cond_groups_2.keys())

                for hemi in ['lh', 'rh']:
                    neuro_data_task_1 = self.neuro_data[hemi][:, self.task2brain_index_dict[session_1][0]:self.task2brain_index_dict[session_1][1]]
                    neuro_data_task_2 = self.neuro_data[hemi][:, self.task2brain_index_dict[session_2][0]:self.task2brain_index_dict[session_2][1]]
                    
                    for cond in common_conditions:
                        idx1 = cond_groups_1[cond]
                        idx2 = cond_groups_2[cond]
                        
                        neuro_data_cond_1 = neuro_data_task_1[:, idx1]
                        neuro_data_cond_2 = neuro_data_task_2[:, idx2]
                        neuro_data_cond = np.concatenate((neuro_data_cond_1, neuro_data_cond_2), axis=-1)
                        neuro_data_cond = neuro_data_cond.mean(axis=-1)  # Average across repetitions
                        
                        data[cond] = data.get(cond, {})
                        data[cond][hemi] = neuro_data_cond
        else:
            raise ValueError("Tasks should be either 3 (vis, img_1, img_2) or 6 (img)")

        return data

    def _load_parcels(self, args):
        """Load Schaefer parcellation and determine max voxels per parcel"""
        parcel_path = Path(args.parcel_dir)
        
        for hemi in ["lh", "rh"]:
            self.parcels[hemi] = torch.load(
                parcel_path / f"{hemi}_labels_s{self.subj:02}.pt",
                weights_only=True,
            )[1:]  # Skip the first parcel because it is medial wall
            
            # Compute the max number of voxels across selected parcels
            for parcel_idx in self.selected_parcel_idx[hemi]:
                voxel_count = len(self.parcels[hemi][parcel_idx])
                self.max_voxels = max(self.max_voxels, voxel_count)
                
        print(f"Maximum voxels per parcel: {self.max_voxels}")
    
    def extract_and_pad(self, fmri_data, hemi):
        """Extract voxels from selected parcels and pad to max_voxels"""
        out = []
        for idx in self.selected_parcel_idx[hemi]:
            voxel_idxs = self.parcels[hemi][idx]
            roi = torch.tensor(fmri_data[voxel_idxs])
            
            if roi.shape[0] < self.max_voxels:
                pad = torch.zeros(self.max_voxels - roi.shape[0])
                roi = torch.cat([roi, pad])
            else:
                roi = roi[:self.max_voxels]
                
            out.append(roi)
            
        return torch.stack(out)  # shape: [num_parcels, max_voxels]
    
    def get_selected_parcel_info(self):
        """
        Get detailed information about selected parcels.
        
        Returns:
            dict: Dictionary containing parcel information including indices, sizes, and voxel counts
        """
        info = {
            "lh": {
                "parcel_indices": self.selected_parcel_idx["lh"].tolist() if hasattr(self.selected_parcel_idx["lh"], 'tolist') else list(self.selected_parcel_idx["lh"]),
                "voxel_counts": [],
                "total_voxels": 0
            },
            "rh": {
                "parcel_indices": self.selected_parcel_idx["rh"].tolist() if hasattr(self.selected_parcel_idx["rh"], 'tolist') else list(self.selected_parcel_idx["rh"]),
                "voxel_counts": [],
                "total_voxels": 0
            }
        }
        
        for hemi in ["lh", "rh"]:
            for parcel_idx in self.selected_parcel_idx[hemi]:
                voxel_count = len(self.parcels[hemi][parcel_idx])
                info[hemi]["voxel_counts"].append(voxel_count)
                info[hemi]["total_voxels"] += voxel_count
        
        info["max_voxels_per_parcel"] = self.max_voxels
        info["total_parcels"] = len(self.selected_parcel_idx["lh"]) + len(self.selected_parcel_idx["rh"])
        
        return info

    def _get_condition2imgfile(self):
        """Map condition names to image filenames from the stimuli_info.csv file"""
        import pandas as pd
        df = pd.read_csv(self.img_dir / "stimuli_info.csv", header=0)
        condition2imgfile = {}
        for _, row in df.iterrows():
            condition = row["condition"]
            img_path = row["file_name"]
            condition2imgfile[condition] = img_path
        return condition2imgfile

    def __len__(self):
        """Return the number of conditions in the dataset"""
        return len(list(self.integrated_neuro_data.keys()))
    
    def __getitem__(self, idx):
        """Get data for a specific condition by index"""
        # Get the condition name by index
        cond = list(self.integrated_neuro_data.keys())[idx]
        fmri_data = self.integrated_neuro_data[cond]
        
        # Process brain data
        lh_fmri = fmri_data["lh"]
        rh_fmri = fmri_data["rh"]

        lh_ = self.extract_and_pad(
            lh_fmri, hemi="lh"
        )
        rh_ = self.extract_and_pad(
            rh_fmri, hemi="rh"
        )

        # Initialize image variables
        img_encoder = None
        img_ipadapter = None

        # Load and process images if available
        if self.has_images and cond in self.condition2imgfile:
            try:
                import PIL.Image
                img_name = self.condition2imgfile[cond]
                img_path = self.stimuli_dir / img_name

                if img_path.exists():
                    img = PIL.Image.open(img_path).convert("RGB")
                    img_ipadapter = self.ipadapter_transform(img)
                    img_encoder = self.img_transform(img)
            except Exception as e:
                print(f"Error loading image for condition {cond}: {e}")
                # Fallback to None if image loading fails

        return {
            "img_encoder": img_encoder,
            "img_ipadapter": img_ipadapter,
            "text_input_ids": torch.zeros(1, dtype=torch.long),
            "brain_lh_f": lh_,  # shape: [num_parcels, max_voxels]
            "brain_rh_f": rh_,  # shape: [num_parcels, max_voxels]
            "condition": cond,  # Return condition name for reference
        }
    
    def get_condition_names(self):
        """Get a list of all condition names in the dataset"""
        return list(self.integrated_neuro_data.keys())
    
    @property
    def num_parcels(self):
        """Get the total number of parcels across both hemispheres"""
        return len(self.selected_parcel_idx["lh"]) + len(self.selected_parcel_idx["rh"])

if __name__ == "__main__":
    # Test code to demonstrate usage
    print("Testing NSD_Imagery_Dataset")
    
    import argparse
    from types import SimpleNamespace
    import os
    from dataset import nsd_topk_parcel_dataset
    # from transformers import CLIPTokenizer
    subj=1
    print("Setting up test configuration for subject ", subj)
    args_nsd = SimpleNamespace(
            subj=subj,
            backbone_arch="dinov2_q",
            data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
            imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
            parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
            hemi=None,
    )
    args_nsd.tokenizer = None
    args_nsd.gen_size = 512
    args_nsd.topk = 100

    test_dataset = nsd_topk_parcel_dataset(args_nsd, split='test', transform=None, topk=args_nsd.topk)
    print(test_dataset.selected_parcel_idx)

    # Create the imagery dataset
    args_imagery = SimpleNamespace(
         subj=1,
        neuro_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_betas/ppdata/",
        behavior_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata/bdata/nsdimagery/",
        parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
        parcel_indices=test_dataset.selected_parcel_idx,
        img_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata/experiments/nsdimagery/rawtargetimages/",
        task='vis',  # Options: 'vis' or 'img'
        gen_size=224  # Smaller size for visualization
    )
    
    # Create dataset
    dataset = NSD_Imagery_Dataset(args_imagery)
    
    # Print information about the dataset
    print(f"\nDataset loaded successfully:")
    print(f"Number of conditions: {len(dataset)}")
    print(f"Conditions: {dataset.get_condition_names()}")
    
    # Get and examine a sample
    sample = dataset[0]
    print(f"\nSample for condition '{sample['condition']}':")
    print(f"Left hemisphere data shape: {sample['brain_lh_f'].shape}")
    print(f"Right hemisphere data shape: {sample['brain_rh_f'].shape}")
    print(f"Image tensor shape: {sample['img_encoder'].shape}, {sample['img_encoder'].min().item()}, {sample['img_encoder'].max().item()}")
    
    # Print parcellation info
    parcel_info = dataset.get_selected_parcel_info()
    print(f"\nParcellation summary:")
    print(f"Left hemisphere parcels: {len(parcel_info['lh']['parcel_indices'])}")
    print(f"Right hemisphere parcels: {len(parcel_info['rh']['parcel_indices'])}")
    print(f"Total parcels: {parcel_info['total_parcels']}")
    print(f"Max voxels per parcel: {parcel_info['max_voxels_per_parcel']}")
    print(f"Total voxels (LH): {parcel_info['lh']['total_voxels']}")
    print(f"Total voxels (RH): {parcel_info['rh']['total_voxels']}")
    
    for i in range(18):
        sample = dataset[i]
        print(f"Condition: {sample['condition']}, LH shape: {sample['brain_lh_f'].shape}, RH shape: {sample['brain_rh_f'].shape}, Image shape: {sample['img_encoder'].shape if sample['img_encoder'] is not None else None}")
