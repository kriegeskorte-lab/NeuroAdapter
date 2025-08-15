import os
import sys
import numpy as np
from pathlib import Path
from itertools import chain
from types import SimpleNamespace

from tqdm import tqdm
import h5py

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

class nsd_dataset_tempate(Dataset):
    def __init__(self, args, split="train", transform=None):
        self.subj = int(args.subj)
        self.hemi = args.hemi
        self.transform = transform
        self.backbone_arch = args.backbone_arch

        neural_data_path = Path(args.data_dir)
        self.metadata = np.load(
            neural_data_path / f"metadata_sub-{self.subj:02}.npy", allow_pickle=True
        ).item()
        self.img_order = self.metadata["img_presentation_order"]

        assert split in [
            "train",
            "test",
            "val",
        ], "split must be either train, test, val, or custom"
        self.split_imgs = self.metadata[f"{split}_img_num"]

        if self.hemi is not None:
            self.betas = h5py.File(
                neural_data_path / f"betas_sub-{self.subj:02}.h5", "r"
            )[f"{self.hemi}_betas"]
        else:
            self.betas = [
                h5py.File(neural_data_path / f"betas_sub-{self.subj:02}.h5", "r")[
                    f"{hemi}_betas"
                ]
                for hemi in ["lh", "rh"]
            ]

        imgs_dir = Path(args.imgs_dir)
        self.imgs = h5py.File(imgs_dir / "nsd_stimuli.hdf5", "r")

        parcel_path = Path(args.parcel_dir)
        if args.hemi is not None:
            self.parcels = torch.load(
                parcel_path / f"{args.hemi}_labels_s{self.subj:02}.pt",
                weights_only=True,
            )
            self.valid_voxel_mask = torch.zeros(len(self.betas[0]), dtype=torch.bool)
            for parcel in self.parcels:
                self.valid_voxel_mask[parcel] = True
            self.num_hemi_voxels = torch.sum(self.valid_voxel_mask).item()
            print("Number of valid voxels: ", self.num_hemi_voxels)

            self.num_parcels = len(self.parcels)
            print("Number of parcels: ", self.num_parcels)
        else:
            self.parcels = {}
            self.valid_voxel_mask = torch.zeros(
                sum([len(b[0]) for b in self.betas]), dtype=torch.bool
            )
            for hemi in ["lh", "rh"]:
                self.parcels[hemi] = torch.load(
                    parcel_path / f"{hemi}_labels_s{self.subj:02}.pt", weights_only=True
                )
                # for parcel in self.parcels:
                #     self.valid_voxel_mask[
                #         parcel + len(self.betas[0][0]) if hemi == "rh" else 0
                #     ] = True

    def plot_parcels(self):
        if self.overlap:
            print("Cannot plot overlapping parcels")
            return

        import cortex
        import cortex.polyutils
        import contextlib
        from io import StringIO
        import sys

        @contextlib.contextmanager
        def suppress_print():
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                yield
            finally:
                sys.stdout = original_stdout

        def plot_parcels(
            lh, rh, title="", fig_path=None, cmap="freesurfer_aseg_256", clip=1
        ):
            plt.rc("xtick", labelsize=19)
            plt.rc("ytick", labelsize=19)

            subject = "fsaverage"
            data = np.append(lh, rh)
            vertex_data = cortex.Vertex(
                data, subject, cmap=cmap, vmin=0, vmax=clip
            )  # "afmhot"

            with suppress_print():
                cortex.quickshow(vertex_data, with_curvature=True)

            plt.title(title)

            if fig_path is not None:
                plt.savefig(fig_path, dpi=300)
            else:
                plt.show()

        fsavg = np.empty((max([torch.max(p) for p in self.parcels]) + 1))
        fsavg[:] = np.nan

        for idx, parcel in enumerate(self.parcels):
            fsavg[parcel.numpy()] = idx

        plot_parcels(
            fsavg if self.hemi == "lh" else np.full_like(fsavg, np.nan),
            fsavg if self.hemi == "rh" else np.full_like(fsavg, np.nan),
            clip=np.nanmax(fsavg),
        )

    def reformat_parcels(self, parcels, metaparcel_idx):
        """
        args:
        parcels: [[(level1, level2, ...), ...], [(level1, level2, ...), ...], ...]

        returns: [level1: [idx1, idx2, ...], level2: [idx1, idx2, ...], ...]
        """
        flattened_parcels = np.array(list(chain.from_iterable(parcels)))
        print(flattened_parcels)
        flattened_parcels = torch.from_numpy(flattened_parcels)
        flattened_parcels = flattened_parcels[flattened_parcels[:, 0] == metaparcel_idx]
        flattened_parcels = flattened_parcels[:, 1]
        uq_parcels = torch.unique(flattened_parcels)

        labels = [[] for _ in range(len(uq_parcels))]
        parcel_to_idx = {p.item(): i for i, p in enumerate(uq_parcels)}
        for v in range(len(parcels)):
            for affiliation in parcels[v]:
                if affiliation[0] != metaparcel_idx:
                    continue
                parcel_idx = parcel_to_idx[affiliation[1]]
                labels[parcel_idx].append(v)

        for i in range(len(labels)):
            labels[i] = torch.tensor(labels[i])

        return labels

    def reformat_parcels_nonoverlapping(self, original_parcels, parcels, position=[]):
        """
        args:
        parcels: [(level1, level2, ...), (level1, level2, ...), ...]

        returns: [level1: [idx1, idx2, ...], level2: [idx1, idx2, ...], ...]
        """
        if len(parcels[0]) == 1:
            t = [
                (original_parcels == torch.tensor(position + [p]))
                .all(dim=1)
                .nonzero(as_tuple=True)[0]
                for p in torch.unique(parcels)
            ]
            return t

        return [
            self.reformat_parcels_nonoverlapping(
                original_parcels,
                parcels[torch.where(parcels[:, 0] == p)[0]][:, 1:],
                [p.item()],
            )
            for p in torch.unique(parcels[:, 0])
        ]

    def transform_img(self, img):
        # img = Image.fromarray(img)
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')

        if self.transform:
            img = self.transform(img)

        if self.backbone_arch:
            if "dinov2" in self.backbone_arch:
                patch_size = 14

                size_im = (
                    img.shape[0],
                    int(np.ceil(img.shape[1] / patch_size) * patch_size),
                    int(np.ceil(img.shape[2] / patch_size) * patch_size),
                )
                paded = torch.zeros(size_im)
                paded[:, : img.shape[1], : img.shape[2]] = img
                img = paded

        return img

    def parcellate_fmri(self, fmri_data, labels):
        fmri = []
        for parcel in labels:
            parcel_data = fmri_data[parcel]
            pad_size = self.max_parcel_size - parcel_data.size(0)
            fmri.append(F.pad(parcel_data, (0, pad_size), mode="constant", value=0))
        return torch.stack(fmri)

    def get_parcel_mask(self):
        mask = torch.zeros(self.num_parcels, self.max_parcel_size, dtype=torch.bool)

        for i, parcel in enumerate(self.parcels):
            pad_size = self.max_parcel_size - parcel.size(0)
            if pad_size == 0:
                mask[i] = 1
            else:
                mask[i][:-pad_size] = 1

        return mask


class nsd_dataset(nsd_dataset_tempate):
    def __init__(
        self, args, split="train", parcel_path=None, transform=None, preload_data=False
    ):
        super().__init__(args, split, transform)

        self.split_idxs = np.where(
            np.isin(self.metadata["img_presentation_order"], self.split_imgs)
        )[0]

    def __getitem__(self, idx):
        split_idx = self.split_idxs[idx]

        img_ind = self.img_order[split_idx]  # image index in nsd
        img = self.imgs["imgBrick"][img_ind]
        img = self.transform_img(img)

        fmri_data = {}
        if self.hemi is not None:
            fmri_data["betas"] = torch.from_numpy(self.betas[split_idx])
        else:
            fmri_data["betas"] = torch.from_numpy(
                np.concatenate([b[split_idx] for b in self.betas])
            )

        return img, fmri_data

    def __len__(self):
        return len(self.split_idxs)


class nsd_dataset_avg(nsd_dataset_tempate):
    def __init__(
        self, args, split="train", parcel_paths=None, transform=None, preload_data=False
    ):
        super().__init__(args, split, transform)

        assert split in [
            "train",
            "test",
            "val",
        ], "split must be either train, test or val"

        # some of the images in split_imgs are were not actually presented, so let's take them out
        self.split_presented_imgs = self.split_imgs[
            np.isin(self.split_imgs, self.metadata["img_presentation_order"])
        ]
        self.img_to_runs = [
            np.where(self.metadata["img_presentation_order"] == img_ind)[0]
            for img_ind in self.split_presented_imgs
        ]

    def __getitem__(self, i):
        img_ind = self.split_presented_imgs[i]  # image index in nsd
        img = self.imgs["imgBrick"][img_ind]

        if self.transform is not None:
            img = self.transform_img(img)

        fmri_data = {}
        data_idxs = self.img_to_runs[i]

        if self.hemi is not None:
            data = torch.from_numpy(self.betas[data_idxs])
            data = torch.mean(data, axis=0)
            fmri_data["betas"] = data
        else:
            data = np.concatenate([b[data_idxs] for b in self.betas], axis=1)
            data = torch.from_numpy(data)
            data = torch.mean(data, axis=0)
            fmri_data["betas"] = data

        return img, fmri_data

    def __len__(self):
        return len(self.split_presented_imgs)

class nsd_topk_parcel_dataset(Dataset):
    def __init__(self, args, split, transform=None, topk=100, selected_parcel_idx={}):
        """
        avg: True if average across trials, else load single-trial data
        split: 'train', 'val', or 'test'
        transform: image transform
        topk: number of top parcels to select based on mean voxel SNR
        """
        assert topk > 0, "topk must be positive"
        self.base_dataset = nsd_dataset_avg(args, transform=None, split=split)  
        self.args = args
        self.transform = transform
        topk = abs(topk)
        self.num_parcels = topk * 2
        print(f"Training on subject: {args.subj}")
        # print(f"choosing {topk} parcels per hemisphere for training")

        self.selected_parcel_idx = selected_parcel_idx.copy()
        self.parcels = {}
        self.max_voxels = 0
        for hemi in ["lh", "rh"]:
            self.parcels[hemi] = self.base_dataset.parcels[hemi][1:] # skip the first parcel because it is medial wall

            if selected_parcel_idx == {}:
                nc = self.metadata(subj=args.subj)[f"{hemi}_ncsnr"].squeeze()
                mean_snr = []
                for voxel_idx in self.parcels[hemi]:
                    mean_snr.append(nc[voxel_idx].mean())
                sorted_idx = (
                    np.argsort(mean_snr)[::-1]
                )
                topk_idx = sorted_idx[:topk]
                self.selected_parcel_idx[hemi] = topk_idx
                self.max_voxels = max(
                    self.max_voxels, max([len(self.parcels[hemi][i_parcel]) for i_parcel in topk_idx])
                )
            else:
                self.max_voxels = max(
                    self.max_voxels, max([len(self.parcels[hemi][i_parcel]) for i_parcel in self.selected_parcel_idx[hemi]])
                )
            
        self.tokenizer = args.tokenizer
        self.gen_size = args.gen_size
        self.ipadapter_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.gen_size),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def metadata(self, subj):
        neural_data_path = Path(self.args.data_dir)
        meta = np.load(
            neural_data_path / f"metadata_sub-{subj:02}.npy", allow_pickle=True
        ).item()
        return meta

    def extract_and_pad(self, fmri_data, hemi):
        out = []
        for idx in self.selected_parcel_idx[hemi]:
            voxel_idxs = self.parcels[hemi][idx]
            roi = fmri_data[voxel_idxs]
            if roi.shape[0] < self.max_voxels:
                pad = torch.zeros(self.max_voxels - roi.shape[0])
                roi = torch.cat([roi, pad])
            else:
                roi = roi[: self.max_voxels]
            out.append(roi)
        return torch.stack(out)  # shape: [200, max_voxels]
    
    def get_selected_voxel_indices(self):
        """
        Get voxel indices for each selected parcel.
        
        Returns:
            dict: Dictionary with 'lh' and 'rh' keys, each containing a list of tensor arrays
                  with voxel indices for each selected parcel
        """
        voxel_indices = {"lh": [], "rh": []}
        
        for hemi in ["lh", "rh"]:
            for parcel_idx in self.selected_parcel_idx[hemi]:
                voxel_idxs = self.parcels[hemi][parcel_idx]
                voxel_indices[hemi].append(voxel_idxs)
        
        return voxel_indices
    
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

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, fmri_data = self.base_dataset[idx]
        
        lh_fmri = fmri_data["betas"][:163842]
        rh_fmri = fmri_data["betas"][163842:]

        lh_ = self.extract_and_pad(
            lh_fmri, hemi="lh"
        )
        rh_ = self.extract_and_pad(
            rh_fmri, hemi="rh"
        )

        img_ipadapter = self.ipadapter_transform(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Process text
        text = ""
        # text_input_ids = self.tokenizer(
        #     text,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # ).input_ids
        text_input_ids = torch.zeros(1, dtype=torch.long)

        return {
            "img_encoder": img,
            "img_ipadapter": img_ipadapter,
            "text_input_ids": text_input_ids,
            "brain_lh_f": lh_, # shape: [200, max_voxels]
            "brain_rh_f": rh_, # shape: [200, max_voxels]
        }
        
class nsd_groupwise_topk_parcel_dataset(Dataset):
    def __init__(self, args, split, transform=None, topk=100, train_subj=list(range(1, 9)), test_subj=None):
        """
        Group-wise parcel selection dataset that aggregates mean SNR ranks across all subjects.
        
        Args:
            args: arguments containing data paths and configuration
            split: 'train', 'val', or 'test'
            transform: image transform
            topk: number of top parcels to select based on mean rank across subjects
            test_subj: subject ID for test/val splits (1-8), ignored for train split
        """
        assert topk > 0, "topk must be positive"
        self.args = args
        self.transform = transform
        self.split = split
        self.topk = abs(topk)
        self.num_parcels = topk * 2  # topk per hemisphere
        
        # For train: use all subjects, for test/val: use specified subject
        if split == 'train':
            self.subjects = train_subj  # subjects 1-8
            print(f"Training mode: using subjects {self.subjects}")
        else:
            assert test_subj is not None and 1 <= test_subj <= 8, "test_subj must be between 1-8 for test/val splits"
            self.subjects = [test_subj]
            print(f"{split.capitalize()} mode: using subject {test_subj}")
        
        # Step 1: Aggregate SNR ranks across all subjects (1-8) for parcel selection
        self.selected_parcel_idx = self._select_parcels_groupwise()

        # Step 2: Load data for the specified subjects
        self.base_datasets = {}
        self.parcels = {}
        self.max_voxels = 0

        for subj in self.subjects:
            # Create args for this subject
            subj_args = SimpleNamespace(**vars(args))
            subj_args.subj = subj
            
            # Load base dataset
            self.base_datasets[subj] = nsd_dataset_avg(subj_args, transform=None, split=split)
            
            # Load parcels for this subject
            self.parcels[subj] = {}
            for hemi in ["lh", "rh"]:
                self.parcels[subj][hemi] = self.base_datasets[subj].parcels[hemi][1:]  # skip medial wall
                
                # Update max_voxels based on selected parcels
                for parcel_idx in self.selected_parcel_idx[hemi]:
                    if parcel_idx < len(self.parcels[subj][hemi]):
                        voxel_count = len(self.parcels[subj][hemi][parcel_idx])
                        self.max_voxels = max(self.max_voxels, voxel_count)
        
        # Setup tokenizer and transforms
        self.tokenizer = args.tokenizer
        self.gen_size = args.gen_size
        self.ipadapter_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.gen_size),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Create index mapping for samples
        self._create_sample_indices()
    
    def _select_parcels_groupwise(self):
        """Select parcels based on mean SNR rank across all subjects (1-8)"""
        parcel_ranks = {"lh": [], "rh": []}
        
        # Get SNR data for all subjects
        all_subject_snr = {}
        max_parcels = {"lh": 0, "rh": 0}
        
        for subj in range(1, 9):  # Always use all subjects for parcel selection
            subj_meta = self._load_metadata(subj)
            
            # Load parcels for this subject
            parcel_path = Path(self.args.parcel_dir)
            subj_parcels = {}
            for hemi in ["lh", "rh"]:
                subj_parcels[hemi] = torch.load(
                    parcel_path / f"{hemi}_labels_s{subj:02}.pt", weights_only=True
                )[1:]  # skip medial wall
                max_parcels[hemi] = max(max_parcels[hemi], len(subj_parcels[hemi]))
            
            all_subject_snr[subj] = {"parcels": subj_parcels, "meta": subj_meta}
        
        # Calculate ranks for each hemisphere
        selected_idx = {}
        for hemi in ["lh", "rh"]:
            # Create rank matrix: subjects x parcels
            rank_matrix = []
            
            for subj in range(1, 9):
                nc = all_subject_snr[subj]["meta"][f"{hemi}_ncsnr"].squeeze()
                parcels = all_subject_snr[subj]["parcels"][hemi]
                
                # Calculate mean SNR per parcel
                mean_snr = []
                for voxel_idx in parcels:
                    mean_snr.append(nc[voxel_idx].mean())
                
                # Convert to ranks (lower rank = better SNR)
                ranks = np.argsort(np.argsort(mean_snr)[::-1])
                rank_matrix.append(ranks)
            
            rank_matrix = np.stack(rank_matrix)  # shape: [8, num_parcels]
            mean_ranks = rank_matrix.mean(axis=0)
            
            # Select top parcels based on mean rank
            valid_parcels = ~np.isnan(mean_ranks)
            valid_mean_ranks = mean_ranks[valid_parcels]
            valid_indices = np.where(valid_parcels)[0]
            
            sorted_indices = valid_indices[np.argsort(valid_mean_ranks)]
            selected_idx[hemi] = sorted_indices[:self.topk]
        
        return selected_idx
    
    def _load_metadata(self, subj):
        """Load metadata for a specific subject"""
        neural_data_path = Path(self.args.data_dir)
        return np.load(
            neural_data_path / f"metadata_sub-{subj:02}.npy", allow_pickle=True
        ).item()
    
    def _create_sample_indices(self):
        """Create mapping from dataset index to (subject, sample_idx)"""
        self.sample_mapping = []
        
        for subj in self.subjects:
            dataset_len = len(self.base_datasets[subj])
            for sample_idx in range(dataset_len):
                self.sample_mapping.append((subj, sample_idx))
                
    def get_selected_voxel_indices(self, subj=None):
        """
        Get voxel indices for each selected parcel.
        
        Args:
            subj: Subject ID (1-8). If None, returns voxel indices for the first subject in self.subjects
        
        Returns:
            dict: Dictionary with 'lh' and 'rh' keys, each containing a list of tensor arrays
                  with voxel indices for each selected parcel
        """
        if subj is None:
            subj = self.subjects[0]
        
        assert subj in self.subjects, f"Subject {subj} not in available subjects: {self.subjects}"
        
        voxel_indices = {"lh": [], "rh": []}
        
        for hemi in ["lh", "rh"]:
            for parcel_idx in self.selected_parcel_idx[hemi]:
                if parcel_idx < len(self.parcels[subj][hemi]):
                    voxel_idxs = self.parcels[subj][hemi][parcel_idx]
                    voxel_indices[hemi].append(voxel_idxs)
                else:
                    # If this subject doesn't have this parcel, return empty tensor
                    voxel_indices[hemi].append(torch.tensor([], dtype=torch.long))
        
        return voxel_indices
    
    def get_selected_parcel_info(self, subj=None):
        """
        Get detailed information about selected parcels for a specific subject.
        
        Args:
            subj: Subject ID (1-8). If None, uses the first subject in self.subjects
        
        Returns:
            dict: Dictionary containing parcel information including indices, sizes, and voxel counts
        """
        if subj is None:
            subj = self.subjects[0]
            
        assert subj in self.subjects, f"Subject {subj} not in available subjects: {self.subjects}"
        
        info = {
            "subject": subj,
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
                if parcel_idx < len(self.parcels[subj][hemi]):
                    voxel_count = len(self.parcels[subj][hemi][parcel_idx])
                else:
                    voxel_count = 0
                info[hemi]["voxel_counts"].append(voxel_count)
                info[hemi]["total_voxels"] += voxel_count
        
        info["max_voxels_per_parcel"] = self.max_voxels
        info["total_parcels"] = len(self.selected_parcel_idx["lh"]) + len(self.selected_parcel_idx["rh"])
        
        return info
    
    def extract_and_pad(self, fmri_data, hemi, subj):
        """Extract and pad fMRI data for selected parcels"""
        out = []
        dtype = fmri_data.dtype
        for parcel_idx in self.selected_parcel_idx[hemi]:
            if parcel_idx < len(self.parcels[subj][hemi]):
                voxel_idxs = self.parcels[subj][hemi][parcel_idx]
                roi = fmri_data[voxel_idxs]
            else:
                # If this subject doesn't have this parcel, use zeros
                roi = torch.zeros((1,), dtype=dtype)

            if roi.shape[0] < self.max_voxels:
                pad = torch.zeros(self.max_voxels - roi.shape[0], dtype=dtype)
                roi = torch.cat([roi, pad])
            else:
                roi = roi[:self.max_voxels]
            out.append(roi)
        
        return torch.stack(out)
    
    def __len__(self):
        return len(self.sample_mapping)
    
    def __getitem__(self, idx):
        subj, sample_idx = self.sample_mapping[idx]
        
        # Get data from the appropriate subject's dataset
        img, fmri_data = self.base_datasets[subj][sample_idx]
        
        # Split brain data by hemisphere
        lh_fmri = fmri_data["betas"][:163842]
        rh_fmri = fmri_data["betas"][163842:]
        
        # Extract and pad selected parcels
        lh_ = self.extract_and_pad(lh_fmri, hemi="lh", subj=subj)
        rh_ = self.extract_and_pad(rh_fmri, hemi="rh", subj=subj)
        
        # Process image
        img_ipadapter = self.ipadapter_transform(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        # Process text (empty for now)
        text = ""
        # text_input_ids = self.tokenizer(
        #     text,
        #     max_length=self.tokenizer.model_max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # ).input_ids
        text_input_ids = torch.zeros(1, dtype=torch.long)
        
        return {
            "img_encoder": img,
            "img_ipadapter": img_ipadapter,
            "text_input_ids": text_input_ids,
            "brain_lh_f": lh_,  # shape: [topk, max_voxels]
            "brain_rh_f": rh_,  # shape: [topk, max_voxels]
        }
        
def get_dominant_roi_per_parcel(dataset, metadata, all_roi_names, min_overlap_threshold=0.1):
    """
    Get the dominant ROI for each selected parcel.
    
    Args:
        dataset: nsd_topk_parcel_dataset instance
        metadata: subject metadata
        all_roi_names: list of ROI names
        min_overlap_threshold: minimum overlap ratio to consider a match
    
    Returns:
        dict: {hemi: [(parcel_idx, dominant_roi, overlap_ratio), ...]}
    """
    schaefer_voxel_indices = dataset.get_selected_voxel_indices()
    dominant_rois = {"lh": [], "rh": []}
    
    for hemi in ['lh', 'rh']:
        for parcel_idx, voxel_idxs in enumerate(schaefer_voxel_indices[hemi]):
            schaefer_voxels = set(voxel_idxs.numpy())
            
            best_roi = None
            best_overlap = 0
            
            for roi in all_roi_names:
                if f'{hemi}_rois' in metadata and roi in metadata[f'{hemi}_rois']:
                    roi_voxels = set([id for id, val in enumerate(metadata[f'{hemi}_rois'][roi]) if val == True])
                    overlap = len(schaefer_voxels.intersection(roi_voxels))
                    overlap_ratio = overlap / len(schaefer_voxels)
                    
                    if overlap_ratio > best_overlap and overlap_ratio >= min_overlap_threshold:
                        best_overlap = overlap_ratio
                        best_roi = roi
            
            parcel_original_idx = dataset.selected_parcel_idx[hemi][parcel_idx]
            dominant_rois[hemi].append((parcel_original_idx, best_roi, best_overlap))
    
    return dominant_rois
        
if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace
    # from transformers import CLIPTokenizer
    subj=1
    print("Setting up test configuration for subject ", subj)
    args = SimpleNamespace(
            subj=subj,
            backbone_arch="dinov2_q",
            data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
            imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
            parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
            hemi=None,
    )
    args.tokenizer = None
    args.gen_size = 512
    args.topk = 100
    
    print("\n" + "="*50)
    print("Testing nsd_topk_parcel_dataset (single subject)")
    print("="*50)
    train_dataset = nsd_topk_parcel_dataset(args, split='train', transform=None, topk=args.topk)
    print(len(train_dataset), "samples in the dataset")
    print(train_dataset.selected_parcel_idx)
    print(len(train_dataset.selected_parcel_idx['lh']), "left hemisphere parcels")
    print(len(train_dataset.selected_parcel_idx['rh']), "right hemisphere parcels")

    data = train_dataset[0]
    print()
    print("\nOutput shapes:")
    print(f"img_encoder: {data['img_encoder'].shape}")
    print(f"img_ipadapter: {data['img_ipadapter'].shape}")
    print(f"brain_lh_f: {data['brain_lh_f'].shape}")
    print(f"brain_rh_f: {data['brain_rh_f'].shape}")
    
    # print("\n" + "="*50)
    # print("Testing nsd_groupwise_topk_parcel_dataset (group-wise)")
    # print("="*50)
    # train_group_dataset = nsd_groupwise_topk_parcel_dataset(args, split='train', transform=None, topk=args.topk, train_subj=list(range(1, 9)))
    # print(len(train_group_dataset), "samples in the dataset")
    # print(train_group_dataset.selected_parcel_idx)
    # print(len(train_group_dataset.selected_parcel_idx['lh']), "left hemisphere parcels")
    # print(len(train_group_dataset.selected_parcel_idx['rh']), "right hemisphere parcels")

    # group_data = train_group_dataset[0]
    # print("\nOutput shapes:")
    # print(f"img_encoder: {group_data['img_encoder'].shape}")
    # print(f"img_ipadapter: {group_data['img_ipadapter'].shape}")
    # print(f"brain_lh_f: {group_data['brain_lh_f'].shape}")
    # print(f"brain_rh_f: {group_data['brain_rh_f'].shape}")

    # shared_lh = list(set(train_dataset.selected_parcel_idx['lh']) & set(train_group_dataset.selected_parcel_idx['lh']))
    # shared_rh = list(set(train_dataset.selected_parcel_idx['rh']) & set(train_group_dataset.selected_parcel_idx['rh']))

    # print(len(shared_lh), shared_lh)
    # print(len(shared_rh), shared_rh)

    from pathlib import Path
    import numpy as np

    neural_data_path = Path(
        "/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data"
    )
    metadata = np.load(
        neural_data_path / f"metadata_sub-{subj:02}.npy", allow_pickle=True
    ).item()
    
    all_roi_names = list(metadata['lh_rois'].keys())[:24] + list(metadata['rh_rois'].keys())[:24]
    print(f' all_roi_names: {all_roi_names}, len: {len(all_roi_names)}')

    roi = all_roi_names[0]
    print(metadata['lh_rois'][roi])
    print(len(metadata['lh_rois'][roi]))
    print([id for id, val in enumerate(metadata['lh_rois'][roi]) if val == True])
    
    dominant_rois = get_dominant_roi_per_parcel(train_dataset, metadata, all_roi_names)
    
    print("\n" + "="*50)
    print("DOMINANT ROI PER PARCEL (>10% overlap)")
    print("="*50)
    
    for hemi in ['lh', 'rh']:
        print(f"\n{hemi.upper()} Hemisphere:")
        for parcel_idx, roi_name, overlap in dominant_rois[hemi]:
            if roi_name:
                print(f"Parcel {parcel_idx:3d}: {roi_name:20s} ({overlap:.1%})")
            else:
                print(f"Parcel {parcel_idx:3d}: No dominant ROI")