import os
import sys
import numpy as np
from pathlib import Path
from itertools import chain

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
    def __init__(self, args, split, transform=None, topk=100):
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
        # print(f"choosing {topk} parcels per hemisphere for training")

        self.parcels, self.selected_parcel_idx = {}, {}
        self.max_voxels = -np.inf
        for hemi in ["lh", "rh"]:
            self.parcels[hemi] = self.base_dataset.parcels[hemi][1:] # skip the first parcel because it is medial wall
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
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "img_encoder": img,
            "img_ipadapter": img_ipadapter,
            "text_input_ids": text_input_ids,
            "brain_lh_f": lh_, # shape: [200, max_voxels]
            "brain_rh_f": rh_, # shape: [200, max_voxels]
        }
        
if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace
    from transformers import CLIPTokenizer

    args = SimpleNamespace(
            subj=1,
            backbone_arch="dinov2_q",
            data_dir="/engram/nklab/datasets/natural_scene_dataset/model_training_datasets/neural_data",
            imgs_dir="/engram/nklab/datasets/natural_scene_dataset/nsddata_stimuli/stimuli/nsd",
            parcel_dir="/engram/nklab/algonauts/ethan/whole_brain_encoder/parcels/schaefer",
            hemi=None,
    )
    args.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    args.gen_size = 512
    args.topk = 100
    train_dataset=nsd_topk_parcel_dataset(args, split='test', transform=None, topk=args.topk)

    data = train_dataset[0]
    print(data['img_encoder'].shape)
    print(data['img_ipadapter'].shape)
    print(data['brain_lh_f'].shape)
    print(data['brain_rh_f'].shape)