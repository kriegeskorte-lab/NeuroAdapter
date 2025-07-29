# Brain Adapter: fMRI-to-Image Generation

A novel approach for generating images from brain activity using fMRI data and diffusion models. This project implements a brain adapter that integrates with Stable Diffusion to decode visual imagery from neural signals.

## Overview

The Brain Adapter leverages the IP-Adapter architecture to inject brain-derived conditioning tokens into the diffusion process, enabling direct fMRI-to-image generation. The system learns to map brain activity patterns to visual features, allowing reconstruction of viewed or imagined images.

## Project Structure

```
brain_decoding/
├── README.md                          # This file
├── train_brain_adapter.py             # Main training script
├── decode_brain_adapter.py            # Inference and decoding script
├── plot_brain_adapter.py              # Visualization utilities
├── metric_brain_adapter.py            # Evaluation metrics
├── acc_config.yaml                    # Accelerate configuration
├── brain_adapter/                     # Core implementation
│   ├── model.py                       # NeuroAdapter and GuidanceGenerator
│   ├── dataset.py                     # NSD dataset handling
│   ├── utils.py                       # Utility functions
│   └── ip_adapter/                    # IP-Adapter components
└── whole_brain_encoder/               # Brain encoder for evaluation
```

## Getting Started

### Installation
```bash
conda env create -f env.yml
conda activate brain
```

### Preparation
1. git clone whole_brain_encoder
2. create a `__init__.py` in whole_brain_encoder
3. replace `import xxx` with in `import whole_brain_encoder.xxx` in whole_brain_encoder
4. create a folder called `checkpoints`
5. go to `/engram/nklab/algonauts/ethan/whole_brain_encoder/checkpoints`
6. organize your folders and copy & paste everything from `/nsd_test/dinov2_q_transformer/schaefer/*`

### Training

Train a brain adapter model using fMRI data:

```bash
# Basic training with linear projection
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 8 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach linear_projection \
    --wandb

# Advanced training with transformer decoder
accelerate launch --config_file acc_config.yaml train_brain_adapter.py \
    --learning_rate 1e-04 \
    --num_train_epochs 100 \
    --train_batch_size 16 \
    --subject_id 1 \
    --topk 100 \
    --condition_dim 768 \
    --num_decoder_queries 50 \
    --sub_approach transformer_decoder \
    --wandb
```

### Inference

Generate images from trained models:

```bash
# Decode subset of test images
python decode_brain_adapter.py \
    --model_weights_dir brain_adapter/model_weights/07_26_2025-22_29 \
    --saved_epochs 100 \
    --start_idx 0 \
    --end_idx 16 \
    --save_all_candidates \
    --num_predictions 8 \
    --subject_id 1

# Decode full test dataset
python decode_brain_adapter.py \
    --model_weights_dir brain_adapter/model_weights/07_26_2025-22_29 \
    --saved_epochs 100 \
    --eval_full_dataset \
    --batch_size 1 \
    --subject_id 1
```

### Evaluation

Compute comprehensive quality metrics:

```bash
# Evaluate subset results
python metric_brain_adapter.py \
    --model_weights_dir brain_adapter/model_weights/07_26_2025-22_29 \
    --saved_epochs 100 \
    --evaluation_mode subset

# Evaluate full dataset results
python metric_brain_adapter.py \
    --model_weights_dir brain_adapter/model_weights/07_26_2025-22_29 \
    --saved_epochs 100 \
    --evaluation_mode full
```

### Visualization

Generate comparison plots:

```bash
# Plot all samples
python plot_brain_adapter.py \
    brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset

# Plot specific range
python plot_brain_adapter.py \
    brain_adapter/decoded_stimuli/07_26_2025-22_29/epoch_100/subset \
    --start_idx 0 --end_idx 10 \
    --plot_type individual
```

## 🔧 Configuration

### Key Parameters

- `--topk`: Number of brain parcels to use (default: 100)
- `--condition_dim`: Conditioning token dimension (default: 768)
- `--sub_approach`: Architecture type (`linear_projection` or `transformer_decoder`)
- `--num_decoder_queries`: Number of transformer queries (default: 50)
- `--learning_rate`: Training learning rate (default: 1e-4)
- `--subject_id`: NSD subject ID (1-8, default: 1)

### Output Structure

```
brain_adapter/
├── model_weights/
│   └── MM_DD_YYYY-HH_MM/
│       └── checkpoint-{epoch}/
├── decoded_stimuli/
│   └── model_name/
│       └── epoch_100/
│           ├── subset/
│           │   ├── sample_000001.npz
│           │   ├── evaluation_metadata.json
│           │   └── metric_subset.json
│           └── full/
└── plotted_stimuli/
    └── model_name/
        └── epoch_100/
            └── subset/
                ├── individual/
                ├── grid_comparison.png
                └── plot_summary.json
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
<!-- 
## 🙏 Acknowledgments

- [Natural Scenes Dataset](https://naturalscenesdataset.org/) for providing the fMRI data
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for the adapter architecture
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the base diffusion model
- [Hugging Face](https://huggingface.co/) for model hosting and utilities

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{brain_adapter_2025,
  title={Brain Adapter: fMRI-to-Image Generation using Diffusion Models},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/brain_decoding}
}
```

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@domain.com
- **Project Link**: [https://github.com/your-username/brain_decoding](https://github.com/your-username/brain_decoding)

---
 -->

