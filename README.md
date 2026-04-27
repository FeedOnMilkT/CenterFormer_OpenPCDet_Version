# CenterFormer — OpenPCDet Implementation

Implementation of **CenterFormer: Center-based Transformer for 3D Object Detection** (Zhou et al., ECCV 2022) within the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) framework.

> **Paper**: [arXiv 2209.05588](https://arxiv.org/abs/2209.05588)  
> **Dataset**: NuScenes (v1.0-mini for verification · v1.0-trainval for full training)

---

## Architecture

CenterFormer replaces the simple regression head of CenterPoint with a multi-scale Transformer decoder. Each candidate proposal queries BEV features across three spatial scales via cross-attention, enabling richer contextual reasoning than a single-point MLP head.

```
Raw Points
    │
    ▼
MeanVFE ──► VoxelResBackBone8x ──► HeightCompression
                                          │
                                          ▼
                              ┌─── CenterFormerCPN ────────────┐
                              │  Conv×6  →  stride-8  (mid)    │
                              │    ↓ stride-2 conv             │
                              │  Conv×5  →  stride-16 (low)    │  × CBAM
                              │    ↑ upsample + skip           │
                              │  concat → stride-4  (high)     │  × CBAM
                              └────────────────────────────────┘
                                          │  3 scale feature maps
                                          ▼
                              ┌─── CenterFormerHead ───────────┐
                              │  Heatmap branch (focal loss)   │
                              │  GT-forced proposals           │
                              │  Transformer Decoder × L:      │
                              │    Self-attn  (N proposals)    │
                              │    Cross-attn (local BEV)      │
                              │  Box head (1-D conv, L1 loss)  │
                              └────────────────────────────────┘
```

### Decoder Variants

| | Standard | Deformable |
|--|----------|------------|
| Cross-attn sampling | Fixed 3×3 grid (K=9) | Learned offsets (K=15) |
| Decoder layers | 3 | 2 |
| Attention heads | 4 | 6 |

Switch via `DECODER_TYPE: standard` or `DECODER_TYPE: deformable` in the model YAML.

---

## New Files

```
pcdet/models/model_utils/centerformer_utils.py   # Learnable PE · gather_bev_features · build_mlp
pcdet/models/backbones_2d/centerformer_neck.py   # CenterFormerCPN (multi-scale BEV + CBAM)
pcdet/models/dense_heads/centerformer_head.py    # CenterFormerHead (heatmap + decoder + loss)
pcdet/models/detectors/centerformer.py           # CenterFormer detector

tools/cfgs/dataset_configs/nuscenes_dataset_mini.yaml
tools/cfgs/nuscenes_models/centerformer_nuscenes_mini.yaml   # mini: D=128, N=64
tools/cfgs/nuscenes_models/centerformer_nuscenes.yaml        # full: D=256, N=500
```

---

## Requirements

- Linux · Python 3.8+ · PyTorch 2.0+ · CUDA 11.6+
- spconv-cu11x v2.x
- nuscenes-devkit == 1.0.5

See [docs/INSTALL.md](docs/INSTALL.md) for step-by-step setup.  
See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for data prep and training commands.

---

## Citation

```bibtex
@inproceedings{zhou2022centerformer,
  title     = {CenterFormer: Center-based Transformer for 3D Object Detection},
  author    = {Zhou, Zixiang and Zhao, Xiangchen and Wang, Yu and Wang, Panqu and Foroosh, Hassan},
  booktitle = {ECCV},
  year      = {2022}
}
```

```bibtex
@misc{openpcdet2020,
  title        = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author       = {OpenPCDet Development Team},
  howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
  year         = {2020}
}
```
