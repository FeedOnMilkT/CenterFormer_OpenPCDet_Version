# Getting Started

All training and evaluation commands must be run from the `tools/` directory.

```bash
cd tools/
```

---

## 1. Dataset Preparation

### NuScenes v1.0-mini (pipeline verification)

Download [NuScenes v1.0-mini](https://www.nuscenes.org/nuscenes#download) and place it under `data/nuscenes/`:

```
data/nuscenes/
└── v1.0-mini/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
```

Generate info files:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_nuscenes_infos \
    --cfg_file cfgs/dataset_configs/nuscenes_dataset_mini.yaml \
    --version v1.0-mini
```

Expected output under `data/nuscenes/v1.0-mini/`:
```
nuscenes_infos_10sweeps_train.pkl
nuscenes_infos_10sweeps_val.pkl
```

### NuScenes v1.0-trainval (full training)

Download the full dataset and place it under `data/nuscenes/v1.0-trainval/`. Then generate info files and the GT sampling database:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_nuscenes_infos \
    --cfg_file cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval

python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_groundtruth_database \
    --cfg_file cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```

---

## 2. Training

### Mini — pipeline verification (single GPU)

Confirms the end-to-end pipeline runs without error. Accuracy on mini is not meaningful.

```bash
python train.py \
    --cfg_file cfgs/nuscenes_models/centerformer_nuscenes_mini.yaml \
    --batch_size 2 \
    --workers 2 \
    --extra_tag mini_smoke_test \
    --fix_random_seed
```

To reduce memory further (e.g. for debugging), override `NUM_PROPOSALS` inline:

```bash
python train.py \
    --cfg_file cfgs/nuscenes_models/centerformer_nuscenes_mini.yaml \
    --batch_size 2 \
    --workers 0 \
    --extra_tag fwd_check \
    --set MODEL.DENSE_HEAD.NUM_PROPOSALS 32
```

### Full training — single GPU

```bash
python train.py \
    --cfg_file cfgs/nuscenes_models/centerformer_nuscenes.yaml \
    --batch_size 4 \
    --workers 4 \
    --extra_tag full_train_v1
```

### Full training — multi-GPU (recommended, 4 × GPU)

```bash
bash scripts/dist_train.sh 4 \
    --cfg_file cfgs/nuscenes_models/centerformer_nuscenes.yaml \
    --batch_size 16 \
    --workers 4 \
    --extra_tag full_train_v1
```

### Deformable decoder variant

Add `--set MODEL.DENSE_HEAD.DECODER_TYPE deformable MODEL.DENSE_HEAD.NUM_DECODER_LAYERS 2 MODEL.DENSE_HEAD.NUM_HEADS 6` to any command above, or edit the YAML directly.

> **Note**: `HIDDEN_CHANNEL` must be divisible by `NUM_HEADS`. For 6 heads use `HIDDEN_CHANNEL: 192` or `384`.

---

## 3. Evaluation

Checkpoints are saved to:
```
output/nuscenes_models/<config_name>/<extra_tag>/ckpt/
```

### Evaluate a single checkpoint

```bash
python test.py \
    --cfg_file cfgs/nuscenes_models/centerformer_nuscenes_mini.yaml \
    --batch_size 1 \
    --ckpt ../output/nuscenes_models/centerformer_nuscenes_mini/mini_smoke_test/ckpt/checkpoint_epoch_2.pth \
    --extra_tag mini_smoke_test
```

### Evaluate all checkpoints in a directory

```bash
python test.py \
    --cfg_file cfgs/nuscenes_models/centerformer_nuscenes.yaml \
    --batch_size 4 \
    --eval_all \
    --ckpt_dir ../output/nuscenes_models/centerformer_nuscenes/full_train_v1/ckpt
```

---

## 4. Key Config Parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `DECODER_TYPE` | `DENSE_HEAD` | `standard` (fixed 3×3) or `deformable` (K=15 offsets) |
| `NUM_DECODER_LAYERS` | `DENSE_HEAD` | 3 (standard) / 2 (deformable) |
| `NUM_HEADS` | `DENSE_HEAD` | 4 (standard) / 6 (deformable) |
| `HIDDEN_CHANNEL` | `DENSE_HEAD` | Query/value dimension D; must be divisible by `NUM_HEADS` |
| `NUM_PROPOSALS` | `DENSE_HEAD` | Proposals per frame: 64 (mini debug) / 500 (full train) |
| `NUM_FILTERS` | `BACKBONE_2D` | CPN output channels; auto-forwarded to head as `input_channels` |
| `FEATURE_MAP_STRIDE` | `TARGET_ASSIGNER_CONFIG` | Set to `4` (CPN upsamples VoxelResBackBone8x stride-8 to stride-4) |
