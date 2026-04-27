# Installation

## Environment

Tested on:

| Component | Version |
|-----------|---------|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.8 – 3.10 |
| PyTorch | 2.0+ |
| CUDA | 11.6 / 11.7 / 11.8 |
| spconv | v2.x (`spconv-cu116` / `spconv-cu117` / `spconv-cu118`) |
| nuscenes-devkit | 1.0.5 |

---

## Step 1 — Clone

```bash
git clone https://github.com/FeedOnMilkT/CenterFormer_OpenPCDet_Version.git
cd CenterFormer_OpenPCDet_Version
```

## Step 2 — Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install spconv-cu118          # match your CUDA version
pip install nuscenes-devkit==1.0.5
pip install -r requirements.txt
```

## Step 3 — Build pcdet

```bash
python setup.py develop
```

## Step 4 — Verify CUDA ops

```bash
python -c "from pcdet.ops.iou3d_nms import iou3d_nms_utils; print('ops OK')"
```

If this fails, rebuild with:

```bash
cd pcdet/ops && python setup.py build_ext --inplace && cd ../..
```
