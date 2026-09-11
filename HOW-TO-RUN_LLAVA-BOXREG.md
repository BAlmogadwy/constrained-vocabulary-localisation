# How to Run the COCO LLaVA Box Regression Experiment

This document explains how to prepare datasets, configure the LLaVA backbone, and run both training and inference for `experiments.coco_llava_boxreg`.

## 0. Requirements
- Python 3.10+ with `pip`
- GPU with at least 24 GB VRAM (recommended for 7B models; CPU works for smoke tests)
- Hugging Face access to the chosen LLaVA checkpoint (default: `llava-hf/llava-1.5-7b-hf`)

Create or activate your environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 1. Download COCO 2017
The training script expects the full COCO 2017 release (train/val images + annotations) under `data/raw/coco2017`. Use the helper:

```bash
python data/download_datasets.py --datasets coco2017
```

This downloads `train2017.zip`, `val2017.zip`, and `annotations_trainval2017.zip`, and extracts them into:

- `data/raw/coco2017/train2017`
- `data/raw/coco2017/val2017`
- `data/raw/coco2017/annotations/instances_train2017.json`
- `data/raw/coco2017/annotations/instances_val2017.json`

If you already have the files, point the training script to the existing paths with `--train-images`, `--train-ann`, `--val-images`, and `--val-ann`.

## 2. Choose a LLaVA backbone
`experiments/coco_llava_boxreg` wraps a LLaVA vision encoder. The hierarchy for resolving the model is:

1. `--model-id` CLI argument (e.g., `liuhaotian/llava-v1.5-7b`)
2. `LLAVA_MODEL_ID` environment variable (falls back to `llava-hf/llava-1.5-7b-hf`)
3. `LLAVA_LOCAL_PATH` environment variable (overrides both and is treated as a filesystem path)

If you have a locally converted checkpoint:

```bash
export LLAVA_LOCAL_PATH=/path/to/llava-weights
```

Otherwise pass a public HF repo id with `--model-id`.

## 3. Train the box regressor
Run the module entrypoint. Example (single epoch, small batch):

```bash
python -m experiments.coco_llava_boxreg.train \
  --model-id llava-hf/llava-1.5-7b-hf \
  --train-images data/raw/coco2017/train2017 \
  --train-ann data/raw/coco2017/annotations/instances_train2017.json \
  --val-images data/raw/coco2017/val2017 \
  --val-ann data/raw/coco2017/annotations/instances_val2017.json \
  --batch-size 4 \
  --epochs 1 \
  --checkpoint-dir experiments/coco_llava_boxreg/checkpoints
```

Key flags:
- `--train-backbone`: unfreezes the LLaVA vision tower.
- `--no-cls-head`: disables the optional classification head (regresses boxes only).
- `--max-train-samples` / `--max-val-samples`: subset COCO for debugging.
- `--device cuda:0` (or `cpu`) to override automatic device selection.

Checkpoints are written to `<checkpoint-dir>/<experiment-name>_best.pt` together with metadata (category list, CLI args, backbone id).

## 4. Run inference/evaluation
Given a saved checkpoint, evaluate on a split (default: `val`):

```bash
python -m experiments.coco_llava_boxreg.infer \
  --checkpoint experiments/coco_llava_boxreg/checkpoints/coco_llava_boxreg_baseline_best.pt \
  --split val \
  --batch-size 4
```

Outputs:
- Per-image JSON dumps in `results/coco_llava_boxreg/per_image/<experiment-name>/`
- Summary file `results/coco_llava_boxreg/<experiment-name>_summary.json` containing mean IoU and classification accuracy (if the head is enabled)

Use `--limit 128` to quickly sanity-check a smaller subset.

## 5. Troubleshooting
- **Missing COCO data**: ensure `train2017` exists. The dataset loader skips images missing from disk.
- **Processor errors**: upgrade `transformers` if the backbone introduces `LlavaNextProcessor`.
- **Out-of-memory**: reduce `--batch-size`, disable the classification head, or freeze the backbone (`omit --train-backbone`).

That’s it—once the dataset is in place and the model weights are accessible, you can iterate on experiments by tweaking the CLI flags above.
