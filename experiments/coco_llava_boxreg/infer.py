from __future__ import annotations

import argparse
import json
import pathlib
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - for older torch versions
    add_safe_globals = None

from utils.env import load_environment

from .dataset import CocoMainObjectDataset, coco_collate_fn
from .model import CocoLLaVABBoxRegressor
from .utils import (
    box_iou_xyxy,
    clamp_normalized_boxes,
    default_coco_paths,
    denormalize_boxes,
    load_image_processor,
    resolve_model_source,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the COCO LLaVA box regression head.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained checkpoint (.pt).")
    parser.add_argument("--model-id", type=str, default=None, help="Optional override for the LLaVA backbone.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (train or val).")
    parser.add_argument("--images", type=Path, default=None, help="Images directory for the target split.")
    parser.add_argument("--ann", type=Path, default=None, help="Annotation JSON for the target split.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of images for quick tests.")
    parser.add_argument("--experiment-name", type=str, default="coco_llava_boxreg_baseline")
    parser.add_argument("--output-dir", type=Path, default=Path("results/coco_llava_boxreg"))
    return parser.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> Dict[str, Path]:
    split = args.split.lower()
    if "train" in split:
        default_images, default_ann = default_coco_paths("train")
    else:
        default_images, default_ann = default_coco_paths("val")
    images = args.images or default_images
    ann = args.ann or default_ann
    limit = args.limit
    return {
        "images": Path(images).expanduser().resolve(),
        "ann": Path(ann).expanduser().resolve(),
        "limit": limit,
    }


def main() -> None:
    load_environment()
    args = parse_args()
    device = select_device(args.device)
    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if add_safe_globals is not None:
        safe_paths = {Path}
        for cls_name in ("PosixPath", "WindowsPath"):
            path_cls = getattr(pathlib, cls_name, None)
            if path_cls is not None:
                safe_paths.add(path_cls)
        add_safe_globals(list(safe_paths))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args") or {}
    saved_model_id = checkpoint.get("model_id")
    head_hidden_dim = saved_args.get("head_hidden_dim", 512)
    no_cls_head = saved_args.get("no_cls_head", False)

    model_id = resolve_model_source(args.model_id or saved_model_id)
    processor = load_image_processor(model_id)

    paths = resolve_dataset_paths(args)
    dataset = CocoMainObjectDataset(
        images_dir=paths["images"],
        annotation_file=paths["ann"],
        image_processor=processor,
        split=args.split,
        max_samples=paths["limit"],
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=coco_collate_fn,
    )

    num_classes = 0 if no_cls_head else len(dataset.category_names)
    model = CocoLLaVABBoxRegressor(
        model_id=model_id,
        head_hidden_dim=head_hidden_dim,
        num_classes=num_classes,
        freeze_backbone=True,
        device=str(device),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    per_image_dir = output_dir / "per_image" / args.experiment_name
    per_image_dir.mkdir(parents=True, exist_ok=True)

    total_iou = 0.0
    total = 0
    correct_cls = 0
    cls_samples = 0

    labels_for_json = dataset.category_names
    print(f"[infer] Writing per-image results to {per_image_dir}")

    for batch in data_loader:
        start = time.perf_counter()
        pixel_values = batch["pixel_values"].to(device)
        target_boxes = batch["target_boxes"]
        target_boxes_device = target_boxes.to(device)
        outputs = model(pixel_values)
        pred_boxes = clamp_normalized_boxes(outputs["pred_boxes"])
        latency = time.perf_counter() - start

        logits = outputs.get("logits")
        probs = torch.softmax(logits, dim=1) if logits is not None else None
        if logits is not None:
            preds = torch.argmax(logits, dim=1)
            correct_cls += int((preds == batch["target_labels"].to(device)).sum().item())
            cls_samples += batch["target_labels"].size(0)

        ious = box_iou_xyxy(pred_boxes, target_boxes_device)
        total_iou += float(ious.sum().item())
        total += target_boxes.size(0)

        pred_boxes_cpu = pred_boxes.detach().cpu()
        pixel_boxes = denormalize_boxes(pred_boxes_cpu, batch["image_sizes"])
        target_pixel_boxes = denormalize_boxes(target_boxes, batch["image_sizes"])
        for idx in range(len(batch["image_ids"])):
            image_id = batch["image_ids"][idx]
            image_path = batch["image_paths"][idx]
            norm_box = [float(value) for value in pred_boxes_cpu[idx].tolist()]
            pix_box = [float(value) for value in pixel_boxes[idx].tolist()]
            gt_norm_box = [float(value) for value in target_boxes[idx].tolist()]
            gt_pix_box = [float(value) for value in target_pixel_boxes[idx].tolist()]
            target_label_idx = int(batch["target_labels"][idx].item())
            target_label = labels_for_json[target_label_idx]
            if probs is not None:
                score_tensor, label_idx = torch.max(probs[idx], dim=0)
                pred_label_idx = int(label_idx.item())
                score_value = float(score_tensor.item())
            else:
                pred_label_idx = target_label_idx
                score_value = 1.0
            pred_label = labels_for_json[pred_label_idx]
            detection = {
                "label": pred_label,
                "score": score_value,
                "box_2d": norm_box,
                "box": pix_box,
                "box_normalized_xyxy": norm_box,
            }

            per_image_payload = {
                "model": args.experiment_name,
                "strategy": "box_regression",
                "image": image_path,
                "labels": labels_for_json,
                "detections": [detection],
                "latency_sec": latency / len(batch["image_ids"]),
                "raw_response": None,
                "target_box_normalized_xyxy": gt_norm_box,
                "target_box_xyxy": gt_pix_box,
                "target_label": target_label,
                "pred_label": pred_label,
                "pred_score": score_value,
            }
            out_path = per_image_dir / f"{image_id:012d}.json"
            out_path.write_text(json.dumps(per_image_payload, indent=2), encoding="utf-8")

    mean_iou = total_iou / max(total, 1)
    cls_acc = (correct_cls / cls_samples) if cls_samples else 0.0
    summary = {
        "checkpoint": str(ckpt_path),
        "model_id": model_id,
        "experiment_name": args.experiment_name,
        "split": args.split,
        "num_images": total,
        "mean_iou": mean_iou,
        "cls_acc": cls_acc,
    }
    summary_path = output_dir / f"{args.experiment_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[infer] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
