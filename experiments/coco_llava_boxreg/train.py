from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.env import load_environment

from .dataset import CocoMainObjectDataset, coco_collate_fn
from .model import CocoLLaVABBoxRegressor
from .utils import (
    box_iou_xyxy,
    clamp_normalized_boxes,
    default_coco_paths,
    load_image_processor,
    resolve_model_source,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LLaVA-based box regressor on COCO.")
    parser.add_argument("--model-id", type=str, default=None, help="HF model id or local path for the LLaVA backbone.")
    parser.add_argument("--experiment-name", type=str, default="coco_llava_boxreg_baseline")
    parser.add_argument("--train-images", type=Path, default=None, help="COCO train images directory.")
    parser.add_argument("--train-ann", type=Path, default=None, help="COCO train annotation JSON path.")
    parser.add_argument("--val-images", type=Path, default=None, help="COCO val images directory.")
    parser.add_argument("--val-ann", type=Path, default=None, help="COCO val annotation JSON path.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-cls-head", action="store_true", help="Disable the optional classification head.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("experiments/coco_llava_boxreg/checkpoints"))
    parser.add_argument("--val-interval", type=int, default=1, help="Number of epochs between validations.")
    parser.add_argument("--train-backbone", action="store_true", help="Enable fine-tuning of the vision backbone.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr-step-size", type=int, default=5)
    parser.add_argument("--lr-gamma", type=float, default=0.2)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    return parser.parse_args()


def prepare_paths(args: argparse.Namespace) -> Dict[str, Path]:
    train_images = args.train_images
    train_ann = args.train_ann
    if train_images is None or train_ann is None:
        default_images, default_ann = default_coco_paths("train")
        train_images = train_images or default_images
        train_ann = train_ann or default_ann

    val_images = args.val_images
    val_ann = args.val_ann
    if val_images is None or val_ann is None:
        default_images, default_ann = default_coco_paths("val")
        val_images = val_images or default_images
        val_ann = val_ann or default_ann

    paths = {
        "train_images": Path(train_images),
        "train_ann": Path(train_ann),
        "val_images": Path(val_images),
        "val_ann": Path(val_ann),
    }
    for key, value in paths.items():
        paths[key] = value.expanduser().resolve()
    return paths


def serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def evaluate(
    model: CocoLLaVABBoxRegressor,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_iou = 0.0
    total = 0
    correct_cls = 0
    cls_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            target_boxes = batch["target_boxes"].to(device)
            target_labels = batch["target_labels"].to(device)
            outputs = model(pixel_values)
            pred_boxes = clamp_normalized_boxes(outputs["pred_boxes"])
            ious = box_iou_xyxy(pred_boxes, target_boxes)
            total_iou += float(ious.sum().item())
            total += target_boxes.size(0)

            logits = outputs.get("logits")
            if logits is not None:
                preds = torch.argmax(logits, dim=1)
                correct_cls += int((preds == target_labels).sum().item())
                cls_samples += target_labels.size(0)

    mean_iou = total_iou / max(total, 1)
    cls_acc = (correct_cls / max(cls_samples, 1)) if cls_samples else 0.0
    return {"mean_iou": mean_iou, "cls_acc": cls_acc, "samples": total}


def main() -> None:
    load_environment()
    args = parse_args()
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    model_source = resolve_model_source(args.model_id)
    processor = load_image_processor(model_source)
    paths = prepare_paths(args)

    train_dataset = CocoMainObjectDataset(
        images_dir=paths["train_images"],
        annotation_file=paths["train_ann"],
        image_processor=processor,
        split="train",
        max_samples=args.max_train_samples,
    )
    val_dataset = CocoMainObjectDataset(
        images_dir=paths["val_images"],
        annotation_file=paths["val_ann"],
        image_processor=processor,
        split="val",
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=coco_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=coco_collate_fn,
    )

    model = CocoLLaVABBoxRegressor(
        model_id=model_source,
        head_hidden_dim=args.head_hidden_dim,
        num_classes=0 if args.no_cls_head else len(train_dataset.category_names),
        freeze_backbone=not args.train_backbone,
        device=str(device),
    )
    model.to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Enable --train-backbone or ensure heads are trainable.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    if args.lr_gamma < 0.999:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.lr_step_size), gamma=args.lr_gamma)
    else:
        scheduler = None

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"{args.experiment_name}_best.pt"

    best_val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_cls_loss = 0.0
        start_time = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            target_boxes = batch["target_boxes"].to(device)
            target_labels = batch["target_labels"].to(device)

            outputs = model(pixel_values)
            pred_boxes = clamp_normalized_boxes(outputs["pred_boxes"])
            box_loss = F.smooth_l1_loss(pred_boxes, target_boxes, beta=0.1)
            loss = box_loss
            cls_loss_value = 0.0
            logits = outputs.get("logits")
            if logits is not None:
                cls_loss = F.cross_entropy(logits, target_labels)
                loss = loss + args.cls_loss_weight * cls_loss
                cls_loss_value = float(cls_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_box_loss += float(box_loss.item())
            epoch_cls_loss += cls_loss_value

            if step % args.log_every == 0 or step == len(train_loader):
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} box={box_loss.item():.4f} cls={cls_loss_value:.4f} lr={lr:.2e}"
                )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.perf_counter() - start_time
        num_batches = len(train_loader)
        print(
            f"[epoch] {epoch}/{args.epochs} | "
            f"loss={epoch_loss/num_batches:.4f} "
            f"box={epoch_box_loss/num_batches:.4f} "
            f"cls={epoch_cls_loss/max(num_batches,1):.4f} "
            f"time={elapsed:.1f}s"
        )

        if epoch % args.val_interval == 0:
            metrics = evaluate(model, val_loader, device)
            print(
                f"[val] epoch={epoch} :: mean_iou={metrics['mean_iou']:.4f} "
                f"cls_acc={metrics['cls_acc']:.4f} samples={metrics['samples']}"
            )
            if metrics["mean_iou"] > best_val_iou:
                best_val_iou = metrics["mean_iou"]
                payload = {
                    "model_state": model.state_dict(),
                    "model_id": model.model_source,
                    "experiment_name": args.experiment_name,
                    "epoch": epoch,
                    "val_iou": metrics["mean_iou"],
                    "val_cls_acc": metrics["cls_acc"],
                    "category_names": train_dataset.category_names,
                    "args": serialize_args(args),
                }
                torch.save(payload, best_ckpt_path)
                print(f"[checkpoint] Saved best model to {best_ckpt_path}")

    if not best_ckpt_path.exists():
        # Save final weights if validation never triggered.
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_id": model.model_source,
                "experiment_name": args.experiment_name,
                "epoch": args.epochs,
                "category_names": train_dataset.category_names,
                "args": serialize_args(args),
            },
            best_ckpt_path,
        )
        print(f"[checkpoint] Saved final model to {best_ckpt_path}")


if __name__ == "__main__":
    main()
