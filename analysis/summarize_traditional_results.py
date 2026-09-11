"""Minimal reconstruction of the (uncommitted) traditional-results summarizer.

`experiments/run_traditional.py` imports `load_json`, `summarize_file`, and
`render_markdown` from here purely to print/write a cosmetic Markdown summary at
the end of a run. None of the per-image detection outputs or downstream COCOeval
metrics depend on this module. Implemented locally for the score-floor audit
because the original `analysis` package was never committed to the repo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_file(data: Dict) -> Dict:
    """Summarize one aggregate detector-output JSON."""
    results = data.get("results", []) or []
    n_images = len(results)
    n_detections = sum(len(r.get("detections") or []) for r in results)
    images_with = sum(1 for r in results if (r.get("detections") or []))
    summary = data.get("summary", {}) or {}
    return {
        "model": data.get("model"),
        "device": data.get("device"),
        "images": n_images,
        "images_with_detections": images_with,
        "detections": n_detections,
        "avg_detections_per_image": round(n_detections / n_images, 3) if n_images else None,
        "avg_latency_sec": summary.get("avg_latency_sec"),
        "wall_time_sec": summary.get("wall_time_sec"),
    }


def render_markdown(summaries: List[Dict]) -> str:
    header = (
        "| model | device | images | imgs w/ det | detections | avg det/img "
        "| avg latency (s) | wall (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for s in summaries:
        lines.append(
            f"| {s.get('model')} | {s.get('device')} | {s.get('images')} "
            f"| {s.get('images_with_detections')} | {s.get('detections')} "
            f"| {s.get('avg_detections_per_image')} | {s.get('avg_latency_sec')} "
            f"| {s.get('wall_time_sec')} |"
        )
    return "\n".join(lines)
