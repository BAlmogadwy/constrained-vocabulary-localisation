"""Offline protocol audit and isolated correction utilities (standard library only).

This module never imports provider clients, reads credentials, or calls a network.
It does not modify the original benchmark. Utilities are proposed controls, not
an assertion that the archived experiments used the corrected protocol.

Run: python protocol_audit.py --repo C:/Users/user/zeroshot/zero-shot-detection-benchmark
An optional --output writes the audit JSON; without it the report goes to stdout.

Fresh inference is required to test shuffled candidate order or a corrected
iterative prompt (explicit label, coordinate units, and multiple-instance rule).
Rescoring retained boxes and excluding overlapping evaluation images are offline
operations. Re-parsing bare boxes is defensible only with retained prompt metadata
identifying the queried class and coordinate convention. No class is inferred
from ground truth, candidate order, or a response's coordinates.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile


def shuffled_candidates(options: list[str], *, seed: int, item_id: str) -> list[str]:
    """Deterministic permutation independent of the input/positive-label order.

    The membership of a GT-conditioned candidate set remains GT-conditioned.
    This removes the ordering channel only. The seed must be fixed before viewing
    results; use repeated predeclared seeds to measure residual order sensitivity.
    """
    if not options or any(not isinstance(x, str) or not x.strip() for x in options):
        raise ValueError("Candidates must be nonempty strings")
    canonical = sorted(set(options))
    def key(label: str) -> tuple[bytes, str]:
        payload = json.dumps(["candidate-order-v1", seed, str(item_id), label],
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).digest(), label
    return sorted(canonical, key=key)


@dataclass
class ParseResult:
    status: str  # valid, empty, or failure
    detections: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a coordinate or score")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite number")
    return number


def _validate_detection(entry: Any, required_label: str | None) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Detection must be an object")
    label = entry.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Missing label")
    if required_label is not None and label != required_label:
        raise ValueError("Response label does not match retained prompt label")
    box = entry.get("box_2d")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("box_2d must contain four coordinates")
    coords = [_number(x) for x in box]
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise ValueError("Box has nonpositive area")
    detection = {"label": label, "box_2d": coords}
    score = entry.get("score", entry.get("confidence"))
    if score is not None:
        score = _number(score)
        if score < 0:
            raise ValueError("Negative score")
        # Retain zero-confidence boxes; score policy must not silently delete them.
        detection["score"] = score
    return detection


def parse_detection_response(text: str | None, *,
                             prompt_metadata: dict[str, Any] | None = None) -> ParseResult:
    """Strict schema validation with an explicit adapter for a known single label.

    Bare [x1,y1,x2,y2] or [x1,y1,x2,y2,score] responses require metadata with
    single_label and coordinate_mode='pixel'|'normalized_1000_xyxy'|
    'normalized_1000_yxyx'|'normalized_1_xyxy'. This function validates and retains
    coordinates; it does not infer units or transform them. Transform before AP.
    JSON null, absent/malformed text and invalid boxes differ from a valid [].
    Any invalid item fails the response rather than silently discarding it.
    """
    if text is None:
        return ParseResult("failure", reason="missing_response")
    if not isinstance(text, str) or not text.strip():
        return ParseResult("failure", reason="empty_text")
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return ParseResult("failure", reason="malformed_json")
    if payload is None:
        return ParseResult("failure", reason="json_null")
    metadata = prompt_metadata or {}
    label = metadata.get("single_label")
    required_label = label if isinstance(label, str) and label.strip() else None
    if isinstance(payload, list) and len(payload) in (4, 5) and all(
            isinstance(x, (float, int)) and not isinstance(x, bool) for x in payload):
        modes = {"pixel", "normalized_1000_xyxy", "normalized_1000_yxyx", "normalized_1_xyxy"}
        if required_label is None or metadata.get("coordinate_mode") not in modes:
            return ParseResult("failure", reason="missing_prompt_metadata_for_bare_box")
        payload = {"label": required_label, "box_2d": payload[:4],
                   **({"score": payload[4]} if len(payload) == 5 else {})}
    if isinstance(payload, dict):
        items = payload.get("detections", [payload])
    else:
        items = payload
    if not isinstance(items, list):
        return ParseResult("failure", reason="invalid_schema")
    if not items:
        return ParseResult("empty")
    try:
        detections = [_validate_detection(x, required_label) for x in items]
    except (ValueError, TypeError, OverflowError) as exc:
        return ParseResult("failure", reason=f"invalid_detection: {exc}")
    return ParseResult("valid", detections)


def parse_counters(results: list[ParseResult]) -> dict[str, int]:
    counts = Counter(x.status for x in results)
    return {"total": len(results), "valid_nonempty": counts["valid"],
            "valid_empty": counts["empty"], "failure": counts["failure"]}


def apply_score_policy(detections: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    """Copy boxes using uniform=1 or native score/confidence with missing=1.

    Uniform/native are alternatives, not post-hoc choices based on whichever wins.
    Native retains zero scores; neither policy reconstructs previously lost boxes.
    """
    if policy not in {"uniform", "native"}:
        raise ValueError("Policy must be uniform or native")
    output = []
    for det in detections:
        score = 1.0 if policy == "uniform" else det.get("score", det.get("confidence", 1.0))
        if score is None:
            score = 1.0
        score = _number(score)
        if score < 0:
            raise ValueError("Negative score")
        copied = dict(det)
        copied.pop("confidence", None)
        copied["score"] = score
        output.append(copied)
    return output


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _response_text(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]
    parts = [c["text"] for item in (raw.get("output") or []) if isinstance(item, dict)
             for c in (item.get("content") or []) if isinstance(c, dict) and isinstance(c.get("text"), str)]
    return "\n".join(parts) if parts else None


def _oracle_audit(repo: Path) -> dict[str, Any]:
    ann = _read(repo / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json")
    labels = _read(repo / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json")["labels_per_image"]
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    filenames = {im["id"]: f"{im['id']:012d}.jpg" for im in ann["images"]}
    by_image = defaultdict(list)
    for a in ann["annotations"]:
        by_image[a["image_id"]].append(a)
    # Execute only the pure historical selector AST. Never import the module:
    # importing it loads environment variables and provider configuration.
    source = (repo / "experiments/run_oracle_localisation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_subset")
    namespace: dict[str, Any] = {}
    exec("from __future__ import annotations\n" + ast.unparse(selector), namespace)
    selected = namespace["select_subset"](cats, filenames, by_image, labels, 40)
    crops = []
    unavailable = []
    for iid in selected:
        filename = filenames[iid]
        options = labels[filename]["labels"]
        for a in by_image[iid]:
            true_label = cats[a["category_id"]]
            if true_label not in options:
                continue
            if not (repo / "data/subsets/lvis_v1_val_unseen/images" / filename).exists():
                unavailable.append(a["id"])
                continue
            crops.append({"image_id": iid, "annotation_id": a["id"], "true_label": true_label,
                          "first_option": options[0], "n_options": len(options)})
    n = len(crops)
    correct = sum(c["true_label"] == c["first_option"] for c in crops)
    return {"selected_images": len(selected), "reconstructed_crops_with_source_images": n,
            "unavailable_annotation_ids": unavailable, "first_option_correct": correct,
            "first_option_accuracy": correct / n if n else None,
            "uniform_random_expected_accuracy": sum(1 / c["n_options"] for c in crops) / n if n else None,
            "positive_prefix_images": sum(v["labels"][:len(v["positives"])] == v["positives"] for v in labels.values()),
            "all_label_images": len(labels), "crop_manifest": crops,
            "per_crop_raw_directory_present": (repo / "results/raw/oracle/oracle_lvis_2026").exists(),
            "interpretation": "Candidate-order leakage is confirmed; historical recognition accuracy does not establish the claimed recognition upper bound. Shuffling membership order requires new inference."}


def _ablation_audit(repo: Path) -> dict[str, Any]:
    runs = []
    for path in sorted((repo / "results/ablation/prompt_b").glob("*/gpt-5/gpt5_lvis_unseen.json")):
        data = _read(path)
        records = data["results"]
        counter: Counter = Counter()
        parse_results = []
        for rec in records:
            raw = rec.get("raw_response")
            counter["images"] += 1
            counter["old_parsed_ok_no_top_level_error"] += int(not (isinstance(raw, dict) and bool(raw.get("error"))))
            counter["images_with_zero_retained_detections"] += int(not rec.get("detections"))
            if not isinstance(raw, dict):
                counter["missing_raw_object"] += 1
                continue
            identification = raw.get("identification")
            counter["identification_without_output_text"] += int(not _response_text(identification))
            if isinstance(identification, dict):
                counter["identification_status_incomplete"] += int(identification.get("status") == "incomplete")
                details = identification.get("incomplete_details") or {}
                counter["identification_token_limit_incomplete"] += int(details.get("reason") == "max_output_tokens")
                counter["identification_error"] += int(bool(identification.get("error")))
            counter["images_without_localization_calls"] += int(not raw.get("localization"))
            for loc in raw.get("localization", []):
                text = _response_text(loc)
                result = parse_detection_response(text)  # No retained per-call prompt label/units; never infer them.
                parse_results.append(result)
                counter["localization_calls"] += 1
                if result.reason == "missing_prompt_metadata_for_bare_box":
                    counter["bare_box_rejected_by_original_labelled_dictionary_schema"] += 1
                if result.reason:
                    counter["reason:" + result.reason] += 1
        runs.append({"file": str(path.relative_to(repo)), "model": data.get("model"),
                     "recorded_num_images": data.get("num_images"), "counts": dict(counter),
                     "strict_parse_without_inferred_metadata": parse_counters(parse_results)})
    return {"historical_100_image_gpt5_runs": runs,
            "manuscript_reduced2026_raw_directory_present": (repo / "results/raw/vlm/ablation_lvis_reduced_2026").exists(),
            "interpretation": "Historical 100-image GPT-5 runs are not the manuscript's reduced 2026-model runs. Their incomplete identification outputs and absence of localization calls must not be attributed to the downstream bare-array parser defect. No-error transport counts cannot be described as schema-parse success. Bare arrays need retained class and unit metadata to recover; corrected prompting and token budgets need new inference."}


def _overlap_audit(repo: Path) -> dict[str, Any]:
    base = repo / "results/raw/live_hybrid"
    ids = {dataset: {int(p.stem) for p in (base / dataset / "per_image").glob("*.json")}
           for dataset in ("lvis", "coco")}
    overlap = sorted(ids["lvis"] & ids["coco"])
    return {"source": "Actual retained live-router per_image records, not all 5000 COCO dynamic-label rows",
            "lvis_images": len(ids["lvis"]), "coco_images": len(ids["coco"]),
            "shared_image_count": len(overlap), "shared_image_ids": overlap,
            "coco_images_after_excluding_lvis": len(ids["coco"] - ids["lvis"]),
            "interpretation": "These evaluated datasets are not image-disjoint. Existing boxes can be rescored on the 868 nonoverlapping COCO images without new inference; all related AP, latency, routing rates and uncertainty estimates must use that same subset."}


def _archives(repo: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted((repo / "results").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            raw_sample = next((n for n in names if n.endswith(".json") and "/per_image/" in n), None)
            result.append({"file": str(path.relative_to(repo)), "entries": len(names),
                           "oracle_entries": [n for n in names if "oracle" in n.lower()],
                           "sample_per_image_json": raw_sample,
                           "sample_has_raw_response": bool(raw_sample and _read_zip_raw(archive, raw_sample))})
    return result


def _read_zip_raw(archive: zipfile.ZipFile, name: str) -> bool:
    data = json.loads(archive.read(name))
    return isinstance(data, dict) and data.get("raw_response") is not None


def build_audit(repo: Path) -> dict[str, Any]:
    return {"repository": str(repo.resolve()), "audit_mode": "offline; original files unchanged",
            "oracle": _oracle_audit(repo), "iterative_ablation": _ablation_audit(repo),
            "evaluation_overlap": _overlap_audit(repo), "retained_archives": _archives(repo),
            "score_policy": {"executed_code": "experiments/compute_metrics.py:prepare_coco_detections preserves det.score; missing VLM score defaults to 1.0",
                             "parser": "models/vlm/parsing.py:normalize_detection preserves score/confidence and discards score <= 0",
                             "offline_salvage": "Recompute native and uniform AP from retained outputs under an explicit common policy. Recover score-zero boxes from raw text if evaluating a policy that would keep them. This audit does not recompute AP."},
            "fresh_inference_required": ["Recognition with predeclared shuffled candidate order and a fixed candidate-set definition; include first-option and random baselines.",
                                         "Iterative comparison with explicit class, coordinate units and multiple-instance instructions, cache keys including complete prompt/configuration.",
                                         "Any missing 2026 raw inference needed to support headline numbers unless independently recoverable from a verified archive."],
            "offline_actions": ["Rescore all retained boxes with documented confidence policy and a declared sensitivity comparison.",
                                "Rescore COCO excluding the 33 overlapping images, and regenerate all dependent summaries and confidence intervals.",
                                "Report transport success, valid-empty response, valid-nonempty response and schema failure separately.",
                                "Retain original results as historical, do not relabel them as corrected-protocol results."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_audit(args.repo), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)


if __name__ == "__main__":
    main()
