# Offline Late-Fusion Hybrid — LVIS + COCO + Sensitivity (cached boxes only)

Detector stage = YOLO-World (score-floor-corrected). VLM stage = Qwen3-VL-8B-Instruct
(best cached serverless VLM, coordinate-decoded). Validator = class-aware IoU agreement
(corroborated boxes boosted). Router = escalate to VLM iff detector max-confidence < τ
(observable, no GT). Evaluated on POSITIVE images (>=1 unseen GT) the VLM covers, so
detector-only / VLM-only / fused are all on the identical image set. Controls: detector-only
and VLM-only reproduce their standalone evals; VLM decode validated by pixel-mode collapse.

## Headline: fusion helps, but the size of the win depends on the detector–VLM gap

| Split (images) | Detector | VLM | Fused-all | Router peak | Oracle ceiling |
|---|---|---|---|---|---|
| LVIS-unseen (396) | 0.347 | 0.317 | **0.476 (+37%)** | 0.481 @ 26% esc | 0.532 |
| COCO-unseen (438) | 0.486 | 0.308 | **0.491 (+1%)** | 0.500 @ 63% esc | 0.535 |

- **LVIS (rare categories):** detector ≈ VLM, and they are complementary → fusion is a large
  win (+0.13 AP, +37%). Escalating 26% of images gives the peak.
- **COCO (common categories):** the detector already dominates the VLM (0.486 vs 0.308), so
  fusion adds almost nothing on average (+0.005). The oracle ceiling (0.535) shows some
  headroom exists, but detector-confidence routing does not capture it — when the detector is
  already strong, its confidence is a poor "escalate" signal.

### Evidence-backed deployment policy (referee Table 6, now with numbers)
- Detector ≈ VLM in strength (hard / rare / long-tail regime) → **the hybrid pays off**; route by
  detector confidence, escalate ~1 in 4 images, gain ~+0.13 AP at a fraction of VLM latency/cost.
- Detector ≫ VLM (easy / common regime) → **use the detector alone**; fusion is not worth the
  VLM cost. This is itself a useful, honest intelligent-system recommendation.

## Fusion sensitivity (LVIS) — fused-all AP over (VLM prior p_v, agreement boost)

|  p_v \ boost | 0.00 | 0.15 | 0.30 | 0.45 |
|---|---|---|---|---|
| 0.20 | 0.366 | 0.477 | 0.524 | 0.557 |
| 0.30 | 0.363 | 0.474 | 0.523 | 0.557 |
| 0.40 | 0.361 | 0.469 | 0.476 | 0.556 |
| 0.50 | 0.355 | 0.467 | 0.474 | 0.495 |
| 0.60 | 0.352 | 0.465 | 0.471 | 0.492 |

- **Robust:** the hybrid beats detector-only (0.347) in *every* cell.
- **The validator is the mechanism:** naive union (boost=0) barely helps (~0.36); rewarding
  detector–VLM agreement drives the gain (up to 0.56). Lower VLM-only prior is mildly better.
- Reported default (p_v=0.40, boost=0.30 → 0.476) is mid-range, not tuned to GT.

## Caveats / next steps
- Serverless roster (Qwen3-VL-8B); a stronger frontier VLM would raise VLM-only and likely the
  fusion gain — pipeline is model-agnostic (drop in decoded frontier boxes).
- COCO VLM coverage is partial; the hybrid is scoped to the 438 positive images Qwen covers.
- Reproduce: `python experiments/hybrid_eval.py --dataset {lvis|coco} [--sensitivity]`
  → `results/hybrid_{lvis,coco}.json`.
