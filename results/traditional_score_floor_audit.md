# Detector Score-Floor Audit (LVIS-unseen, constrained protocol)

**Question.** Were the traditional detector baselines depressed by a pre-evaluation
confidence threshold applied *before* COCOeval? (Referee §3.7.)

**Finding.** Yes — for YOLO-World (severely) and OWL-ViT (modestly); Grounding DINO
is insensitive. The "VLMs beat detectors" headline survives, but the detector table
must be corrected and the "more than double" wording dropped.

## Method

- The three detectors apply a confidence floor *at inference* before caching:
  Grounding DINO `box_threshold=0.25`, YOLO-World `conf=0.25`, OWL-ViT `score_threshold=0.20`.
  The cached outputs therefore contained only threshold survivors (~1.3 boxes/image),
  so re-scoring the cache alone cannot recover the recall curve.
- We regenerated all detector outputs with a near-zero floor (**0.001**; GD text_threshold
  kept at 0.25), on the identical 396 LVIS-unseen images, with the identical per-image
  candidate labels (reconstructed from the cached run so **only the threshold changed**),
  and re-ran COCOeval with the identical scope (396 images × 10 unseen categories, **maxDets=100**).

## Results

| Detector | Baseline AP (0.25/0.20) | Low-floor AP (0.001) | ΔAP | AP@0.5 base → low | Boxes/img base → low |
|---|---|---|---|---|---|
| Grounding DINO | 0.267 | **0.268** | +0.001 (flat) | 0.312 → 0.313 | 1.2 → 895 |
| YOLO-World | 0.221 | **0.347** | **+0.126 (+57%)** | 0.265 → 0.445 | 1.2 → 13 |
| OWL-ViT | 0.142 | **0.161** | +0.019 (+13%) | 0.223 → 0.266 | 1.5 → 314 |

**Best detector flips from Grounding DINO (0.267) to YOLO-World (0.347).**

## Control (proves ΔAP is threshold-only)

Filtering the low-floor outputs back to the original thresholds reproduces the published
baseline exactly — YOLO-World 0.2205, OWL-ViT 0.1418 (AP and AP@0.5 identical), Grounding
DINO 0.2684 with identical box count (488), the 0.001 gap being GPU float nondeterminism.
This confirms the low-floor run is a faithful superset of the baseline; the only variable
is the confidence floor.

## Interpretation (referee Case B)

- Frontier VLMs remain ahead: best VLM (Gemini 3.5 Flash 0.607, Gemini 3.1 Pro 0.565) vs
  best corrected detector (YOLO-World 0.347). Gemini lead margin drops from ~2.7× to ~1.6–1.7×.
- **Action items for the manuscript:**
  1. Correct Table 1: YOLO-World AP = 0.347 (strongest detector), report the score floor used.
  2. Drop "more than double"; state the corrected margin.
  3. Add the score-floor audit as a methodology/results subsection (threshold-free detector
     evaluation), mirroring the coordinate-decoding correction on the VLM side — both are
     evaluation-layer artefacts, and correcting both keeps the comparison internally consistent.
  4. Report full detector configs (checkpoint, floor, NMS IoU, input resolution).

## Reproduction

- GT: `data/processed/lvis_v1/val/lvis_v1_val_unseen.json` (from `prepare_zero_shot_splits.py` on LVIS v1 val).
- Labels: `data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json` (reconstructed from cached `labels_used`).
- Low-floor outputs: `results/raw/traditional_lowthr/`; config: `experiments/configs/lvis_unseen_traditional_lowthr.yaml`.
- Baseline (unchanged): `results/raw/traditional/`.
- Eval: `python experiments/compute_metrics.py --dataset lvis-unseen --eval-labels-file <dynamic labels> --traditional-dir <root>`.

## COCO-unseen replication (5000 imgs × 12 unseen cats, maxDets=100)

Same protocol re-run on COCO-unseen. Baseline computed from the cached (thresholded) outputs
(the paper reports no COCO detector numbers), low-floor at 0.001, filter-back control confirms
threshold-only (reproduces baseline: GD 0.487, YOLO 0.381, OWL 0.185).

| Detector | Baseline AP | Low-floor AP | ΔAP | (LVIS ΔAP for comparison) |
|---|---|---|---|---|
| Grounding DINO | 0.487 | 0.487 | +0.000 (flat) | +0.001 (flat) |
| YOLO-World | 0.382 | 0.414 | +0.032 (+8%) | +0.126 (+57%) |
| OWL-ViT | 0.185 | 0.213 | +0.028 (+15%) | +0.019 (+13%) |

**The finding replicates:** Grounding DINO is threshold-insensitive on both splits; OWL-ViT is
modestly penalized on both (~13–15%); YOLO-World is penalized on both but far more on LVIS's
rare categories than COCO's common ones — consistent with YOLO's *correct* boxes for rare
objects sitting at lower confidence, so more of them were cut by the 0.25 floor. Direction and
per-detector pattern are identical across datasets; only the YOLO magnitude is category-difficulty
dependent. Conclusion unchanged: report detector thresholds / use threshold-free AP, and the
pre-threshold most penalizes the fast detectors (YOLO/OWL), not Grounding DINO.
