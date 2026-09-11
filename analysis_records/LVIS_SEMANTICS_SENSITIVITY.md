# LVIS annotation-semantics sensitivity

This separate reevaluation preserves the frozen `replay_results.json`. It uses the official LVIS 0.5.3 API on the same 396-image, 10-category subset and unchanged cached predictions. It is not a full LVIS benchmark result.

The subset contains 23 images with selected categories marked not exhaustive. LVIS ignores unmatched detections for those image/category pairs and removes detections in categories whose presence or absence is unverified. COCO-style evaluation does not implement these rules.

| Model | Frozen COCO AP | LVIS AP, global 300 | LVIS AP50, global 300 | LVIS AP, compatible 100/category |
|---|---:|---:|---:|---:|
| gemini-3.5-flash | 0.548531 | 0.601160 | 0.754141 | 0.601160 |
| gemini-3.1-pro | 0.547422 | 0.590779 | 0.735389 | 0.590779 |
| qwen3-vl-235b-openrouter | 0.348216 | 0.355093 | 0.475450 | 0.355093 |
| yolo_world | 0.346508 | 0.517869 | 0.657823 | 0.517869 |
| grounding_dino | 0.268433 | 0.278948 | 0.325539 | 0.278948 |
| owl_vit | 0.160950 | 0.214853 | 0.366044 | 0.214861 |

Official default: at most 300 detections per image across categories, before federated filtering. The compatible-cap sensitivity first applies a stable top-100 cap per category/image, then disables the LVIS global cap; it isolates annotation semantics relative to frozen COCO-style AP and is not the official LVIS cap. Uniform-score ties retain source order. Detector scores remain native.

LVIS 0.5.3 refers to `np.float`; this script restores that removed alias as `float` in memory without changing evaluator source or scoring logic. Source input and evaluator file hashes are recorded in JSON. AP and AP50 are fractions. No bootstrap was added to this bounded sensitivity.

See [REVIEWER_GUIDE.md](../REVIEWER_GUIDE.md) for input preparation. Optional numerical rerun from the repository root: `python analysis_records/lvis_semantics_sensitivity.py`, after installing `requirements-review.txt`. New outputs go to `analysis_records/recomputed/`. The results above remain the original saved analysis.

Official API: https://github.com/lvis-dataset/lvis-api

Do not reuse the frozen COCO bootstrap intervals as LVIS intervals. This sensitivity does not resolve ground-truth-conditioned candidate labels, historical model provenance, or pretraining overlap.
