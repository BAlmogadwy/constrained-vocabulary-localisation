# Official LVIS paired bootstrap

For input preparation and dependencies, see [REVIEWER_GUIDE.md](../REVIEWER_GUIDE.md). Optional numerical rerun from the repository root: `python analysis_records/lvis_paired_bootstrap.py --bootstrap 1000 --seed 12345`. New outputs go to `analysis_records/recomputed/`; the results below are the retained analysis, not a new rerun.

This separate artifact uses the same cached boxes as `lvis_semantics_sensitivity.json`, with the official LVIS 0.5.3 bbox evaluator, preserved federated metadata, and its default maximum of 300 detections per image globally. It does not modify the earlier COCO-style replay. All 396 images are included; Gemini boxes accepted by the fixed historical parser receive score 1.0, while YOLO retains its archived scores at inference floor 0.001.

| Comparison | AP difference | Paired percentile 95% interval |
|---|---:|---:|
| Gemini Flash minus YOLO | 0.083291 | [-0.018969, 0.168169] |
| Gemini Pro minus YOLO | 0.072909 | [-0.008550, 0.166160] |
| Gemini Flash minus Pro | 0.010382 | [-0.081705, 0.046768] |

Both Gemini models have higher point AP than YOLO, but these image-bootstrap intervals include zero. The bootstrap therefore does not establish superiority at the two-sided 95% interval level. It also does not distinguish the two Gemini models.

The bootstrap uses 1,000 paired image resamples with seed 12345. Each resample repeats complete per-image detection sequences in canonical source-image ID order and uses stable score sorting, retaining tied-score behavior. It reuses official evaluator matching and ignore flags, recomputes precision envelopes and 101-point recall integration, then averages the ten categories with nonignored ground truth present in each resample. Categories absent in a resample are omitted, matching evaluator semantics.

Validation reproduced official full-sample AP to an absolute error of at most 1.11e-16. A separately materialized bootstrap sample (seed 7421), with new image and annotation IDs and copied negative/non-exhaustive metadata, was evaluated with the official LVIS evaluator. Its AP agreed exactly with the cached replication for all three models. No weighted TP/FP shortcut was used.

Total runtime was 13.12 seconds, including validation; the 1,000 resamples required 12.13 seconds. JSON contains exact estimates, individual intervals, validation errors, input hashes, and script hash. Intervals describe image-sampling uncertainty conditional on this selected ten-category subset and retained predictions. They do not resample prompts/model calls or establish performance on full LVIS or categories absent from model pretraining.

Dependency: install `requirements-review.txt`, including `lvis==0.5.3`. The recorded run used a task-local `_deps` installation; portable scripts discover the installed package, with the removed `np.float` alias restored to `float` in memory for NumPy compatibility; evaluator source and scoring algorithm are unchanged. The script records hashes of the evaluator files and supporting scripts.

Official evaluator source: https://github.com/lvis-dataset/lvis-api
