# Offline replay audit for the Neurocomputing revision

Run in PowerShell from the shared workspace:

```powershell
python C:\Users\user\zeroshot\Neurocomputing_submission\verification\replay_audit.py --bootstrap 1000
```

Python needs NumPy and pycocotools. The JSON records the versions used. No provider SDK, API credentials, model download, GPU inference or paid call is involved. The script reads original inputs and writes only `replay_results.json` beside itself. The original experimental files are preserved. Typical runtime on this workstation is approximately one minute, including both 1,000-resample bootstraps.

## Scope and provenance

This is a **new offline reevaluation of retained evidence**, rather than a claim to reproduce every ESWA aggregate. Original aggregate values are retained separately under each VLM's `legacy_aggregate` field. The source ZIP, annotations, original parser/evaluator code, every detector and router input record, and the audit script have SHA-256 records in the output. Absolute paths identify all inputs.

The retained `results/ablation_normalization.zip` contains full LVIS raw outputs for five current-roster models. Each has 396 image records. Gemini Flash and Pro each have 394 batch dictionaries and two string-format pilot responses. The historical normalization-ablation code ignored the string-format responses; this audit reparses all 396 records. The seven other current-roster VLMs and the headline COCO Gemini results do not have located complete raw evidence in this archive and are not presented as reproduced results.

Each response is parsed once using the historical parser, decoded with the submitted manuscript's specified per-model convention, clipped to actual dimensions, and restricted to its requested per-image candidate labels. Gemini uses normalized 0–1000 `yxyx`; the three OpenRouter models use the manuscript's assumed normalized 0–1000 `xyxy`. The latter conventions are inherited experimental assumptions, not independently established provider facts. Every LVIS dimension is at most 640 pixels, so the manuscript's explanation relying on images larger than 1,000 pixels is inapplicable to these LVIS data.

`conditions` separates archived decoded arrays, corrected raw decoding, and a deliberately incorrect pixel-coordinate interpretation. Each is evaluated with supplied confidence and uniform confidence. Uniform means score 1.0 **after acceptance by the fixed historical parser**, which rejects nonpositive supplied scores. A separate in-memory sensitivity probe removes only that score gate and records additional geometrically valid parsed boxes; it does not alter primary metrics. No newly recovered boxes were found in the five-model archive.

The VLM scoreboard is evaluated on all 396 LVIS images and ten categories (624 annotations). No recognition claim is inferred from category coverage. The label candidates are conditioned on ground truth, so this is constrained-vocabulary localization. The benchmark does not establish absence of categories from model pretraining.

The detector evidence comes from the separate `traditional_lowthr` inference records at floor 0.001. The router detector outputs come from a later retained live run. Their YOLO LVIS AP values differ slightly (approximately 0.0000035); these separate runs are kept distinct.

## Bootstrap and router repair

Bootstrap uses 1,000 paired image resamples, seed 12345, shared image multiplicities across the compared models, and percentile 95% intervals. It caches COCO image matches, then **replicates complete per-image detection sequences**, sorts scores stably, and recomputes precision envelopes. It preserves original image-ID order for the resampled multiset and within-image response order, including equal scores. Category AP is averaged over categories with nonignored ground truth in each resample, as in COCOeval; IoU 0.50–0.95, all areas, maxDets 100 per category/image.

Every cached accumulator is checked against ordinary COCOeval with all image weights one. For YOLO and Gemini Flash, a resample with duplicates is additionally materialized into a new in-memory COCO dataset and reevaluated; observed absolute errors are below 1e-12. The validation results are saved in JSON. `superiority_proportion` is a descriptive bootstrap fraction, not a posterior probability or a journal acceptance probability.

The original router evaluation shares 33 image IDs between LVIS and the 901-image COCO subset. The revised COCO analysis removes these images without changing routing parameters, predictions, fusion or scores, leaving 868 images. The original 901-image result and the revised result are both retained. The exclusion is **a post hoc repair**, and its confidence interval should be reported with that qualification. LVIS remains a development analysis because the routing threshold was selected there.

Router VLM-only primary metrics use uniform scores, matching `run_live_hybrid.vlm_to_simple`; a separately named sensitivity field reports supplied scores. Router fused boxes retain the saved fusion scores. This distinction prevents conflating the original main-roster scorer, which preserves supplied scores, with the router's uniform VLM-only scorer.

## JSON locations

- `vlm.<model>.conditions.raw_correct_decoder.uniform`: revised primary VLM AP/AP50.
- `vlm.<model>.conditions.raw_correct_decoder.supplied`: supplied-score sensitivity.
- `vlm.<model>.conditions.raw_pixel_decoder.uniform`: coordinate-decoding sensitivity.
- `detectors`: three detector metrics at floor 0.001 on all 396 LVIS images.
- `paired_lvis_bootstrap_uniform_vlm`: paired intervals for uniform-score Gemini vs YOLO.
- `router.coco_exclude_lvis_overlap`: 868-image router, detector and VLM-only metrics.
- `paired_coco_disjoint_router_bootstrap`: paired interval for repaired COCO router vs detector.
- `dataset.shared_image_ids`: exact overlap list.
- `input_sha256`: source file hashes.

Retained raw outputs support the local observations in this reevaluation. They do not establish that earlier publications made decoding mistakes, or that all weak localization results reflect failures of recognition or reasoning.
