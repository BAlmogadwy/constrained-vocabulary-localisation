# Retained inference provenance manifest

Prepared 9 September 2026 from selected local result, manifest, source and configuration files. No credentials, environment files or environment values were read. No model client was imported and no provider was contacted. Paths below are relative to the original `zero-shot-detection-benchmark` directory.

## Evidence boundaries

Provider response metadata, a retained run manifest, and current source defaults provide different levels of evidence. A source-derived prompt reconstruction does not prove the historical per-request prompt. This manifest documents the distinction rather than filling missing fields by assumption.

## Principal LVIS runs

All three principal runs have 396 records under `ablation_normalization/ON_fromraw/lvis/per_image/<model>/` in `results/ablation_normalization.zip`. Each retains `model`, `strategy`, `image`, `labels`, `detections`, `latency_sec`, and `raw_response`; the strategy is `single_query`. No full request/prompt or per-record inference-start timestamp is retained in these wrappers.

| Archive model | Independent response evidence | Timestamp evidence |
|---|---|---|
| `gemini-3.5-flash` | 394 dictionary responses identify `modelVersion=gemini-3.5-flash`; two responses are strings without endpoint metadata | Inference timestamps not present in inspected response records |
| `gemini-3.1-pro` | 394 dictionary responses identify `modelVersion=gemini-3.1-pro-preview`; two responses are strings without endpoint metadata | Inference timestamps not present in inspected response records |
| `qwen3-vl-235b-openrouter` | 396 dictionary responses identify `qwen/qwen3-vl-235b-a22b-instruct` | Provider `created` values span 2026-06-25T23:55:09+00:00 to 2026-06-27T13:47:24+00:00 |

Provider-created ranges are retained response metadata, not measured end-to-end run intervals. File and ZIP modification dates are not treated as inference dates.

## Exact recovered prompt template

The following template is recorded verbatim in `results/live_hybrid_manifest.json`:

```text
Detect the following objects in the image: <CLASSES>. For each detected object, return JSON in the form [{"box_2d": [x1, y1, x2, y2], "label": "class_name", "confidence": score}, ...]. Use integer pixel coordinates relative to the input image. Return only JSON.
```

The stored prompt SHA-256 is `1e86f40e3f4df15162620a79ec48ffd343c7410c6b1cb70289e40af37c93f741`. The recomputed UTF-8 SHA-256 is `1e86f40e3f4df15162620a79ec48ffd343c7410c6b1cb70289e40af37c93f741`.

The placeholder `<CLASSES>` denotes the candidate names joined by comma and space. The current retained `_prompt` functions in `experiments/vlm_batch.py` and `models/vlm/openai_compatible_detector.py` both reconstruct this exact template. That is source-derived evidence for the Gemini batch and Qwen OpenAI-compatible paths; it is not a recovered request log for every principal response. Historical overrides, code changes and pilot differences remain unverified. The older `models/vlm/qwen_detector.py` has a different prompt and should not be substituted as proof of the Qwen3 OpenRouter requests.

The two frontier batch YAMLs listed below identify the Gemini models, `single_query`, no dataset limit, and the LVIS dynamic-label mapping. The serverless-refresh YAML identifies the Qwen3 OpenRouter model and the same mapping. Source constructors permit environment overrides. Effective historical temperature, output-token budgets, retry settings, full system messages, image encoding/preprocessing settings and request endpoints are not established by these response records; source defaults are not certified historical settings.

## Router manifest and timestamps

- Manifest created: `2026-07-07T06:41:17.053274+00:00`.
- Manifest updated: `2026-07-07T07:41:07.621174+00:00`.
- Recorded VLM slug: `qwen/qwen3-vl-235b-a22b-instruct`.
- These manifest timestamps do not establish the first and last inference time; in particular, later VLM-only records extend beyond the manifest update.
- Fixed routing parameters: threshold `tau=0.4`, unmatched VLM score `p_v=0.4`, score boost `0.3`, matching IoU `0.5`; detector is described as YOLO-World at score floor `0.001` on local GPU. Failure policy is detector-only fallback after the retry budget.
- Historical manifest versions: torch `2.3.1+cu121`, ultralytics `8.4.72`, requests `2.32.3`, pycocotools recorded as `unavailable`. These are not the new replay environment versions.

The following ranges were recovered from nested numeric provider `created` fields in the retained JSON responses. For routed outputs, only records containing an actual VLM response contribute a timestamp; this is not the detector-only image processing interval.

| Response directory | Total records | Provider-created range, UTC |
|---|---:|---|
| `results/raw/live_hybrid/lvis/per_image/` | 396 | 2026-07-07T06:41:21+00:00 to 2026-07-07T06:50:18+00:00 |
| `results/raw/live_hybrid/lvis/vlm_only/` | 396 | 2026-07-07T06:50:30+00:00 to 2026-07-07T07:17:22+00:00 |
| `results/raw/live_hybrid/coco/per_image/` | 901 | 2026-07-07T07:17:38+00:00 to 2026-07-07T07:40:30+00:00 |
| `results/raw/live_hybrid/coco/vlm_only/` | 901 | 2026-07-07T07:41:09+00:00 to 2026-07-07T09:10:53+00:00 |

## LVIS annotation and candidate audit

- Retained subset: 396 images, 624 annotations and 10 categories.
- Selected categories flagged non-exhaustive: 23 image/category pairs across 23 images.
- Candidate distractors, computed as each stored `labels` set minus its stored `positives` set: 1584 image/category entries. Of these, 17 occur in the image's official `neg_category_ids`; 1567 do not. The latter are not thereby proven visually present or absent.
- The stored dynamic-label entries contain `labels` and `positives`; they do not have a `negatives` field. The current generator samples distractors from classes without retained annotations, not exclusively verified negatives.
- COCO-style replay does not implement LVIS federated/non-exhaustive handling. Candidate information and evaluator semantics are separate limitations.

## Inspected-file fingerprints

These SHA-256 values identify the files inspected on the preparation date, not a historical executed-code revision. Individual router response hashes are covered by the full replay audit.

- `results/ablation_normalization.zip`: `53519642b00b65f1a553861c643f3de1ae94b55e2466bd97262800fef851fd5f`
- `results/live_hybrid_manifest.json`: `7592363737f609515925cf02f4fbeb602031a03a38e40c70475a7deb262ea54d`
- `experiments/vlm_batch.py`: `5379585a5a8624c917530a7a55b6300074a074c1880be92bf74bf902329f1aa0`
- `models/vlm/openai_compatible_detector.py`: `4325328ba85122befbbf82cc6888518b7c23e64f27b9ba0dc2b3f40d7d59fe4d`
- `experiments/configs/lvis_unseen_frontier_batch_cheap_full_2026.yaml`: `6df9797ad63250542f3dc863c33225c143033191bf7a2ae634424a4b31e8a2ab`
- `experiments/configs/lvis_unseen_frontier_batch_expensive_full_2026.yaml`: `2b3f94d1bbd4cab5e39411e6a9eee4d6d8ab801146cb234c6fc0fc32415d5d03`
- `experiments/configs/lvis_unseen_serverless_refresh_2026.yaml`: `16646fd7fb47bbb65061262dc92d01498870ae688d8d1fb7bde3d40077404e2d`
- `data/processed/lvis_v1/val/lvis_v1_val_unseen.json`: `11bd1b64efc51b7476ab110c06d72623a91ee421ff9b6e7c970353bc3065765c`
- `data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json`: `38887e637d5554f1c178dbbb5c6da2688cd147f9a0dcd3a9d8b415cb726647df`

## Missing provenance

- Full per-request prompts and image payloads for the principal runs, especially the string-response Gemini pilots.
- A contemporaneous executed-code commit and non-secret snapshot of effective parameters.
- Exact Gemini inference dates and precise router run boundaries beyond the metadata described above.
- Independent pre-evaluation coordinate calibration. The recovered request asks for pixel coordinates; the normalized-coordinate replay assumptions must remain separately identified.
- A persistent reviewer-accessible deposit. A separate anonymous check of the claimed GitHub URL returned HTTP 404; a local manifest does not establish public availability.
