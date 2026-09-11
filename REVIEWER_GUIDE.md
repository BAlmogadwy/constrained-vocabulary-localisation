# Reviewing the saved evidence

Run commands below from the repository root. Python 3.10 or later is sufficient for preparation and integrity checks. Neither command imports model/evaluator code, accesses credentials, downloads data or calculates metrics.

```bash
python analysis_records/verify_review.py
python analysis_records/prepare_review_inputs.py
```

The first command checks the tracked evidence and archive checksums, then prints values from the saved result JSON. The second checks and unpacks 396 Grounding DINO and 396 OWL-ViT records from `results/reviewer_detector_outputs.zip`. These are the original detector outputs, not regenerated predictions. Both commands can be run again. Preparation refuses to overwrite a file whose contents differ.

## Inputs for the COCO router replay

Download the **2017 Train/Val annotations** from the [official COCO download page](https://cocodataset.org/#download), and extract `annotations/instances_val2017.json`. No image downloads are needed for this replay. Supply that existing file to the helper:

```bash
python analysis_records/prepare_review_inputs.py --coco-annotations /path/to/instances_val2017.json
python analysis_records/verify_review.py --all-inputs
```

On Windows, quote a path with spaces, for example `--coco-annotations "C:/Data/COCO/instances_val2017.json"`. The helper checks its checksum against the retained evaluation input before copying it to `data/raw/coco2017/annotations/instances_val2017.json`. Dataset annotations retain the dataset's terms. Images and raw downloads are not committed here.

`--all-inputs` checks all 3,790 benchmark input files listed in the replay record, including the unpacked detectors and external COCO annotations. Without this flag, the checker verifies the compressed detector archive and skips the external COCO annotation file. It checks byte hashes, allowing only LF/CRLF line-ending differences through separately recorded hashes. It is an integrity check, not an independent reproduction of AP.

## Manuscript results and their source files

| Manuscript item | Retained source |
|---|---|
| Table 1 detector configurations | `experiments/configs/`, detector wrappers in `models/traditional/`, and `analysis_records/inference_manifest.md` |
| Table 2 coordinate/score sensitivity | `analysis_records/replay_results.json`: `vlm.<model>.conditions.raw_correct_decoder` and `raw_pixel_decoder`, each with `supplied` and `uniform` |
| Table 3 and Figure 1(a), official LVIS AP | `analysis_records/lvis_semantics_sensitivity.json`: `models.<model>.official_lvis.metrics`; corresponding COCO values in `reproduced_frozen_coco_style` |
| Figure 1(b), paired LVIS intervals | `analysis_records/lvis_paired_bootstrap.json`: `models`, especially `paired_delta_percentile_95_ci` |
| Detection-cap control | `analysis_records/lvis_semantics_sensitivity.json`: `compatible_cap_lvis` |
| Table 4, 868-image router | `analysis_records/replay_results.json`: `router.coco_exclude_lvis_overlap` |
| Full 901-image comparison | The same file: `router.coco_original` |
| 868-image paired interval | The same file: `paired_coco_disjoint_router_bootstrap` |
| Table B.1 deployment values | Retained original aggregates in `results/metrics_summary.json` and `results/traditional_score_floor_audit.md` |
| Table B.1 low-floor values | `analysis_records/replay_results.json`: `detectors` |
| Appendix C development tables | Retained original summaries in `results/hybrid_summary.md` and `results/hybrid_lvis.json`; these were not recomputed in the current analysis |

## Optional offline numerical reruns

The saved JSON already contains the manuscript results. The following commands are optional and **do** recompute AP and/or bootstrap intervals, but use retained predictions only. They do not run inference or call an LLM API. These reruns were **not** performed as part of the repository packaging repair.

Use a separate Python environment; Python 3.11 was used for the recorded analysis. On Windows:

```powershell
python -m venv .venv-review
.venv-review/Scripts/python.exe -m pip install -r requirements-review.txt
.venv-review/Scripts/python.exe analysis_records/replay_audit.py --bootstrap 1000
.venv-review/Scripts/python.exe analysis_records/lvis_semantics_sensitivity.py
.venv-review/Scripts/python.exe analysis_records/lvis_paired_bootstrap.py --bootstrap 1000 --seed 12345
```

On Linux/macOS, use `.venv-review/bin/python` in place of `.venv-review/Scripts/python.exe`. Prepare all inputs first. Requirements pin the recorded NumPy, pycocotools and LVIS evaluator versions. Installing packages requires internet access; running the prepared analysis does not. The LVIS compatibility adjustment restores the removed `np.float` alias in memory without changing evaluator logic.

Reruns write to `analysis_records/recomputed/`, leaving the frozen JSON untouched. The LVIS sensitivity and bootstrap scripts read the retained companion JSON in `analysis_records/` for their original consistency checks; they do not silently substitute the files from a previous rerun. Input/output locations can be overridden with `CVL_REPO_ROOT` and `CVL_RESULTS_DIR`. The output directory cannot be the frozen `analysis_records/` directory.

## Provenance and limits

The original scripts are in `analysis_records/historical_scripts/`. Their paths reflect the original workstation and are retained for provenance, not as the portable entry points. Original result JSON, script hashes, version records and absolute input paths have not been rewritten. `review_manifest.json` supplies a portable path/checksum map. The changes to the three numerical entry points concern paths, dependency discovery and output locations, not scoring or bootstrap calculations.

Several older preparation reports in `analysis_records/` describe intermediate manuscript versions and may contain obsolete page counts or submission-status notes. They are historical records. Use this guide and the current manuscript for the submission. Proposed controls in `protocol_audit.py` and material listed in `LEGACY_NOTES.md` are not additional evidence for the paper's claims.

Repository visibility is private. Authorised access or a reviewer-accessible archive still needs to be arranged; this guide does not claim that the URL grants access or promise a post-publication release.
