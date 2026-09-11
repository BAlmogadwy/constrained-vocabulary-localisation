# Constrained-Vocabulary Object Localisation

Code and retained evidence for **Evaluating Vision-Language Models for Constrained-Vocabulary Object Localisation**, prepared for submission to Neurocomputing.

The study examines how coordinate decoding, confidence assignment and annotation rules change comparisons between generative vision-language models (VLMs) and specialised detectors. It uses saved model outputs; no new inference was run for the manuscript's re-analysis. The main results were re-evaluated offline. Appendix B's deployment-floor column and Appendix C reproduce retained original aggregate results, and latency comes from recorded execution timings.

## Start here

To check the saved evidence and display the main recorded results, run this from the repository root:

```bash
python analysis_records/verify_review.py
```

This command needs only Python 3.10 or later. It does not need API keys, a GPU or any third-party Python packages, and it does not recompute AP or run a bootstrap. See **[REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)** for input preparation, the table-to-file map and optional offline rerun commands. Installing the full model environment is unnecessary for this check.

## Scope

- The LVIS evaluation is a custom selection of 396 images and ten categories, not an official unseen split or a full-vocabulary benchmark. Category exposure during pretraining is unknown.
- Each selected image has at least one retained annotation in the selected categories. Candidate lists are partly derived from annotations and place annotated categories first; additional categories are not necessarily absent.
- The main LVIS-aware comparison includes Gemini 3.5 Flash, Gemini 3.1 Pro, Qwen3-VL-235B, YOLO-World, Grounding DINO and OWL-ViT. A separate COCO-style decoding/score analysis also includes Gemma 3 27B and Mistral Large 3.
- Gemini decoding follows its documented convention. The Qwen, Gemma and Mistral conventions are assumptions carried over from the original runs, not independently verified provider behaviour.
- Some actual requests, token limits and image inputs are unavailable. Historical requests cannot all be reconstructed exactly. The image-bootstrap intervals are exploratory and do not cover variation across new hosted calls.
- The router ran on 901 COCO images. Excluding 33 images shared with LVIS development was post hoc, leaving 868 images. The AP gain is 0.0064 before exclusion and 0.0115 after exclusion. The reported paired interval applies only to the 868-image comparison. LVIS router sweeps are development results.

## Repository layout

| Location | Purpose |
|---|---|
| [analysis_records/](analysis_records/) | Offline analysis scripts, frozen result JSON, integrity manifest and supporting records |
| [results/ablation_normalization.zip](results/ablation_normalization.zip) | Retained raw-response archive used by the five-model sensitivity analysis |
| [results/reviewer_detector_outputs.zip](results/reviewer_detector_outputs.zip) | 792 retained low-floor Grounding DINO and OWL-ViT output files; unpack with the reviewer helper |
| `results/raw/` | Other retained per-image detector, VLM and router records |
| `data/processed/` | Retained annotation subsets and per-image candidate lists |
| `models/` and `experiments/` | Original model wrappers, configurations and execution code |
| [LEGACY_NOTES.md](LEGACY_NOTES.md) | Provenance of superseded claims and exploratory material not used by the paper |

The `.json` result records are frozen evidence. Optional reruns write to `analysis_records/recomputed/` by default. Historical script copies and original workstation paths in provenance records are retained for traceability; portable entry points are documented in the reviewer guide.

## Data and access

Source images are not redistributed. Offline evaluation of the retained boxes does not need images. The COCO router replay does need the original COCO 2017 validation annotation file; the reviewer guide explains where to obtain it and how to verify it. The required LVIS subset, candidate lists and saved predictions are included.

The repository provides the code and retained evidence. The full manuscript PDF and its LaTeX source are maintained separately and are not included here.

## Original model environment

The original inference code is retained for provenance and future use. It is separate from the offline reviewer workflow. `requirements.txt` contains the larger model environment; hosted inference requires the appropriate provider credentials and may incur charges. Do not run `run_vlm.py`, `run_vlm_batch.py`, `run_live_hybrid.py` or detector runners to inspect the saved results.

## Licence

Source code is covered by the [MIT Licence](LICENSE). Dataset annotations and retained model outputs remain subject to the terms of their respective datasets and providers; the code licence does not replace those terms.
