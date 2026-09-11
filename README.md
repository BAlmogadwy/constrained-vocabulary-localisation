# Constrained-Vocabulary Object Localisation: evaluation-convention study

Code, configurations and retained outputs for **"Evaluating Vision-Language Models for Constrained-Vocabulary Object Localisation"** (under review).

The study re-evaluates **saved** model outputs to measure how three evaluation choices change a comparison between generative vision-language models (VLMs) and specialised open-vocabulary detectors:

1. **Coordinate decoding** — how text coordinates are converted into pixel boxes.
2. **Confidence assignment** — supplied scores versus a uniform score after parser acceptance.
3. **Annotation handling** — COCO-style scoring versus the LVIS federated rules.

No model is trained here, and no new inference was run for the analysis. Every number in the manuscript comes from re-scoring retained responses.

---

## Scope and limitations

Read this before using any number in this repository.

- The LVIS selection is a **custom** set of 396 images and ten categories. It is **not** an official LVIS unseen split, and category exposure during model pretraining is unknown.
- Both evaluation sets are **positives-only**: each image contains at least one annotation in the selected categories (396 of the LVIS unseen images; 901 of the 5,000 COCO 2017 validation images). AP on these sets is **not** comparable to full-vocabulary LVIS or COCO-val leaderboard AP.
- Per-image candidate lists are **partly derived from annotations** and place annotated categories first, followed by sampled categories without a retained annotation. The lists are not independent of ground truth, and categories without annotations are not necessarily absent.
- Coordinate conventions for the OpenRouter-served models are **assumptions carried over from the original runs**, not independently verified provider behaviour.
- The main response archives do not retain the actual requests, token limits or image inputs, so historical runs cannot be reconstructed exactly.
- Confidence intervals are exploratory and are not corrected for model selection or multiple comparisons.

## Superseded claims

An earlier version of this work was prepared for a different journal and was **not published**. Several of its claims did not survive later auditing and are **withdrawn**. Do not cite or reuse them:

| Withdrawn claim | Why |
|---|---|
| The 396-image set is an official LVIS "unseen" partition whose categories never appear in training | It is a custom selection; pretraining exposure is unknown |
| Frontier VLMs significantly exceed the best detector | Under LVIS annotation rules both paired Gemini–YOLO-World intervals include zero |
| Oracle crop recognition demonstrates a semantic-versus-spatial separation | Always selecting the first candidate scores 55/56, because candidate lists place positives first |
| Iterative prompting fails to repair grounding | The iterative prompt requested bare coordinates while the parser required labelled objects |
| COCO was held out from LVIS development | 33 image IDs overlap; the corrected comparison uses 868 images |
| All VLM boxes were scored 1.0 | The pipeline preserved supplied scores; both policies are now reported separately |

The earlier manuscript source is retained locally but is **not** part of this repository.

---

## Repository layout

```
models/
  traditional/     YOLO-World (Ultralytics), Grounding DINO, OWL-ViT wrappers
  vlm/             VLM detectors + shared parsing/coordinate normalisation (parsing.py)
experiments/
  run_traditional.py         detector runs from a YAML config
  run_vlm.py                 synchronous VLM runner (OpenRouter / DashScope)
  run_vlm_batch.py           batch-API runner
  compute_metrics.py         COCO-style AP / AP@0.5
  run_live_hybrid.py         detector->VLM router execution
  hybrid_eval.py             routing and fusion rule
  bootstrap_ci.py            image-level bootstrap
  configs/                   per-dataset, per-model YAML configs
data/                        split preparation and dynamic candidate-label generation
results/                     retained per-image outputs, metrics and summaries
```

Offline re-analysis scripts used for the manuscript (`replay_audit.py`, `lvis_paired_bootstrap.py`,
`lvis_semantics_sensitivity.py`, `protocol_audit.py`) and their saved outputs are held with the
manuscript records and are provided alongside this repository.

## Models evaluated

**Detectors** (run locally, GPU, single-image): `yolov8x-world.pt` via Ultralytics; `grounding-dino-base`
and `owlvit-base-patch32` via Hugging Face Transformers. Deployment score floors 0.25 / 0.25 / 0.20;
the reported comparison uses a 0.001 floor.

**Hosted VLMs.** Gemini via the batch interface (`gemini-3.5-flash`, `gemini-3.1-pro-preview`);
`qwen/qwen3-vl-235b-a22b-instruct`, `google/gemma-3-27b-it` and `mistralai/mistral-large-2512`
via OpenRouter. OpenRouter routed these requests through several upstream providers, so the
results characterise the recorded routes rather than each vendor's own service.

`results/raw/vlm/` also holds retained per-image records for additional models from earlier
development rounds (Gemini 2.5 Flash, GPT-5, Qwen3-VL-8B, Llama-3.2-11B-Vision, LLaVA-1.6,
Gemma 3 12B). Only the five models with complete raw LVIS responses in the retained
normalisation-ablation archive are used in the manuscript's sensitivity analysis; Qwen3-VL-8B is
used for the router development sweeps.

## Datasets

LVIS and COCO images are distributed by their respective projects under their own terms and are
**not** redistributed here as a matter of policy. Use `data/download_datasets.py` and
`data/prepare_zero_shot_splits.py` to obtain and rebuild the splits. Image identifiers and
per-image candidate mappings are under `data/processed/` and `data/subsets/`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .example.env .env   # add API keys; never commit .env
```

Python 3.10+. Detectors need a CUDA GPU; hosted VLMs need provider credentials.

## Superseded exploratory material

The repository also retains material from earlier exploratory rounds that the manuscript does
**not** use and does not report:

- `VISUAL_COT_README.md` and `results/visual_cot/` — a visual chain-of-thought prompting trial.
- `HOW-TO-RUN_LLAVA-BOXREG.md` — notes for a LLaVA box-regression experiment.
- `YOLO Stuff/` and `results/ablation/` — earlier development scratch and older-generation runs.

These are kept for transparency about what was explored, not as evidence for any claim in the
paper. Nothing in the manuscript depends on them.

## Licence

Code in this repository is released under the MIT Licence (see `LICENSE`). Model outputs, dataset
images and annotations remain subject to the terms of their respective providers and projects.
