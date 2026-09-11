# Visual Chain-of-Thought Experiments

This repository includes a `run_visual_cot.py` driver that extends the standard VLM ablation stack with a five-step Visual Chain-of-Thought (Visual CoT) loop. Each image is processed baseline-first and then refined across four additional iterations, drawing the previous detections directly onto the image to encourage self-correction.

## Highlights
- **Dataset parity**: Reuses the exact 100-image LVIS subset cached in `data/subsets/lvis/ablation_100_ids.json` (seed 1337). Missing files are automatically backfilled from `data/raw/coco2017/val2017` if needed.
- **Refinement loop**: Step 0 uses the original prompt + clean image. Steps 1–4 overlay the prior detections (red 3px boxes) and augment the prompt with a refinement instruction.
- **Full dialogue capture**: Per-image outputs include the entire `refinement_chain` (latency, prompt, raw responses, annotated image paths) so analysts can replay the loop.

## Requirements
1. Populate the usual API keys for the models you plan to run (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.).
2. Ensure `data/subsets/lvis_v1_val_unseen/images/` exists. If specific subset entries are missing, the script clones them automatically from `data/raw/coco2017/val2017`.
3. Activate the project virtual environment and install dependencies from `requirements.txt`.

## Running a Model
A separate config is provided per model under `experiments/configs/visual_cot_lvis_<model>.yaml`. Run one at a time, for example:

```bash
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_gpt5.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_claude.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_gemini.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_gemma.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_llama.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_llava.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_paligemma.yaml
python experiments/run_visual_cot.py --config experiments/configs/visual_cot_lvis_qwen.yaml
```

Useful flags:
- `--test` – run only the first two images (smoke test).
- `--force` – recompute results even if per-image caches already exist.
- `--resample-subset` – rebuild the subset cache (only if you intentionally want a different set of 100 images).

Each config writes to `results/visual_cot/lvis_<model_name>/` with per-model aggregate JSON and per-image records under `per_image` plus annotated PNGs under `debug_images/`.

## Output Anatomy
Per-image JSON files contain:
- `final_detections` and `final_latency_sec` – summary of the last refinement.
- `refinement_chain` – list of five entries describing each step (inputs, prompt, latency, raw response, normalized boxes, annotated image path).
- `metadata.image_size` – width/height used to normalize bounding boxes.

Aggregate JSON files include wall-clock totals, per-temperature/prompt summaries, and pointers (`chain_file`) back to the detailed per-image records.

## Troubleshooting
- **Missing image errors**: The script now copies absent subset files from fallback directories. Ensure `data/raw/coco2017/val2017/` is populated; otherwise download COCO 2017 val images before running Visual CoT.
- **API rate limits**: The refinement loop multiplies calls per image, so consider limiting concurrency or running the `--test` mode first to verify credentials.
- **Disk usage**: Annotated PNGs live in `results/visual_cot/.../debug_images/<image_id>/step_<n>.png`. Clean these directories if you need to reclaim space after analysis.

Feel free to adapt the configs (different models, prompts, or datasets) as long as the subset references remain valid.
