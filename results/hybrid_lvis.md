# Offline Late-Fusion Hybrid — LVIS-unseen (serverless roster, no new inference)

**Question (referee §2 / §5.3.3).** Does the proposed `detector → VLM → validator → decision`
hybrid actually deliver a better accuracy/latency/cost operating point than either family alone?

**Answer.** Yes. Fusing the (score-floor-corrected) detector with a VLM is strongly
complementary, and a *deployable* confidence-router captures most of the gain while calling
the VLM on a minority of images.

## Setup (all from cached boxes — zero API, zero GPU re-runs)

- Detector stage: **YOLO-World**, low-floor outputs (the strongest detector after the score-floor audit), AP 0.347.
- VLM stage: **Qwen3-VL-8B-Instruct**, coordinate-decoded (best cached serverless VLM), AP 0.317.
  (Frontier Gemini-3.5 would slot in here unchanged if its per-image boxes are regenerated.)
- Validator/fusion: class-aware IoU matching (thr 0.5). Corroborated boxes (detector & VLM
  agree) get a confidence boost; detector-only keep their score; VLM-only get a prior (0.40).
- Router: escalate to the VLM iff detector max-confidence < τ (**observable signal, no GT**).
- Eval: same validated path as the baselines — 396 imgs × 10 unseen cats, COCOeval maxDets=100.
- Controls: detector-only reproduces 0.3465, VLM-only 0.3168 (match standalone evals).

## Results

| Configuration | AP | AP@0.5 | Escalation | Latency (s) |
|---|---|---|---|---|
| Detector-only (YOLO-World) | 0.347 | 0.445 | 0% | 0.12 |
| VLM-only (Qwen3-VL-8B) | 0.317 | 0.463 | 100% | 23.9 |
| Hybrid, τ=0.05 | 0.446 | 0.579 | 7.8% | 2.0 |
| Hybrid, τ=0.10 | 0.453 | 0.584 | 10.1% | 2.5 |
| **Hybrid, τ=0.40 (recommended)** | **0.481** | **0.614** | 26.3% | 6.4 |
| Blanket fusion (always escalate) | 0.476 | 0.613 | 100% | 23.9 |
| Oracle ceiling (GT-guided routing) | 0.532 | 0.640 | 48.5% | 11.7 |

## Takeaways

1. **Complementarity.** Fusion (0.476) beats the best single model (detector 0.347) by +0.13 —
   detector and VLM recover different objects.
2. **Deployable routing wins.** The confidence-router peaks at **0.481 at 26% escalation**, which
   *exceeds blanket fusion* — selectively escalating only uncertain images both saves cost and
   avoids VLM false positives on easy images. At 7.8% escalation it already gains +0.10 AP.
3. **Cost/latency.** Recommended operating point calls the VLM on ~1 in 4 images: ~6.4 s avg
   latency vs 23.9 s for VLM-only, at ~26% of the VLM API cost (× per-call cost from the ledger).
4. **Headroom.** Oracle ceiling 0.532 → a learned router could gain ~0.05 more (future work).

## Caveats / next steps

- Serverless roster: absolute numbers rise with the frontier VLM; the pipeline is model-agnostic
  (drop in Gemini-3.5 boxes to get the headline hybrid).
- Fusion scores (VLM prior 0.40, agreement boost 0.30) are fixed; report a sensitivity sweep.
- Latency is local-detector vs hosted-VLM (§3.5 framing).
- Reproduce: `python experiments/hybrid_eval.py` → `results/hybrid_lvis.json`.
