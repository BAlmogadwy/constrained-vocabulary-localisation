# Offline protocol verification

These files audit retained benchmark artifacts and provide isolated controls for future experiments. They do not alter the original benchmark, load credentials, instantiate models, or call hosted APIs. The numerical replay in `replay_audit.py` uses the historical parser. The stricter parser in `protocol_audit.py` is a proposed correction and is not the parser used for the manuscript's replayed AP.

For the main paper, start with [REVIEWER_GUIDE.md](../REVIEWER_GUIDE.md). These optional historical protocol checks are separate from the numerical replay. Run from the repository root:

```powershell
python -m unittest discover -s analysis_records -p test_protocol_audit.py
python analysis_records/protocol_audit.py --repo .
```

The audit and its tests require only Python's standard library. The `--repo .` option selects the current checkout. Omitting `--output` prints JSON instead of writing a report. The report reconstructs the historical crop selector without importing its provider-configuring module.

## Verified results

| Check | Finding |
|---|---|
| Recognition subset | 40 selected images, 56 reconstructed annotated crops with source images available. |
| Always-first baseline | 55/56 correct, or 98.2143%; mean uniform-random expectation 19.8810%. |
| Candidate order | All 396 LVIS lists place annotated positives before distractors. The actual 901 evaluated COCO lists do likewise. |
| Historical GPT-5 iterative runs | At each of temperatures 0.0, 0.2 and 0.5, all 100 records have incomplete identification responses because of the token limit, no identification output text, no localization calls, and zero detections. Absence of a top-level API error would incorrectly count all of them as “parsed OK.” |
| Missing later evidence | The 2026 oracle per-crop directory and reduced 2026 prompt-ablation directory referenced by the analyzer are absent. The oracle table retained in `results/tables.zip` is an aggregate, not per-crop evidence. |
| Evaluated overlap | 396 LVIS router images and 901 COCO router images share 33 identifiers. Removing those from COCO leaves 868. The full COCO candidate mapping has 5,000 entries and must not be mistaken for the evaluated subset. |
| Raw normalization archive | `results/ablation_normalization.zip` contains retained per-image raw responses; the blanket claim that these responses are unavailable is incorrect. |

The historical GPT-5 failures occur before the incompatible localization parser is reached. They must not be described as empirical proof that the bare-array parser defect caused the zero detections. The defect is independently established by the source contract and a small deterministic parser probe.

## Eight regression tests

1. Shuffling produces the same permutation for identical candidate membership regardless of which label was originally first. It receives no ground-truth label.
2. A bare coordinate array receives a class only from explicit retained prompt metadata.
3. Missing class or coordinate-unit metadata prevents bare-array adaptation; the utility does not guess.
4. A returned class conflicting with the retained single-class prompt is rejected.
5. Valid empty JSON `[]` is counted separately from missing text, JSON `null`, malformed JSON, and invalid detections.
6. Invalid geometry, nonfinite coordinates, and partially invalid response arrays fail validation instead of being silently accepted.
7. Native and uniform scoring preserve zero-score boxes, make missing-score fallback explicit, and leave the input objects unchanged.
8. An incomplete reasoning-only response with null content is recognized as having no answer text.

The stricter utility classifies **whole responses** as valid nonempty, valid empty, or failure with a reason. Any invalid detection item fails that response. It does not provide a per-entry partial-success counter. The bare-array adapter validates the declared coordinate representation but does not convert units; the caller must perform that transformation before metric evaluation.

## Which results can be repaired offline?

Existing detection outputs support explicit native/uniform rescoring and an evaluation excluding overlapping image IDs. Regenerate AP, routing rates, latency summaries, and confidence intervals on the same retained subset. Uniform scoring in the historical replay assigns 1.0 only **after historical parser acceptance**. The historical parser can discard nonpositive-confidence entries; the isolated utility's different behavior must not be retroactively attributed to those results.

Candidate-order shuffling changes the model's input and therefore needs fresh inference. Membership in the original candidate lists remains annotation-derived even after shuffling. A future design should separately predeclare candidate membership and order, retain a no-image baseline, and record seeds before inspecting outcomes.

A corrected iterative prompt needs explicit class and coordinate conventions, sufficient output-token budget, and a rule for multiple instances. Existing bare responses may be adapted only where retained request metadata establishes the class and units. The current 300 historical GPT-5 records have no localization answer to recover. Missing 2026 raw responses cannot be reconstructed from aggregate tables.

For fresh runs, cache keys must include the complete prompt, candidate order, model/configuration, image identity and preprocessing. Reusing a historical per-image or per-crop cache after changing the prompt would bypass the intended control.
