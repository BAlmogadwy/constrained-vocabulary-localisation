# Earlier experiments and superseded claims

These notes preserve the provenance of older material. Use the current README and REVIEWER_GUIDE.md for the current paper.

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


## Exploratory material

The repository also retains material from earlier exploratory rounds that the manuscript does
**not** use and does not report:

- `VISUAL_COT_README.md` and `results/visual_cot/` — a visual chain-of-thought prompting trial.
- `HOW-TO-RUN_LLAVA-BOXREG.md` — notes for a LLaVA box-regression experiment.
- `YOLO Stuff/` and `results/ablation/` — earlier development scratch and older-generation runs.

These are kept for transparency about what was explored, not as evidence for any claim in the
paper. Nothing in the manuscript depends on them.
