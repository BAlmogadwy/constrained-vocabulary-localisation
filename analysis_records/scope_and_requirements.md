# Neurocomputing: scope and submission-readiness review

Verified 9 September 2026. This is a bounded requirements and positioning review, not an acceptance prediction or confirmation that the complete submission passes every check. The official guide was inspected in the browser; its submission portal was not inspected. No submission or author declaration was made.

## Scope and a defensible contribution

Neurocomputing's regular-article scope requires a direct connection to neural networks or learning systems. Its exclusions include work that merely combines existing algorithms without that connection. A diagnostic study of vision-language neural models can establish the connection through a clear research question about their localization behavior and the validity of its measurement. Journal fit still depends on the strength of the evidence and the contribution beyond existing work. [Official guide, Aims and scope and Article types](https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors).

Suggested title: **Auditing Vision-Language Models for Constrained-Vocabulary Object Localization**.

Suggested contribution statement, conditional on the corresponding analyses actually being completed: “We audit category-conditioned bounding-box prediction by vision-language models under explicitly supplied candidate vocabularies. We document how coordinate interpretation, output parsing, detection filtering, confidence ordering, and evaluation choices affect localization comparisons, and provide the artifacts needed to reproduce the reported results.” Narrow this statement if any listed diagnostic is absent. Candidate lists containing annotated present categories plus distractors constitute additional information; the results should be presented under that condition, without implying unrestricted category discovery.

Ramachandran et al. already compare several model families with DETR, Co-DETR, and 4M-21; their study includes both recursive grid-based localization and direct coordinate regression. GroundingME assesses localization from referring expressions and includes rejection of ungroundable descriptions. Liu et al. provide a spatial-intelligence survey with an empirical comparison of 37 models on nine question-answering benchmarks. Therefore, multi-family evaluation, comparison with detectors, or a general semantic-versus-geometric gap cannot alone establish this manuscript's novelty. A reproducible, controlled audit offers a more defensible contribution. [Ramachandran et al.](https://arxiv.org/html/2507.01955v1), [GroundingME](https://openaccess.thecvf.com/content/CVPR2026/html/Li_GroundingME_Exposing_the_Visual_Grounding_Gap_in_MLLMs_through_Multi-Dimensional_CVPR_2026_paper.html), [Liu et al.](https://link.springer.com/article/10.1007/s10462-026-11671-x).

## Verified requirements

The following distinctions come from the [current official Guide for Authors](https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors). “Required” includes explicit instructions for material that applies to this submission; it does not mean every item has its own mandatory portal upload field.

| Item | Status | Implementation for this package |
|---|---|---|
| Review identity | Single-anonymized review | Include author identities. The guide says editors conduct the initial suitability assessment. |
| Source files | Required | Supply editable manuscript and supporting source files; PDF alone is insufficient. Word must use one column; LaTeX may use two columns. |
| LaTeX template | Encouraged | Use the publisher template if convenient; it is not stated as compulsory. |
| Title page | Required | Title; correctly ordered author names; full institutional affiliations and postal addresses including country; identified corresponding author and email. |
| Abstract | Required | Self-contained and factual; no more than 250 words. |
| Keywords | Required | 1–7 English keywords. |
| Competing interests | Required process | Complete Elsevier's declarations tool even for no interests; upload its resulting Word document. Signatures are not required. A draft statement does not complete this process. |
| Funding | Required when support exists | Identify support and sponsors' roles, including lack of involvement where applicable. A no-specific-grant statement is recommended if true; do not infer it from silence. |
| Author contributions | Required | Assign actual CRediT roles to authors; author confirmation is needed for factual accuracy. |
| AI assistance in manuscript preparation | Required when applicable | Disclose tool/service and purpose, with human review and responsibility, in a section immediately before references. This package's substantive AI-assisted drafting falls within the disclosure requirement. Basic spelling, grammar, and reference checks are exempt. Research-model usage also belongs in Methods; it does not replace the writing disclosure. |
| Research data | Option C applies | Deposit the research data in a relevant repository, then cite and link it; if sharing is impossible, state the reason. The guide's research-data discussion includes code, software, models, and other materials. |
| Data availability statement | Required at submission | State the actual availability and any reason for non-sharing. Avoid future-tense promises presented as current availability. |
| Author biographies and photographs | Explicit guide request | Provide an editable biography of at most 100 words for each author and a passport-type photograph as a separate figure. Do not fabricate personal details or create synthetic author photos. |
| Highlights | Encouraged | If included, submit a separate editable file with “highlights” in its name; 3–5 bullets, at most 85 characters each including spaces. |
| References | Consistency required at initial submission | Initial format is flexible. The journal's stated style is numeric square brackets, ordered by first citation; use that style in the prepared package. Cite the published ICLR 2026 and CVPR 2026 versions where available. |
| Tables, equations, figures | Required format when present | Keep tables and equations editable; supply suitable figure files and captions. Ensure text citations match the submitted assets. |
| Submission declarations | Required | Final authorship approval, exclusive submission, and other factual declarations must be accurate. A prior rejection does not itself establish these facts. |

The requested AI-disclosure section title is: **Declaration of generative AI and AI-assisted technologies in the manuscript preparation process**.

No regular-article total page or word cap, mandatory cover-letter instruction, or graphical-abstract requirement was found in the inspected guide. This is an absence of a verified requirement, not proof that the submission system has no additional fields. A tailored cover letter is useful; do not describe it as guide-mandated. The guide alone supports numeric reference styling; no separate claim about the current `elsarticle` template options was verified in this check.

## Repository verification and data readiness

The intended URL is `https://github.com/BAlmogadwy/zero-shot-detection-benchmark`. An unauthenticated HTTP request to `https://api.github.com/repos/BAlmogadwy/zero-shot-detection-benchmark` returned **404 Not Found** on 9 September 2026. Web retrieval also failed. This does not distinguish a private repository from an absent or renamed repository, but it does not support a claim of public availability. No credentials were used for the API check.

Before using a public-availability statement, verify anonymous access to the correct repository and inspect the exact release referenced by the manuscript. A repository link alone does not establish reproducibility. The release should identify the code revision, environment, model/checkpoint and API versions, exact prompts, category lists, dataset image identifiers and splits, retained raw outputs or permissible prediction artifacts, coordinate-decoding rules, evaluation commands, and scripts producing the reported tables. This artifact list is a reviewer-readiness recommendation rather than a verbatim journal checklist. Follow the original datasets' distribution terms; link to their official releases when redistributing images is inappropriate. A persistent archived release would make the data citation more stable.

## Scientific checks before submission

These are reviewer-facing evidence requirements, not findings that every item is currently unresolved in the latest draft:

1. **Protocol identity:** state exactly how candidate labels are generated and what annotation information reaches each model. Define the evaluated task consistently in the title, abstract, methods, tables, and conclusion.
2. **Coordinates and parsing:** validate model-specific coordinate conventions against image dimensions and raw outputs. Report invalid, unparsable, empty, and out-of-range responses, together with any repair or clipping rules. Recompute affected results after changes.
3. **Detector comparison:** document detector prompts, preprocessing, confidence thresholds, maximum detections, and postprocessing. Establish that filtering does not create an arbitrary performance disadvantage.
4. **AP interpretation:** explain the origin and comparability of confidence scores, ties and ordering, IoU thresholds, matching rules, class mapping, and the evaluator. Equal or artificial scores can alter the interpretation of ranked detection metrics.
5. **Split independence:** audit image overlap wherever a split is used for prompt selection, routing, thresholds, or model selection. Distinguish genuine held-out evaluation from diagnostic reuse. Explain any unavoidable overlap rather than presenting it as independence.
6. **Routing claims:** include only results supported by a specified router and selection procedure. Separate diagnostic or oracle upper bounds from implementable performance. A simple router needs evidence of useful behavior beyond its component models.
7. **Scope of conclusions:** identify the selected categories, images, datasets, and model versions. Generalize only as far as that sample permits; report uncertainty when making comparative claims.
8. **Claim-to-artifact consistency:** confirm that every central number is reproducible from the released configuration and that related-work descriptions accurately acknowledge prior benchmarks. Remove claims that remain unsupported after correction.

The manuscript is ready to submit only after the scientific claims, final artifacts, and author-supplied declarations agree. This review does not assign an acceptance probability.
