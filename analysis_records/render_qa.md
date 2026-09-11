# Artifact verification

Date: 9 September 2026.

## Manuscript

- Built with the installed Elsevier `elsarticle` class, pdfLaTeX and `elsarticle-num` BibTeX style. The missing `natbib` dependency was installed through MiKTeX.
- Final manuscript: **17 pages**. The abstract contains **179 whitespace-delimited words**; six keywords remain together on the first page.
- Every final page was rendered with Poppler and visually inspected: pages 1–6, 7–12 and 13–17 each passed. Tables, both plots, equations, references and declarations are legible; no clipping or overlap was observed.
- Corrected the split keywords, duplicate appendix label, overfull prompt lines, ambiguous decoder-table caption and reference spacing found during review.
- The final compile has no undefined citations/references or overfull/underfull boxes. MiKTeX/hyperref emits nonfatal PDF metadata-string warnings from frontmatter macros. These do not affect the inspected page content.
- The long prompt quotation continues across two pages with complete text. This is a cosmetic page break, not a missing-content issue.

## Editable Word files

- All three DOCX files render to one page each and were visually inspected in their final content state. Headings, paragraphs and bullets fit without clipping; the default title border was removed.
- The prescribed `render_docx.py` was attempted but could not find LibreOffice. The selected Windows dependency runtime does not bundle LibreOffice. No desktop LibreOffice fallback was used.
- Instead, `render_word.ps1` created an invisible, isolated Microsoft Word automation instance, opened only these newly created files read-only, exported PDF previews, closed them without saving, and quit that instance. Poppler rendered the previews for inspection.
- A subsequent manuscript-only rebuild was checked to leave the Word document XML unchanged from the inspected versions.
- Highlights: four bullets; lengths including spaces **74, 72, 74 and 74**, all below 85. Proposed biography: **40 words**, below 100.

## Scientific verification

- Eight meaningful protocol regression tests passed. They validate the proposed strict utility, not a retroactive change to the historical parser used for reported replay values.
- COCO-style and LVIS-aware results are separately recorded, with exact input hashes and software versions.
- Each paired analysis uses 1,000 image resamples with seed 12345. Cached accumulation matches full evaluator results; repeated-image validation confirms that tied predictions are preserved.
- LVIS-aware Gemini–YOLO intervals include zero; the draft does not use the COCO-style intervals to claim LVIS-aware superiority.
- No inference/API calls, external manuscript upload or author declaration submission occurred during this preparation.

These checks verify the local draft's numbers and presentation. They do not resolve the scientific controls, public deposition and author confirmations listed in `REVISION_PLAN.md`.
