# LaTeX report

**Reproducing ACE and Evaluating Diagnostic-Free Offline Training on Formula**  
WANG Haowei (1155244499), FTEC5660, 10 November 2026.

This folder is self-contained: `main.tex`, `author.tex`, `references.bib`, `figures/`, and the verified compiled `main.pdf`. Upload the entire folder to a LaTeX editor, selecting `main.tex` as the root. Use pdfLaTeX with BibTeX, or Tectonic.

```bash
# Run inside this folder (first Tectonic build may download TeX packages)
tectonic main.tex
# Alternative: a standard TeX Live/MiKTeX installation
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`figures/learning_flow.tex` is editable TikZ source. The two PNGs are unchanged copies of the measured-study plots in `../repro/artifacts/`; `v4_results.png` is used in the report and `v3_results.png` is retained as an optional additional figure. No API calls are required to compile.

The report distinguishes the original official-test reproduction, archived-response replay, fresh paired training efficiency, and the validation-derived fixed-book accuracy audit. It reports negative findings and the failed accuracy interval criterion. Exact numerical evidence is maintained in the repository artifacts rather than inferred from figure pixels.
