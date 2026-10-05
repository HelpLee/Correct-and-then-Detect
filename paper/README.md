The current second-round manuscript, figures and response letter are in [../2nd_rev/](../2nd_rev/). This folder preserves the earlier manuscript and author title page.

# Earlier manuscript snapshot

Main file: `Fault-detection_TL_reorganized.tex`. Author title page: `title_page.tex`.

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build Fault-detection_TL_reorganized.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build title_page.tex
```

The project includes every graphic referenced by the current manuscript, `cas-refs.bib`, and the original Elsevier CAS class/style files. Compilation outputs belong in `build/`. The final verified manuscript and title-page PDFs are supplied at the project root. Preserve the copyright notices in the vendor style files.

The latest detector results use τ=1.8 °C. The main manuscript retains its existing review anonymization. Scientific parameter discrepancies are documented in `../docs/PROVENANCE.md`.
