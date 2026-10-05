# Second-round revision

- [Current manuscript PDF](manuscript_r2/manuscript_r2.pdf) and [LaTeX source](manuscript_r2/manuscript_r2.tex).
- [Response letter PDF](response_letter_r2/response_letter_r2.pdf) and [LaTeX source](response_letter_r2/response_letter_r2.tex).
- `manuscript_r2/figs/` contains exactly the 17 PNGs referenced by the manuscript.
- Bibliography, Elsevier CAS styles and response templates are included; intermediate build files are excluded.

Compile from each document's folder with `latexmk -xelatex -interaction=nonstopmode -halt-on-error manuscript_r2.tex` or `latexmk -xelatex -interaction=nonstopmode -halt-on-error response_letter_r2.tex`.

Experiment outputs are in `experiments/window_sensitivity/`. Plotting scripts and extracted figure data are in `scripts/plotting/` and `results/reference/figure_data/`.
