# Paper source

LaTeX source of *Weight-Dictionary Decomposition: Reading a Transformer's Residual Stream with Its Own Writes*.

- `wdd_tmlr.tex`: the master file, written in the TMLR template with anonymous authors. The other two are derived from it.
- `wdd_preprint.tex`: the named preprint. Its PDF is at [doi:10.5281/zenodo.22749478](https://doi.org/10.5281/zenodo.22749478).
- `wdd_iclr.tex`: the anonymous ICLR 2027 submission, number 13712, withdrawn on 2026-11-21. The runs that answer the reviewers' questions are in [`../post_review/`](../post_review/).
- `appendix.tex`: the appendix, shared by all three.

Build any of them with [tectonic](https://tectonic-typesetting.github.io), for example `tectonic wdd_preprint.tex`. Each comes to 30 pages.

The appendix tables and figures, Table 1 and the left panel of Figure 1 are generated from the result files in [`../results/`](../results/), not typed by hand:

- `tools/ci_tables.py` writes `ci_tables.tex` and `table1_rows.tex`, and `tools/z_tables.py` writes `z_tables.tex`.
- `tools/splice_table1.py` rewrites Table 1 inside `wdd_tmlr.tex`, and `tools/fig1.py` rewrites the left panel of Figure 1 there.
- `tools/appendix.py` writes `appendix.tex`.
- `tools/derive.py` derives `wdd_preprint.tex` and `wdd_iclr.tex` from `wdd_tmlr.tex`, using `tools/ai_statement.tex` for the ICLR version.
- `tools/make.sh` runs all of these except `fig1.py` and then builds the three PDFs. It needs Python 3 with numpy, tectonic and pdftotext.

Run against this repository's `results/`, the scripts leave every `.tex` file here unchanged (checked on 2026-11-21).

`iclr2027_conference.sty` and `tmlr.sty` are the venues' template files, and `fancyhdr.sty` and `natbib.sty` are the packages they load. They are included only so that the sources compile.
