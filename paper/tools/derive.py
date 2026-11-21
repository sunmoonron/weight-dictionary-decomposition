"""Derive wdd_preprint.tex (named, arXiv) and wdd_iclr.tex (ICLR 2027 template, anonymous) from wdd_tmlr.tex, the single source."""
import pathlib, re
src = pathlib.Path('wdd_tmlr.tex').read_text()
AUTH_ANON = "\\author{\\name Anonymous authors \\\\\n\\addr Paper under double-blind review}"
assert src.count(AUTH_ANON) == 1
# ---- preprint ----
pp = src.replace('\\usepackage{tmlr}\n', '\\usepackage[preprint]{tmlr}\n', 1).replace(AUTH_ANON, "\\author{\\name Ronish Bhatt \\\\\n\\addr \\href{https://ronishbhatt.com}{ronishbhatt.com}}")
pathlib.Path('wdd_preprint.tex').write_text(pp)
# ---- ICLR ----
ic = src
ic = ic.replace(ic[:ic.index('\\usepackage[utf8]{inputenc}')], "% Weight-Dictionary Decomposition, ICLR 2027 submission (derived from wdd_tmlr.tex by tools/derive.py)\n\\documentclass{article}\n\\usepackage{iclr2027_conference,times}\n% \\iclrfinalcopy  % camera-ready only\n\n")
ic = ic.replace(AUTH_ANON, "\\author{Anonymous authors}")
ai = pathlib.Path('tools/ai_statement.tex').read_text()
bi = ic.index('\\subsubsection*{Broader Impact Statement}'); ap = ic.index('\\appendix')
statements = ic[bi:ap].replace('\\subsubsection*{Broader Impact Statement}', '\\subsection*{Ethics statement}').replace('\\subsubsection*{Reproducibility Statement}', '\\subsection*{Reproducibility statement}')
ic = ic[:bi] + ic[ap:]
bib = ic.index('\\begin{thebibliography}')
ic = ic[:bib] + ai + statements + ic[bib:]
pathlib.Path('wdd_iclr.tex').write_text(ic)
print('derived wdd_preprint.tex and wdd_iclr.tex')
