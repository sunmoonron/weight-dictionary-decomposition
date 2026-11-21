#!/usr/bin/env bash
# one command from result files to the three PDFs
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/ci_tables.py
python3 tools/z_tables.py
python3 tools/splice_table1.py
python3 tools/appendix.py
python3 tools/derive.py
for f in wdd_tmlr wdd_preprint wdd_iclr; do tectonic -X compile --keep-logs $f.tex 2>&1 | grep -iE "^error|! " | head -3 || true; printf "%-13s %2s pages, overfull>5pt: %s\n" $f "$(pdftotext $f.pdf - | python3 -c 'import sys; print(sys.stdin.read().count(chr(12)))')" "$(grep -oE 'Overfull \\hbox \([0-9.]+pt' $f.log | grep -oE '[0-9.]+' | awk '$1>5' | wc -l | tr -d ' ')"; done
