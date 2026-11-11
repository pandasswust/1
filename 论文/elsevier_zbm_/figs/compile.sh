#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Install a TeX distribution first."
  exit 1
fi
if ! command -v bibtex >/dev/null 2>&1; then
  echo "bibtex not found. Install a TeX distribution first."
  exit 1
fi

rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.out manuscript.log
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
echo "Built manuscript.pdf"
