#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /abs/path/to/file.tex" >&2
    exit 2
fi

doc="$1"
doc_dir="$(cd "$(dirname "$doc")" && pwd)"
doc_name="$(basename "$doc")"
job_name="${doc_name%.tex}"
out_dir="$doc_dir/latex-tmp"
texbin="/home/sam/.local/texlive/2026/bin/x86_64-linux"
pdflatex="$texbin/pdflatex"

mkdir -p "$out_dir"

"$pdflatex" \
    -synctex=1 \
    -interaction=nonstopmode \
    -file-line-error \
    -output-directory="$out_dir" \
    "$doc"

"$pdflatex" \
    -synctex=1 \
    -interaction=nonstopmode \
    -file-line-error \
    -output-directory="$out_dir" \
    "$doc"
