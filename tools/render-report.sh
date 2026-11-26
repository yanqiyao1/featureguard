#!/bin/sh
# Optional publishing tool: Pandoc 3.8+, XeLaTeX (with xeCJK), and Fontconfig.
set -eu
cd "$(dirname "$0")/.."
for tool in pandoc xelatex fc-match; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 2; }
done

font=${CJK_FONT:-}
if [ -z "$font" ]; then
  for candidate in 'Noto Serif CJK SC' 'Source Han Serif SC' 'Songti SC' 'PingFang SC'; do
    if [ "$(fc-match -f '%{family[0]}' "$candidate")" = "$candidate" ]; then
      font=$candidate
      break
    fi
  done
fi
[ -n "$font" ] || { echo 'Install a Simplified Chinese font or set CJK_FONT.' >&2; exit 2; }

pandoc docs/report.md --from=markdown --pdf-engine=xelatex --syntax-highlighting=none \
  -V papersize=a4 -V fontsize=12pt -V linestretch=1.06 \
  -V geometry:'left=22mm,right=22mm,top=24mm,bottom=22mm' \
  -V "CJKmainfont=$font" -V "CJKsansfont=$font" -V "CJKmonofont=$font" \
  -V colorlinks -V urlcolor=accent -V linkcolor=accent \
  --include-in-header=tools/report-header.tex -o docs/report.pdf
