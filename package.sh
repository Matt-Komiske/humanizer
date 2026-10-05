#!/bin/bash
# Make a shareable zip of this app (code only, under 1 MB). The other computer's first run
# downloads the free models automatically.
cd "$(dirname "$0")" || exit 1
OUT="../humanizer.zip"
rm -f "$OUT"
(cd .. && zip -rq humanizer.zip humanizer -x "humanizer/.venv/*" -x "humanizer/models/*" \
    -x "*/__pycache__/*" -x "*.DS_Store" -x "humanizer/eval/data/*")
echo "Created $(cd .. && pwd)/humanizer.zip ($(du -h "$OUT" | cut -f1))"
