#!/usr/bin/env bash
# Whole main run, in order. Resumable: re-run this script after any interruption or failure; rows already
# in runs/<benchmark>/<model>.jsonl are skipped. Everything is logged to runs/run_all.log.
#
#   scripts/run_all.sh            # full run
#   scripts/run_all.sh --dry-run  # print the plan for every step, call no model
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
exec > >(tee -a runs/run_all.log) 2>&1

PY=.venv/bin/python
EXTRA=("$@")  # e.g. --dry-run or --limit 3
status=0
step() { echo; echo "##### $(date '+%F %T')  $*"; }
run() { "$PY" -m evaluation.run "$@" ${EXTRA[@]+"${EXTRA[@]}"} || { echo "!!! step failed (exit $?): $*"; status=1; }; }

step "1. Belebele, 5 models, 18 human conditions"
run --benchmark belebele --models all

step "2. Global-MMLU-Lite (en hi bn or)"
run --benchmark gmmlu_lite --models all

step "3. Global-MMLU (en hi bn te ne)"
run --benchmark gmmlu --models all

step "4. Belebele sensitivity: Llama 3.1 8B at Q3_K_M"
run --benchmark belebele --models llama3.1-8b-q3

# Derived conditions exist only after the transliteration / translation steps have run.
existing() {  # existing <ids...> -> echoes those whose file exists, reports the others
  local c
  for c in "$@"; do
    if [[ -f "data/derived/conditions/belebele/$c.jsonl" ]]; then echo "$c"
    else echo "skipping $c: data/derived/conditions/belebele/$c.jsonl not found" >&2; fi
  done
}

step "5a. Belebele derived: +xlit (romanized -> native script)"
XLIT=(); while IFS= read -r c; do XLIT+=("$c"); done < <(existing hin_Latn+xlit ben_Latn+xlit urd_Latn+xlit npi_Latn+xlit)
if ((${#XLIT[@]})); then run --benchmark belebele --models all --conditions "${XLIT[@]}"; else echo "no +xlit files yet"; fi

step "5b. Belebele derived: +mt_en (native -> English)"
MT=(); while IFS= read -r c; do MT+=("$c"); done < <(existing hin_Deva+mt_en ben_Beng+mt_en urd_Arab+mt_en npi_Deva+mt_en mar_Deva+mt_en tel_Telu+mt_en \
  tam_Taml+mt_en guj_Gujr+mt_en kan_Knda+mt_en mal_Mlym+mt_en pan_Guru+mt_en ory_Orya+mt_en asm_Beng+mt_en)
if ((${#MT[@]})); then run --benchmark belebele --models all --conditions "${MT[@]}"; else echo "no +mt_en files yet"; fi

step "6. Belebele exploratory: Indian English (+inde) and Hinglish (+cm)"
EXP=(); while IFS= read -r c; do EXP+=("$c"); done < <(existing eng_Latn+inde hin_Latn+cm)
if ((${#EXP[@]})); then run --benchmark belebele --models all --conditions "${EXP[@]}"; else echo "no exploratory files yet"; fi

step "done (status $status)"
exit $status
