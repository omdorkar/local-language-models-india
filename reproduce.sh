#!/usr/bin/env bash
# End-to-end reproduction of "Language and Script Gaps in Local Language Models for Indian Users": from a fresh clone to every result table and figure.
#
#   ./reproduce.sh            # everything below, in order
#   ./reproduce.sh --check    # only check the prerequisites, then stop
#
# TIME: about 2 days on an Apple M4 Pro MacBook (24 GB). The original run took 45.6 hours of model inference
# (runs/run_all.log), plus about 5 hours to build the machine-made conditions (NLLB translation 4.2 h) and
# downloads. Run it on mains power with the lid open, and close other heavy apps (only one model may be in memory).
#
# PREREQUISITES THAT THIS SCRIPT DOES NOT DO FOR YOU (they are large, interactive or machine-specific):
#   1. Tools on PATH: uv, git, ollama (running: `ollama serve`), llama.cpp (`llama-server`, `llama-quantize`; build b11669 was used).
#   2. The Ollama models, pulled once (about 36 GB):
#        ollama pull gemma3:12b ; ollama pull qwen3:8b ; ollama pull llama3.1:8b ; ollama pull gpt-oss:20b
#        ollama pull llama3.1:8b-instruct-q3_K_M          # sensitivity check for the Q3 confound
#      Exact digests used: reports/model_versions.md (Ollama 0.32.5).
#   3. The Sarvam-30B Q3_K_M file at models/sarvam-30b/sarvam-30b-Q3_K_M.gguf. It is built locally from a community BF16
#      GGUF with llama-quantize; follow reports/sarvam_setup.md exactly (the official Q4_K_M does not fit in 19 GB of GPU memory).
#
# RESUMABLE: every step is idempotent. Re-run this script after any interruption: environments are reused, downloads are
# verified by hash, sample/condition builders use caches, and scripts/run_all.sh skips every (model, condition, item) row that is
# already in runs/. Results are deterministic only up to runtime version and hardware (see paper/paper.md section 3.3).
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
BR=.venv-bridge/bin/python
EX=.venv-extra/bin/python
step() { echo; echo "##### $(date '+%F %T')  $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------------------------- 0. prerequisites
step "0. Check prerequisites"
for tool in uv git ollama llama-server llama-quantize; do
  command -v "$tool" >/dev/null || die "'$tool' not found on PATH"
done
curl -fsS http://127.0.0.1:11434/api/version >/dev/null || die "Ollama is not running (start it with: ollama serve)"
PULLED="$(ollama list | awk 'NR>1 {print $1}')"
for tag in gemma3:12b qwen3:8b llama3.1:8b gpt-oss:20b llama3.1:8b-instruct-q3_K_M; do
  grep -qx "$tag" <<<"$PULLED" || die "Ollama model '$tag' is not pulled (see the header of this script)"
done
[[ -f models/sarvam-30b/sarvam-30b-Q3_K_M.gguf ]] || die "models/sarvam-30b/sarvam-30b-Q3_K_M.gguf is missing (build it with reports/sarvam_setup.md)"
echo "prerequisites OK"
[[ "${1:-}" == "--check" ]] && exit 0

# ---------------------------------------------------------------------------------------------------------------- 1. main environment
step "1. Main environment (.venv, Python 3.11) + data download"
uv venv --python 3.11 --allow-existing
uv pip install -e ".[dev,analysis]"
$PY -m evaluation.download                      # Belebele, Global-MMLU(-Lite), COMI-LINGUA (~110 MB); writes data/raw/MANIFEST.json
# Optional (not used by any analysis script): $PY -m evaluation.download --dakshina   # streams 2 GB, keeps 190 MB
$PY -m evaluation.inspect_belebele              # -> reports/belebele_scripts.md
$PY -m evaluation.model_fit                     # -> reports/model_fit.md

# ---------------------------------------------------------------------------------------------------------------- 2. samples
step "2. Item samples (seed 1909) -> data/derived/samples/"
$PY -m evaluation.samples

# ---------------------------------------------------------------------------------------------------------------- 3. bridge: +xlit and +mt_en
step "3. Bridge environment (.venv-bridge, Python 3.10) and fixes: IndicXlit (+xlit) and NLLB (+mt_en)"
if ! $BR -c "import importlib.metadata as m; m.version('fairseq'); m.version('ai4bharat-transliteration')" 2>/dev/null; then
  # Same commands as bridge/README.md (setup section). fairseq must be built from its git tag.
  uv venv .venv-bridge --python 3.10 --allow-existing
  uv pip install --python $BR "numpy<2" "cython<3" wheel "setuptools<70" "torch==2.5.1"
  FAIRSEQ_SRC="${TMPDIR:-/tmp}/fairseq-src"
  [[ -d "$FAIRSEQ_SRC" ]] || git clone --depth 1 --branch v0.12.2 https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_SRC"   # commit 4a388e64cd64
  CXXFLAGS=-Wno-invalid-specialization CFLAGS=-Wno-invalid-specialization \
    uv pip install --python $BR --no-build-isolation "$FAIRSEQ_SRC"
  uv pip install --python $BR --no-deps ai4bharat-transliteration==1.1.3 urduhack==1.1.1
  uv pip install --python $BR -r bridge/requirements-bridge.txt
fi
# 4 x +xlit and 13 x +mt_en files (300 rows each) + validation reports. About 4.5 h cold (NLLB: 4.2 h), ~35 s with warm caches.
# Downloads on first use: IndicXlit models (~1 GB) and NLLB (5.5 GB; an fp16 copy is cached in data/derived/cache/).
$BR -m bridge.build_conditions

# ---------------------------------------------------------------------------------------------------------------- 4. extra: +inde and +cm
step "4. Extra environment (.venv-extra, Python 3.10) and exploratory conditions: Indian English (+inde), Hinglish (+cm)"
if ! $EX -c "import importlib.metadata as m; m.version('value-nlp'); m.version('simalign')" 2>/dev/null; then
  # Same commands as extra/README.md (setup section).
  uv venv --python 3.10 .venv-extra --allow-existing
  uv pip install --python $EX -r extra/requirements-extra.txt
  uv pip install --python $EX \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
  NLTK_DATA="$PWD/.venv-extra/nltk_data" $EX -m nltk.downloader -d .venv-extra/nltk_data wordnet cmudict
fi
$EX -m extra.inde        # about 40-60 min on CPU (cached in reports/extra/inde_field_cache_v2.json)
$EX -m extra.cm          # about 3-5 min after model download

# ---------------------------------------------------------------------------------------------------------------- 5. the experiment
step "5. Run all models (76,400 answers; ~45 h). Resumable; log: runs/run_all.log"
# Starts/stops llama-server for Sarvam itself and unloads Ollama models between models.
# Order: Belebele (5 models x 18 conditions) -> Global-MMLU-Lite -> Global-MMLU -> Llama Q3 control -> +xlit -> +mt_en -> +inde/+cm.
scripts/run_all.sh
# Use `scripts/run_all.sh --dry-run` to print the plan without calling any model.

# ---------------------------------------------------------------------------------------------------------------- 6. analysis
step "6. Statistics and figures"
$PY -m evaluation.stats        # -> reports/results/summary.md and CSVs (10,000 bootstrap resamples, seed 1909)
$PY -m evaluation.exploratory_style  # -> reports/results/exploratory_style.{csv,md} (Indian English, Hinglish)
$PY -m evaluation.plots        # -> reports/figures/fig1..fig6 (.png and .svg)

# ---------------------------------------------------------------------------------------------------------------- 7. tests
step "7. Tests"
.venv/bin/pytest -q

step "done. Results: reports/results/summary.md. Dashboard: .venv/bin/streamlit run dashboard/app.py --server.address localhost"
