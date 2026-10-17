#!/usr/bin/env bash
# Fixed PTQ vs best QAT evaluation only. Never trains, calibrates or builds engines.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

stage="${1:-all}"
tag="${INDEPENDENT_TAG:-v1}"
if [[ ! "$tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "INDEPENDENT_TAG must contain only letters, digits, underscores or hyphens" >&2
  exit 2
fi
case "$stage" in all|prepare|ptq|qat|compare) ;; *)
  echo "Usage: bash server_opt_0101/run_independent_eval.sh [all|prepare|ptq|qat|compare]" >&2
  exit 2 ;;
esac

ptq_config="server_opt_0101/config.old-awq-int4-lmhead-mixed-proj-vision-fp16-blocks-residual-a07.json"
qat_config="outputs/qat-teacher-512-v1/config.qat.json"
data="dataset/eval-independent-${tag}"
ptq_out="${PTQ_EVAL_OUT:-outputs/eval-independent-${tag}-ptq}"
qat_out="${QAT_EVAL_OUT:-outputs/eval-independent-${tag}-qat}"
comparison="${EVAL_COMPARISON_OUT:-outputs/eval-independent-${tag}-comparison.json}"

if [[ "$stage" == all || "$stage" == prepare ]]; then
  # Lock the candidate before inspecting fresh evaluation results.
  python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/qat-teacher-512-v1/qat_report.json")
d = json.loads(p.read_text())
if d.get("status") != "complete" or d.get("best_step") != 60:
    raise SystemExit("Expected completed QAT run with best_step=60; verify the fixed candidate first")
print("Locked candidates: original mixed PTQ vs QAT best step 60")
PY
  python server_opt_0101/prepare_public_eval.py \
    --root . --config "$ptq_config" \
    --exclude dataset/qat-vqav2-new-v1/train.jsonl \
    --exclude dataset/qat-vqav2-new-v1/eval.jsonl \
    --per-task 128 --out "$data"
fi

if [[ "$stage" == all || "$stage" == ptq || "$stage" == qat ]]; then
  python - "$data" <<'PY'
import json
import sys
from pathlib import Path
d = json.loads((Path(sys.argv[1]) / "DATASET_INFO.json").read_text())
if d.get("status") != "complete" or d.get("sample_count") != 512:
    raise SystemExit("Refusing to evaluate an incomplete 512-image suite")
PY
fi
if [[ "$stage" == all || "$stage" == ptq ]]; then
  python server_opt_0101/evaluate_llm.py \
    --root . --config "$ptq_config" --eval "$data/eval.jsonl" --out "$ptq_out"
fi
if [[ "$stage" == all || "$stage" == qat ]]; then
  python server_opt_0101/evaluate_llm.py \
    --root . --config "$qat_config" --eval "$data/eval.jsonl" --out "$qat_out"
fi
if [[ "$stage" == all || "$stage" == compare ]]; then
  python server_opt_0101/compare_llm_reports.py \
    --baseline "$ptq_out/alignment.json" --candidate "$qat_out/alignment.json" \
    --out "$comparison"
fi
