#!/usr/bin/env bash
# One fixed 512/128 performance point: generate config, launch, benchmark, stop.
#
#   PY=/path/to/.venv/bin/python LOCKS=/abs/asset-locks \
#     tools/run_perf_point.sh OUT_DIR [REQUESTS=64] [WARMUP=4] [RUN_STEPS=128]
#
# RUN_STEPS=0 writes a config without decode_run_steps (for the original code).
# Optional env: GPUS (default 1,2,3,4), MODELS_ROOT, REFERENCE (bench JSON to
# compare output tokens against). The launcher is always stopped with SIGINT
# (normal drain); the script waits for it to exit.
set -euo pipefail
RUNTIME="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: run_perf_point.sh OUT_DIR [REQUESTS] [WARMUP] [RUN_STEPS]}"
REQUESTS="${2:-64}"; WARMUP="${3:-4}"; RUN_STEPS="${4:-128}"
PY="${PY:?set PY to the SGLang environment python}"
LOCKS="${LOCKS:?set LOCKS to the personal asset lock directory}"
MODELS_ROOT="${MODELS_ROOT:-/storage/crosspool-models/huggyllama}"
GPUS="${GPUS:-1,2,3,4}"
REFERENCE="${REFERENCE:-}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

"$PY" - "$RUNTIME/configs/sglang_a100.json" "$OUT/config.json" "$LOCKS" "$MODELS_ROOT" "$RUN_STEPS" <<'PYEOF'
import json, sys
source, target, locks, root, steps = sys.argv[1:]
config = json.load(open(source))
config["asset_lock_dir"] = locks
config.pop("decode_run_steps", None)
if int(steps) > 0:
    config["decode_run_steps"] = int(steps)
for model, size in zip(config["models"], ("7b", "13b", "30b")):
    model["model_path"] = model["tokenizer_path"] = f"{root}/llama-{size}"
    model["decode_mps"] = 40  # paired point: all three models D40
json.dump(config, open(target, "w"), indent=2)
print("CONFIG", target, "decode_run_steps=%s" % config.get("decode_run_steps", "absent"))
PYEOF

busy="$(nvidia-smi -i "$GPUS" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
if [ -n "$busy" ]; then
    echo "GPUS_BUSY on $GPUS; not starting:"; echo "$busy"; exit 3
fi

cd "$RUNTIME"
CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup "$PY" -m muxserve_sglang.launcher "$OUT/config.json" \
    --startup-timeout 7200 > "$OUT/launcher.log" 2>&1 &
SUPERVISOR=$!
echo "$SUPERVISOR" > "$OUT/supervisor.pid"
echo "LAUNCHED supervisor=$SUPERVISOR log=$OUT/launcher.log"

stop_launcher() {
    if kill -0 "$SUPERVISOR" 2>/dev/null; then
        echo "STOPPING supervisor=$SUPERVISOR (normal drain)"
        kill -INT "$SUPERVISOR" || true
        for _ in $(seq 1 120); do
            kill -0 "$SUPERVISOR" 2>/dev/null || { echo "STOPPED"; return; }
            sleep 5
        done
        echo "STOP_TIMEOUT supervisor=$SUPERVISOR still alive; check $OUT/launcher.log and use the scoped cleanup tool"
    fi
}
trap stop_launcher EXIT

"$PY" "$RUNTIME/tools/bench_fixed.py" --port 18080 --requests "$REQUESTS" --warmup "$WARMUP" \
    --ready-timeout 7200 --supervisor-pid "$SUPERVISOR" --out "$OUT/bench.json" \
    ${REFERENCE:+--reference "$REFERENCE"} 2>&1 | tee "$OUT/bench.log"
echo "RESULT $OUT/bench.json"
