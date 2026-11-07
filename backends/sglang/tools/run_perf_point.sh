#!/usr/bin/env bash
# One fixed 512/128 performance point: generate config, launch, benchmark, stop.
#
#   PY=/path/to/.venv/bin/python LOCKS=/abs/asset-locks \
#     tools/run_perf_point.sh OUT_DIR [REQUESTS=64] [WARMUP=4] [RUN_STEPS=128]
#
# RUN_STEPS=0 writes a config without decode_run_steps (for the original code).
# Optional env: GPUS (default 1,2,3,4), MODELS_ROOT, REFERENCE (bench JSON to
# compare output tokens against), KV_FRACTION (size the KV pool like original
# MuxServe: gpu_memory_utilization x total GPU memory, instead of kv_bytes_per_rank),
# HEADROOM_MIB (startup free-memory gate per card; default 2048 from the sample
# config, minimum 1024; lowering it is a deviation from the approved design),
# TRACE=1 (record a timing trace into OUT_DIR/trace and write
# OUT_DIR/trace-report.txt after stopping; recording only), SOLO=1 (with the
# same deployment, afterwards run each model alone: SOLO_REQUESTS, default 8),
# DECODE_GRAPH=1 (D-stage decode CUDA graphs: config decode_cuda_graph=true),
# PARK=1 (decode runs park for a waiting prefill: config decode_run_park=true),
# PREFILL_GRAPH=1 (P-stage breakable prefill CUDA graphs: config prefill_cuda_graph=true),
# PROFILE=1 (torch.profiler on each model's D rank 0 for 10 decode steps at
# its 300th and 1600th run step; summaries in OUT_DIR/profile; diagnosis only;
# MXS_PROFILE_RANKS=P0 MXS_PROFILE_AT=3,12 MXS_PROFILE_STEPS=4 profiles prefills;
# MXS_PROFILE_PY=1 uses cProfile instead: Python functions by self/cum time). The launcher is always stopped with SIGINT
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
KV_FRACTION="${KV_FRACTION:-}"
HEADROOM_MIB="${HEADROOM_MIB:-}"
TRACE="${TRACE:-}"
SOLO="${SOLO:-}"
SOLO_REQUESTS="${SOLO_REQUESTS:-8}"
PROFILE="${PROFILE:-}"
DECODE_GRAPH="${DECODE_GRAPH:-}"
PARK="${PARK:-}"
PREFILL_GRAPH="${PREFILL_GRAPH:-}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

"$PY" - "$RUNTIME/configs/sglang_a100.json" "$OUT/config.json" "$LOCKS" "$MODELS_ROOT" "$RUN_STEPS" "$KV_FRACTION" "$HEADROOM_MIB" "$DECODE_GRAPH" "$PARK" "$PREFILL_GRAPH" <<'PYEOF'
import json, sys
source, target, locks, root, steps, fraction, headroom, graph, park, pgraph = sys.argv[1:]
config = json.load(open(source))
config["asset_lock_dir"] = locks
if fraction:
    config.pop("kv_bytes_per_rank", None)
    config["kv_memory_fraction"] = float(fraction)
if headroom:
    config["extra_headroom_bytes_per_rank"] = int(headroom) * 1024**2
config.pop("decode_run_steps", None)
config.pop("decode_cuda_graph", None)
if graph:
    config["decode_cuda_graph"] = True
config.pop("decode_run_park", None)
if park:
    config["decode_run_park"] = True
config.pop("prefill_cuda_graph", None)
if pgraph:
    config["prefill_cuda_graph"] = True
if int(steps) > 0:
    config["decode_run_steps"] = int(steps)
for model, size in zip(config["models"], ("7b", "13b", "30b")):
    model["model_path"] = model["tokenizer_path"] = f"{root}/llama-{size}"
    model["decode_mps"] = 40  # paired point: all three models D40
json.dump(config, open(target, "w"), indent=2)
print("CONFIG", target, "decode_run_steps=%s" % config.get("decode_run_steps", "absent"),
      "kv=%s" % (("fraction %s" % config["kv_memory_fraction"]) if "kv_memory_fraction" in config
                 else "%d bytes" % config["kv_bytes_per_rank"]),
      "headroom=%dMiB" % (config["extra_headroom_bytes_per_rank"] // 1024**2),
      "decode_graph=%s" % config.get("decode_cuda_graph", False),
      "park=%s" % config.get("decode_run_park", False),
      "prefill_graph=%s" % config.get("prefill_cuda_graph", False))
PYEOF

busy="$(nvidia-smi -i "$GPUS" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
if [ -n "$busy" ]; then
    echo "GPUS_BUSY on $GPUS; not starting:"; echo "$busy"; exit 3
fi

cd "$RUNTIME"
PHASE_ARGS=()
if [ -n "$TRACE" ]; then
    rm -rf "$OUT/trace" "$OUT/phases.txt"
    export MXS_TRACE_DIR="$OUT/trace"
    PHASE_ARGS=(--phase-file "$OUT/phases.txt")
    echo "TRACE dir=$MXS_TRACE_DIR"
fi
if [ -n "$PROFILE" ]; then
    rm -rf "$OUT/profile"
    export MXS_PROFILE_DIR="$OUT/profile"
    echo "PROFILE dir=$MXS_PROFILE_DIR at=${MXS_PROFILE_AT:-300,1600} steps=${MXS_PROFILE_STEPS:-10} ranks=${MXS_PROFILE_RANKS:-D0} python=${MXS_PROFILE_PY:-0}"
fi
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
    ${PHASE_ARGS[@]+"${PHASE_ARGS[@]}"} ${TRACE:+--phase-name concurrent} \
    ${REFERENCE:+--reference "$REFERENCE"} 2>&1 | tee "$OUT/bench.log"
echo "RESULT $OUT/bench.json"

if [ -n "$SOLO" ]; then
    for model in llama7b llama13b llama30b; do
        "$PY" "$RUNTIME/tools/bench_fixed.py" --port 18080 --models "$model" \
            --requests "$SOLO_REQUESTS" --warmup 1 --ready-timeout 600 \
            --supervisor-pid "$SUPERVISOR" --out "$OUT/bench-solo-$model.json" \
            ${PHASE_ARGS[@]+"${PHASE_ARGS[@]}"} ${TRACE:+--phase-name "solo-$model"} 2>&1 | tee "$OUT/bench-solo-$model.log"
    done
fi

if [ -n "$TRACE" ]; then
    stop_launcher
    "$PY" "$RUNTIME/tools/trace_report.py" "$OUT/trace" --phases "$OUT/phases.txt" \
        --json "$OUT/trace-report.json" | tee "$OUT/trace-report.txt"
    echo "TRACE_REPORT $OUT/trace-report.txt"
fi

if [ -n "$PROFILE" ]; then
    stop_launcher
    for file in "$OUT"/profile/*.txt "$OUT"/profile/*.error; do
        [ -e "$file" ] || continue
        echo "===== $(basename "$file")"; cat "$file"
    done
fi
