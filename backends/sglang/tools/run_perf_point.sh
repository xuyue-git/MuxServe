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
# INLINE_START=1 (controller STARTs a batch the moment it is startable: config inline_start=true),
# SKIP_P=1 (decode batches involve only D ranks: config decode_skip_prefill_ranks=true),
# EARLY_TOKEN=1 (publish the prefill token at commit, not after the D handoff:
# config prefill_early_token=true; moves the TTFT boundary, adds the handoff to TPOT),
# CLEANUP_SKIP_P=1 (request CLEANUP only on D ranks once P released it: config
# cleanup_skip_prefill_ranks=true), CLEANUP_WITH_QUIESCE=1 (queue a finished request's
# CLEANUP behind its last D QUIESCE: config cleanup_with_quiesce=true), TRIM_MIB=N
# (empty_cache only with >= N MiB in wholly free cached segments: config idle_cache_trim_mib=N),
# HANDOFF_PREPARE=1 (first decode batch prepared behind the HANDOFF: config
# handoff_prepare_decode=true; needs INLINE_START, EARLY_TOKEN, SKIP_P),
# INLINE_META=1 (HANDOFF carries staged P metadata: config handoff_inline_metadata=true),
# FINAL_STEP=1 (final run step inside GPU_DONE: config final_step_in_gpu_done=true),
# EVENT_PUMP=1 (events to the gateway on their own thread: config event_pump_thread=true),
# BOOTSTRAP_POLL=1 (P bootstrap readiness in long-poll replies: config bootstrap_in_poll=true),
# INLINE_PREPARE=1 (prefill prepared at submit: config inline_prepare=true; needs INLINE_START),
# OWN_TABLE=1 (PREPARE carries/validates the receiving rank's table: config prepare_own_rank_table=true),
# REGISTER_READY=1 (D destinations ride in READY: config register_in_ready=true),
# BROKER_POLL=1 (broker status in long-poll replies: config broker_status_in_poll=true),
# EARLY_TERMINAL=1 (TERMINAL when CLEANUP is queued: config terminal_before_cleanup=true),
# SLOT_AT_CLEANUP=1 (D-only cleanup frees the slot: config slot_free_at_cleanup=true; needs CLEANUP_SKIP_P),
# DELIVER_FIRST=1 (HANDOFF reaches D before the decode batch is reserved: config
# handoff_deliver_first=true; needs HANDOFF_PREPARE), ACK_IN_POLL=1 (broker ACK rides on the
# long poll: config handoff_ack_in_poll=true; needs BROKER_POLL), COMBINED_REPORT=1
# (HANDOFF_READY and READY in one RPC: config handoff_report_with_ready=true; needs
# HANDOFF_PREPARE), START_IN_REPLY=1 (START in the last READY's reply: config
# start_in_report_reply=true), TARGETED_WAKE=1 (long polls woken per rank: config targeted_wake=true),
# RPC_INLINE=1 (short control RPCs on the connection thread: config rpc_inline_fast_methods=true),
# FROZEN_BUNDLE=/abs/frozen-synthetic-512-128.json (original MuxServe's fixed-workload
# input per model; every output must equal its expected tokens), WARMUP_TOTAL=128
# (warmup split 43/43/42 over the models as the original; overrides WARMUP),
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
INLINE_START="${INLINE_START:-}"
SKIP_P="${SKIP_P:-}"
EARLY_TOKEN="${EARLY_TOKEN:-}"
CLEANUP_SKIP_P="${CLEANUP_SKIP_P:-}"
CLEANUP_WITH_QUIESCE="${CLEANUP_WITH_QUIESCE:-}"
TRIM_MIB="${TRIM_MIB:-}"
HANDOFF_PREPARE="${HANDOFF_PREPARE:-}"
INLINE_META="${INLINE_META:-}"
FINAL_STEP="${FINAL_STEP:-}"
EVENT_PUMP="${EVENT_PUMP:-}"
BOOTSTRAP_POLL="${BOOTSTRAP_POLL:-}"
INLINE_PREPARE="${INLINE_PREPARE:-}"
OWN_TABLE="${OWN_TABLE:-}"
REGISTER_READY="${REGISTER_READY:-}"
BROKER_POLL="${BROKER_POLL:-}"
EARLY_TERMINAL="${EARLY_TERMINAL:-}"
SLOT_AT_CLEANUP="${SLOT_AT_CLEANUP:-}"
DELIVER_FIRST="${DELIVER_FIRST:-}"
ACK_IN_POLL="${ACK_IN_POLL:-}"
COMBINED_REPORT="${COMBINED_REPORT:-}"
START_IN_REPLY="${START_IN_REPLY:-}"
TARGETED_WAKE="${TARGETED_WAKE:-}"
RPC_INLINE="${RPC_INLINE:-}"
FROZEN_BUNDLE="${FROZEN_BUNDLE:-}"
WARMUP_TOTAL="${WARMUP_TOTAL:-}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

"$PY" - "$RUNTIME/configs/sglang_a100.json" "$OUT/config.json" "$LOCKS" "$MODELS_ROOT" \
    "$RUN_STEPS" "$KV_FRACTION" "$HEADROOM_MIB" "$DECODE_GRAPH" "$PARK" "$PREFILL_GRAPH" \
    "$INLINE_START" "$SKIP_P" "$EARLY_TOKEN" "$CLEANUP_SKIP_P" "$CLEANUP_WITH_QUIESCE" "$TRIM_MIB"     "$HANDOFF_PREPARE" "$INLINE_META" "$FINAL_STEP" "$EVENT_PUMP" "$BOOTSTRAP_POLL" "$INLINE_PREPARE" "$OWN_TABLE" "$REGISTER_READY" "$BROKER_POLL" "$EARLY_TERMINAL" "$SLOT_AT_CLEANUP" "$DELIVER_FIRST" "$ACK_IN_POLL" "$COMBINED_REPORT" "$START_IN_REPLY" "$TARGETED_WAKE" "$RPC_INLINE" <<'PYEOF'
import json, sys
(source, target, locks, root, steps, fraction, headroom, graph, park, pgraph,
 inline, skip_p, early, cleanup_skip_p, cleanup_with_quiesce, trim_mib,
 handoff_prepare, inline_meta, final_step, event_pump, bootstrap_poll,
 inline_prepare, own_table, register_ready, broker_poll, early_terminal,
 slot_at_cleanup, deliver_first, ack_in_poll, combined_report, start_in_reply,
 targeted_wake, rpc_inline) = sys.argv[1:]
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
for key, value in (("inline_start", inline), ("decode_skip_prefill_ranks", skip_p),
                   ("prefill_early_token", early), ("cleanup_skip_prefill_ranks", cleanup_skip_p),
                   ("cleanup_with_quiesce", cleanup_with_quiesce),
                   ("handoff_prepare_decode", handoff_prepare), ("handoff_inline_metadata", inline_meta),
                   ("final_step_in_gpu_done", final_step), ("event_pump_thread", event_pump),
                   ("bootstrap_in_poll", bootstrap_poll), ("inline_prepare", inline_prepare),
                   ("prepare_own_rank_table", own_table), ("register_in_ready", register_ready),
                   ("broker_status_in_poll", broker_poll), ("terminal_before_cleanup", early_terminal),
                   ("slot_free_at_cleanup", slot_at_cleanup),
                   ("handoff_deliver_first", deliver_first), ("handoff_ack_in_poll", ack_in_poll),
                   ("handoff_report_with_ready", combined_report),
                   ("start_in_report_reply", start_in_reply), ("targeted_wake", targeted_wake),
                   ("rpc_inline_fast_methods", rpc_inline)):
    config.pop(key, None)
    if value:
        config[key] = True
config.pop("idle_cache_trim_mib", None)
if trim_mib:
    config["idle_cache_trim_mib"] = int(trim_mib)
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
      "prefill_graph=%s" % config.get("prefill_cuda_graph", False),
      "inline_start=%s" % config.get("inline_start", False),
      "skip_p=%s" % config.get("decode_skip_prefill_ranks", False),
      "early_token=%s" % config.get("prefill_early_token", False),
      "cleanup_skip_p=%s" % config.get("cleanup_skip_prefill_ranks", False),
      "cleanup_with_quiesce=%s" % config.get("cleanup_with_quiesce", False),
      "trim_mib=%s" % config.get("idle_cache_trim_mib", 0),
      "handoff_prepare=%s" % config.get("handoff_prepare_decode", False),
      "inline_meta=%s" % config.get("handoff_inline_metadata", False),
      "final_step=%s" % config.get("final_step_in_gpu_done", False),
      "event_pump=%s" % config.get("event_pump_thread", False),
      "bootstrap_poll=%s" % config.get("bootstrap_in_poll", False),
      "inline_prepare=%s" % config.get("inline_prepare", False),
      "own_table=%s" % config.get("prepare_own_rank_table", False),
      "register_ready=%s" % config.get("register_in_ready", False),
      "broker_poll=%s" % config.get("broker_status_in_poll", False),
      "early_terminal=%s" % config.get("terminal_before_cleanup", False),
      "slot_at_cleanup=%s" % config.get("slot_free_at_cleanup", False),
      "deliver_first=%s" % config.get("handoff_deliver_first", False),
      "ack_in_poll=%s" % config.get("handoff_ack_in_poll", False),
      "combined_report=%s" % config.get("handoff_report_with_ready", False),
      "start_in_reply=%s" % config.get("start_in_report_reply", False),
      "targeted_wake=%s" % config.get("targeted_wake", False),
      "rpc_inline=%s" % config.get("rpc_inline_fast_methods", False))
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
    ${REFERENCE:+--reference "$REFERENCE"} ${FROZEN_BUNDLE:+--frozen-bundle "$FROZEN_BUNDLE"} \
    ${WARMUP_TOTAL:+--warmup-total "$WARMUP_TOTAL"} 2>&1 | tee "$OUT/bench.log"
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
