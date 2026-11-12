# Runtime switches and control-plane behaviour

All switches below are deployment-config fields (see `muxserve/config.py`), off by
default; `tools/run_perf_point.sh` maps the upper-case environment names to them.

## Decode runs and control-plane performance

`decode_run_steps` (default 1 = original per-token grants; the sample config
uses 128) lets one D grant cover several consecutive decode forwards. KV for
the whole run is reserved once; every TP rank reports each step through
`decode_step`, a token is published once all ranks agree, and one memoized
continue decision per step keeps ranks in lockstep (with
`decode_leader_reports`, D rank 0 alone reports a continuing step and the other
ranks read its decision from shared memory). A run yields at the next
step boundary when a prefill waits for credits, cancellation/timeout arrives,
or capacity recovery protects another request. Ranks keep one authenticated
RPC connection, long-poll their commands, and the driver wakes on controller
activity instead of polling. `tools/bench_fixed.py` reproduces the fixed
512/128 closed-loop point against the gateway (`/metrics` exposes the
controller's per-stage totals); `acceptance/control_plane_e2e_check.py` runs
the full control plane with simulated rank processes on CPU.

`decode_cuda_graph: true` captures D-stage decode forwards as SGLang-native
full CUDA graphs (batch sizes 1..`max_model_inflight`; P stays eager). Capture
runs at model-runner initialization, before the rank registers, on static
per-layer slot / page-table / length buffers sized for 2048-token contexts.
Every replay still passes the per-forward grant check and `validate_forward`
of the live batch; the validated grant is then copied into those buffers.

`decode_run_park: true` changes how a decode run yields to a waiting prefill.
Instead of ending (GPU_DONE, KV commit, quiesce) and being re-prepared later,
the run parks at the step boundary: once every D rank finished that step its
credits are released, while its KV reservation and native state stay. The
ranks long-poll `decode_resume`; the run resumes when no prefill holds or
awaits credits and its own credits are free again, or ends exactly like an
ordinary yield on cancel/timeout/fence or after 2 s parked. D wall-time
feedback then includes the parked time. In this mode a prefill also prepares
(KV reservation, PREPARE/READY handshake) without credits while decode runs
keep going; it takes all credits at START. Runs park only for a READY
prefill, and no new decode batch is prepared while one waits.

`prefill_cuda_graph: true` enables SGLang's breakable prefill CUDA graph on
P ranks for prompts of exactly 512 tokens (`launcher.PREFILL_GRAPH_TOKENS`;
other lengths stay eager). The transformer body is captured in segments at
model-runner init; Head-wise attention runs eagerly at every graph break.
Capture writes no KV. On replay the live batch was already validated by the
grant check, the runner's static batch must carry exactly that grant's
geometry, and K/V go to the grant's write slots.

Opt-in control-plane fast paths (all off by default): `inline_start` lets the
controller START a batch as soon as it is startable instead of on the driver's
next pass; `decode_skip_prefill_ranks` runs decode batches with the D ranks
only (P ranks no longer prepare/quiesce decode batches they never execute);
`prefill_early_token` publishes the prefill token once every P rank's result
is committed, as original MuxServe returns it at prefill end (it moves the
TTFT boundary; the request is not shorter). `tools/run_perf_point.sh` takes
`INLINE_START=1 SKIP_P=1 EARLY_TOKEN=1`, and `FROZEN_BUNDLE=... WARMUP_TOTAL=128`
to bench original MuxServe's frozen 512/128 inputs with its 43/43/42 warmup,
checking every output against the original's tokens.

The remaining opt-in paths are grouped by where they act. Each is documented
where it is defined in `muxserve/config.py`; `benchmark/paired/paired_a100.json`
turns them all on, and `tools/run_perf_point.sh` lists the matching switches.

| Stage | Config keys |
|---|---|
| Before the first token | `inline_prepare`, `prepare_own_rank_table`, `register_in_ready`, `bootstrap_in_poll` |
| Prefill to first decode step | `handoff_prepare_decode`, `handoff_inline_metadata`, `broker_status_in_poll`, `handoff_ack_in_poll`, `handoff_report_with_ready`, `handoff_prepare_after_ack`, `decode_prepare_light` |
| Decode steps | `decode_leader_reports`, `final_step_in_gpu_done` |
| Request end | `cleanup_skip_prefill_ranks`, `cleanup_with_quiesce`, `idle_cache_trim_mib`, `terminal_before_cleanup`, `slot_free_at_cleanup` |
| Control transport | `event_pump_thread`, `start_in_report_reply`, `targeted_wake`, `rpc_inline_fast_methods` |

Open-loop load (requests sent at trace arrival times, so many wait in the
queue) uses larger limits and two opt-in KV paths, all off or unchanged by
default: `max_model_inflight` (1..16), `max_new_tokens` (256..2047),
`request_timeout_ms` and `gateway_connections` (one HTTP stream per queued
request). `capacity_recovery_all_decode` handles a full KV pool as original
MuxServe does: only new prefills wait, every decode that still fits keeps
running, a one-step decode batch that cannot get its KV preempts the youngest
idle decoding request younger than the batch, and a finished request lets
prefills resume. `decode_kv_trim` returns a decode run's unused up-front KV
reservation to the pool when the run ends. `tools/run_perf_point.sh` takes
`INFLIGHT`, `MAX_NEW_TOKENS`, `REQUEST_TIMEOUT_MS`, `GATEWAY_CONNECTIONS`,
`RECOVERY_ALL_DECODE=1` and `KV_TRIM=1`. Independently of any switch, the
gateway wakes only the stream whose request changed, a request stays known
until its batch is fed back, and a decode run is shortened so that its event
reservation fits the group's event window (this only applies to batches of
eight or more requests).

Two further opt-in switches change how a decode run admits work, both off by
default. With `decode_step_join` a running decode run no longer ends for its
model's next request: at a step boundary its D ranks take that prefill's
PREPARE, REVOKE, CLEANUP and KV HANDOFF (the owner holds every rank at the
same boundary until each handoff, which contains a TP collective, finished on
all of them), and a request whose handoff finished joins the running batch at
the next boundary, up to the batch limit, as original MuxServe adds requests
to its running batch. With `decode_kv_per_page` (requires `decode_step_join`)
a run reserves only its first step's page and gets one more page at the
boundary before a request needs it; a page that cannot be had ends the run and
the full-pool recovery above applies. `tools/run_perf_point.sh` takes
`STEP_JOIN=1` and `KV_PER_PAGE=1`. With `decode_step_pipeline` (requires
`decode_leader_reports`; `STEP_PIPELINE=1`) the decode leader no longer waits
for the owner at a step boundary the owner granted ahead: each answer to a step
report also says whether the next boundary may simply continue, and the leader
then reports that step while the next forward runs. A boundary with anything
new for the run (a prefill waiting for credits, prefill work for its D ranks, a
request to join, a page needed, a request stopping) is decided as before, one
step later than without the switch. With `decode_step_leave` (requires
`decode_step_join` and `decode_leader_reports`; `STEP_LEAVE=1`) a request that
finishes, is cancelled or times out leaves the running decode run at the next
step boundary: its KV is committed on its own and released by the ordinary
cleanup while the others go on, and a run is sized by its longest member rather
than its shortest. `decode_kv_grow_pages` (1..8, `KV_GROW_PAGES=N`) sets how many
pages `decode_kv_per_page` adds at a time; a boundary that grows one member also
tops up every member within a page of its end, and only the new pages are sent.
Three switches take work off the owner process, all off by default:
`decode_wait_commands` (`WAIT_COMMANDS=1`; requires `decode_step_join` and
`handoff_ack_in_poll`) lets a D rank waiting at a run boundary take its
commands, broker state and handoff ACKs through the `decode_resume` call it
makes anyway; `prefill_stage_in_gpu_done` (`STAGE_IN_DONE=1`) has a P rank
send its staged KV metadata inside GPU_DONE instead of one `broker_stage` RPC
per request; `driver_min_interval_ms` (0..50, `DRIVER_INTERVAL_MS=N`) spaces
driver passes and builds the policy snapshot from one batch index.

Prefill and decode overlap, all off by default: `prefill_graph_tokens`
(`PREFILL_GRAPH_TOKENS=16,32,...`) sets the token counts captured for prefill
CUDA graphs; `prefill_break_slim` (`BREAK_SLIM=1`) makes each layer's graph
break lighter (same results, checked on GPU by
`acceptance/graph_break_slim_gpu_check.py`); `prefill_start_unparked`
(`START_UNPARKED=1`) starts a READY prefill before every decode run parked;
`handoff_local_success` (`HANDOFF_LOCAL=1`) lets a waiting D rank pop a
validated handoff at once; `prefill_overlap_decode` (`OVERLAP=1`, requires
`decode_run_park`) keeps decode runs running while a prefill runs, with MPS
sharing the GPUs. `kv_release_at_finish` (`KV_EARLY_FREE=1`) returns a finished
request's KV when it leaves its run instead of after its cleanup.

ADBS: `policy_arrival_rates` (`ARRIVAL_RATES=model=rate,...`) sizes each model's
starting request limit from its arrival rate exactly as the source's
`init_max_num_seqs`; `policy_step_history` (`STEP_HISTORY=1`) feeds the 8 s
re-sizing one batch size per decode-run step, as the source appends
`hist_num_seqs` per decode batch (without it a run is one entry, which
understates busy models under `decode_step_join`/`decode_step_leave`);
`policy_log_adjustments` (`ADBS_LOG=1`) prints one `POLICY_ADJUST` line per
re-sizing.

Owner process, all off by default: `gateway_process` (`GATEWAY_PROC=1`) runs
the HTTP gateway (sockets, SSE, tokenizer) in a child process fed by one pipe;
`event_lean_copies` (`LEAN_EVENTS=1`) passes runtime events on without deep
copies; `driver_lean_locking` (`DRIVER_LEAN=1`) takes the owner lock fewer
times per driver pass; `owner_switch_interval_us` (`OWNER_SWITCH_US=500`) sets
the owner's interpreter switch interval. `acceptance/owner_speed_check.py`
covers them on CPU.

The fastest open-loop ShareGPT configuration so far (alpha 0.7, rate scale 1.0,
257 requests; three rounds 510-525 tok/s against the original's 480.56) is the
xbench common switches plus `OPEN_LOOP=1 RECOVERY_ALL_DECODE=1 KV_TRIM=1
STEP_JOIN=1 STEP_PIPELINE=1 STEP_LEAVE=1 KV_PER_PAGE=1 KV_GROW_PAGES=4
WAIT_COMMANDS=1 STAGE_IN_DONE=1 DRIVER_INTERVAL_MS=5 PER_MODEL=16
PAIR_HOLD_MS=10 START_UNPARKED=1 HANDOFF_LOCAL=1 PREFILL_GRAPH_TOKENS=16,32,64,128,256,512,1024,2048
BREAK_SLIM=1 OVERLAP=1 ARRIVAL_RATES=llama7b=2.0,llama13b=1.2311444133449163,llama30b=0.9269261135439395
KV_EARLY_FREE=1 GATEWAY_PROC=1 LEAN_EVENTS=1 DRIVER_LEAN=1 OWNER_SWITCH_US=500
STEP_HISTORY=1`.

## Known limitations

Prefill CUDA graphs (unless `prefill_cuda_graph`), overlap scheduling and custom all-reduce remain disabled.
Each run start still ships and validates the complete multi-rank KV view once
(as packed int32 tables). The sample
configuration uses D30 for 7B and D40 for 13B/30B; it is not the all-D40 paired
experiment configuration. Asset paths now use the common protected resolver,
with no hardcoded lab model aliases. No original-paper performance reproduction
or clean-install GPU validation is claimed for this migration.

The repository excludes raw results, reports and delivery archives. The standard
xbench catalogue for client-mode benchmarks is in `benchmark/xbench/`.
