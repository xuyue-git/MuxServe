# 补丁消融结果（累加）

由 `make_configs.py` 生成，数值列由实测填写；未实测的格子保持 `待测`，不得填估计值。
负载：xbench case `2bdea08f`（Paper Figure 5 point: ShareGPT, alpha 0.7, rate scale 1.0, power-law Poisson 2.0 / 1.2311 / 0.9269 req/s, 60 s, 257 requests）。
每一步一次大实验（不开 TRACE / PROFILE），服务每步重新启动。

| 步 | 提交 | 类型 | 打开的开关 | 配置 | 吞吐 tok/s | 相对上一步 | TTFT p50 / p99 ms | TPOT p50 / p99 ms | 完成 / 失败 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | — | 基线 | （全部关闭；含常开修复与 capacity_recovery_all_decode） | `configs/ablation/s00-baseline.toml` | 待测 | — | 待测 | 待测 | 待测 |
| 1 | `c4fef51` | 性能 | `rpc_persistent_connections` | `configs/ablation/s01-rpc-persistent.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | `b0a54ca/db6d210` | 性能 | `decode_run_steps` | `configs/ablation/s02-decode-runs.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 3 | `46c2a15` | 性能 | `decode_cuda_graph` | `configs/ablation/s03-decode-graph.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | `39ce0df` | 性能 | `decode_run_park` | `configs/ablation/s04-decode-park.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 5 | `a2ba6da` | 性能 | `prefill_cuda_graph` | `configs/ablation/s05-prefill-graph.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 6 | `a1083db` | 性能 | `inline_start`, `decode_skip_prefill_ranks`, `prefill_early_token` | `configs/ablation/s06-control-fast-paths.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 7 | `749fc99` | 性能 | `cleanup_skip_prefill_ranks`, `cleanup_with_quiesce`, `idle_cache_trim_mib` | `configs/ablation/s07-request-end.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 8 | `ad93739` | 性能 | `handoff_prepare_decode`, `handoff_inline_metadata`, `final_step_in_gpu_done`, `event_pump_thread`, `bootstrap_in_poll` | `configs/ablation/s08-handoff-overlap.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 9 | `110395a` | 性能 | `inline_prepare`, `prepare_own_rank_table` | `configs/ablation/s09-prepare-at-submit.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10 | `8fa6708` | 性能 | `register_in_ready`, `broker_status_in_poll`, `terminal_before_cleanup`, `slot_free_at_cleanup` | `configs/ablation/s10-fewer-round-trips-1.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 11 | `1b21ef2` | 性能 | `handoff_ack_in_poll`, `handoff_report_with_ready`, `start_in_report_reply`, `targeted_wake` | `configs/ablation/s11-fewer-round-trips-2.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 12 | `8f5f246` | 性能 | `rpc_inline_fast_methods` | `configs/ablation/s12-rpc-inline.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 13 | `9921cd9` | 性能 | `decode_leader_reports`, `handoff_prepare_after_ack`, `decode_prepare_light` | `configs/ablation/s13-decode-leader.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 14 | `00fb8d0` | 性能 | `decode_run_merge`, `prefill_pair_hold_ms` | `configs/ablation/s14-paired-prefill.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 15 | `aefa2f0` | 性能 | `decode_kv_trim` | `configs/ablation/s15-kv-trim.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 16 | `abfb0a0` | 原版对齐 | `decode_step_join`, `decode_kv_per_page` | `configs/ablation/s16-step-join.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 17 | `c9251ba` | 性能 | `decode_step_pipeline` | `configs/ablation/s17-step-pipeline.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 18 | `82bc15f` | 原版对齐 | `decode_step_leave`, `decode_kv_grow_pages` | `configs/ablation/s18-step-leave.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 19 | `ea0d698` | 性能 | `decode_wait_commands`, `prefill_stage_in_gpu_done`, `driver_min_interval_ms` | `configs/ablation/s19-boundary-round-trips.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 20 | `c96e81d` | 性能 | `prefill_graph_tokens` | `configs/ablation/s20-prefill-buckets.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 21 | `c4da5a3` | 性能 | `prefill_start_unparked`, `handoff_local_success` | `configs/ablation/s21-start-unparked.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 22 | `c0b1de5` | 原版对齐 | `policy_arrival_rates` | `configs/ablation/s22-adbs-arrival-rates.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 23 | `76367a7` | 性能 | `prefill_break_slim` | `configs/ablation/s23-break-slim.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 24 | `0e102f3` | 性能 | `prefill_overlap_decode` | `configs/ablation/s24-prefill-overlap.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 25 | `19318e3` | 原版对齐 | `kv_release_at_finish` | `configs/ablation/s25-kv-early-free.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 26 | `8a16781` | 性能 | `gateway_process`, `event_lean_copies`, `driver_lean_locking`, `owner_switch_interval_us` | `configs/ablation/s26-owner-speed.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 27 | `370794c` | 原版对齐 | `policy_step_history`, `policy_log_adjustments` | `configs/ablation/s27-adbs-step-history.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |
| 28 | `pending` | 原版对齐 | `policy_slot_history` | `configs/ablation/s28-adbs-slot-history.toml` | 待测 | 待测 | 待测 | 待测 | 待测 |

## 每个补丁的动机与对模型适配的影响

| 步 | 提交 | 类型 | 动机 | 对后续模型适配的影响 |
|---|---|---|---|---|
| 1 | `c4fef51` | 性能 | 每次 rank 调 owner 的 RPC 都重新建连接并做双向 HMAC 认证；改为每个 rank 一条连接、只认证一次。会上报告的“4 → 76”是旧代码上的测量。 | 与模型无关（控制面）。 |
| 2 | `b0a54ca/db6d210` | 性能 | 一次 decode 授权连续跑最多 128 步，不再每个 token 走一遍 PREPARE/READY/START/QUIESCE；有 prefill 等待时仍在步边界让出（保持 ADBS 的 prefill 优先）。 | 与模型无关；步数受 max_new_tokens 和事件窗口限制。 |
| 3 | `46c2a15` | 性能 | decode 步被 CPU 发射拖住（7B/13B/30B 的 GPU 时间 10/13/26 ms，整步 28/33/50 ms）；D rank 改用 SGLang 原生 decode CUDA Graph。 | 换模型时要求该模型能被 SGLang 的 decode graph 捕获并走 Head-wise 后端；graph 的上下文容量固定 2048。 |
| 4 | `39ce0df` | 性能 | decode run 给等待的 prefill 让路时“停靠”（保留 KV 和原生状态），不再结束后重新准备。 | 与模型无关。 |
| 5 | `a2ba6da` | 性能 | P rank 使用 SGLang 的可断开 prefill CUDA Graph，每个断点处的注意力由 Head-wise 后端即时计算。 | 要求模型各层支持 SGLang 的可断开 graph；每个捕获的 token 桶都占显存。 |
| 6 | `a1083db` | 性能 | batch 一旦可启动就立即 START，不等 driver 下一轮；decode batch 只涉及 D rank；首 token 由 prefill 直接发布。 | 与模型无关。 |
| 7 | `749fc99` | 性能 | 缩短请求收尾：跳过已释放的 P rank，CLEANUP 紧跟最后一次 QUIESCE，只有可回收显存足够时才 empty_cache。 | 与模型无关。 |
| 8 | `ad93739` | 性能 | P→D 的 KV 交接期间就准备第一个 decode batch；TTFT 路径上的消息更少。 | 与模型无关。 |
| 9 | `110395a` | 性能 | 请求提交时就准备 prefill；每条 PREPARE 只带接收 rank 自己的块表。 | 与模型无关；块表大小随每 rank 的层数×头数增长。 |
| 10 | `8fa6708` | 性能 | 减少 prefill 和收尾前后的协议往返。 | 与模型无关。 |
| 11 | `1b21ef2` | 性能 | 减少从 prefill 到第一个 decode 步的 RPC 往返；long poll 按 rank 唤醒。 | 与模型无关。 |
| 12 | `8f5f246` | 性能 | 只占 owner 锁的短 RPC 在连接线程上直接执行，不再每次调用新建 Python 线程。 | 与模型无关。 |
| 13 | `9921cd9` | 性能 | 只有 D rank 0 汇报继续的步（其他 rank 读共享内存）；第一个 decode 预留推迟到全部 ACK 之后；READY 不再做 GPU 回读。 | 与模型无关；各 rank 的 token 摘要仍在 GPU_DONE 时比对。 |
| 14 | `00fb8d0` | 性能 | 同一模型的请求端到端作为一个 batch；10 ms 内到达的请求并入同一次 prefill，对应原版调度“取走所有等待请求”。 | 与模型无关；prefill graph 也按合并后的 token 数捕获。 |
| 15 | `aefa2f0` | 性能 | decode run 结束时归还多预留的 KV，2 GiB 共享池能接纳更多请求。 | 与模型无关（Head-wise 页）。 |
| 16 | `abfb0a0` | 原版对齐 | 请求在步边界加入正在运行的 decode run，对应原版向运行中的 batch 加请求；KV 按页增长，不再一次预留到底。 | 与模型无关（Head-wise 页）。 |
| 17 | `c9251ba` | 性能 | owner 已答复下一个边界可继续时，leader 在下一次前向计算期间汇报本步。 | 与模型无关。 |
| 18 | `82bc15f` | 原版对齐 | 完成的请求在下一个步边界离开 run，run 继续，对应原版运行中的 batch；每次增长 4 页。 | 与模型无关。 |
| 19 | `ea0d698` | 性能 | 减少 decode 步边界上的 owner 往返；driver 至多每 5 ms 一轮。 | 与模型无关。 |
| 20 | `c96e81d` | 性能 | prefill graph 的 token 桶按负载的 prompt 长度设置。 | 桶与负载和显存相关；换 prompt 分布要重新调。 |
| 21 | `c4da5a3` | 性能 | READY 的 prefill 立即开始；等待中的 D rank 在本地接收已校验的交接。 | 与模型无关。 |
| 22 | `c0b1de5` | 原版对齐 | ADBS 的初始上限按各模型到达率计算，与原版 init_max_num_seqs 完全一致。 | 配置里需要写入负载的各模型到达率。 |
| 23 | `76367a7` | 性能 | 更轻的 prefill graph 断点：每次前向只检查一次授权，Q/K/V 保持视图（结果相同，有 GPU 检查）。 | 绑定 Head-wise 注意力断点的实现。 |
| 24 | `0e102f3` | 性能 | prefill 运行时 decode run 继续运行，由 MPS 分享 GPU（原版的 prefill 与 decode 进程也在 MPS 下并发）。 | 与模型无关；依赖 MPS 百分比。 |
| 25 | `19318e3` | 原版对齐 | 完成的请求立即归还 KV，对应原版释放已完成请求的缓存，不再等 CLEANUP 往返。 | 与模型无关。 |
| 26 | `8a16781` | 性能 | 开环负载下每次 rank 汇报要等 owner 锁 8–16 ms：网关移到子进程、事件少拷贝、driver 少加锁、缩短解释器切换间隔。 | 与模型无关。 |
| 27 | `370794c` | 原版对齐 | ADBS 每个 decode 步记录一次 batch 大小，对应原版每个 decode batch 追加 hist_num_seqs；按 run 记录会低估繁忙模型。policy_log_adjustments 只打日志。 | 与模型无关。 |
| 28 | `pending` | 原版对齐 | ADBS 每个 decode 步记录该模型过了 prefill 的全部占用名额（交接中、等待加入 decode、decode 中、等待清理），对应原版 decode batch 取走全部 running 请求；只记 run 的成员会把名额已满的模型（重负载下的 30B）当成没用满，8 秒重分配把它的名额给了 7B。 | 与模型无关。 |

## 基线里常开、不在表中单独计量的改动

这些提交没有开关，包含在第 0 步里，所以表中没有它们各自的增量。

| 提交 | 改动 |
|---|---|
| `b0a54ca` | Ranks long-poll commands instead of 5 ms sleeps; event-driven driver; multi-step decode grants (the grant length itself is step 2) |
| `db6d210` | Decode runs advance locally on the ranks; numpy KV metadata |
| `65011f3` | Fix: reap per-call RPC workers on persistent connections |
| `485c2b6` | Batch free-block selection; static attention layer checks once per layer |
| `1fd96a5` | Packed block tables in control messages; prefill credits released at GPU_DONE |
| `d135962` | Store views carry packed tables |
| `63e37d5` | reserve_kv takes the needed blocks without sorting the whole free pool |
| `f08e38c` | decode_step wakes only the driver, not every long-poll thread |
| `8b3147e` | Decode credits released at all-rank GPU_DONE |
| `5781fa9` | Prefill attention-backend CPU overhead trimmed |
| `d8bdef6` | One weight audit per idle cache trim |
| `a8f4866` | numpy block tables and free stacks in the KV store |
| `aefa2f0` | Decode run length capped by the event window; a token wakes only its own stream |
| `8fa6708` | Event pump reads the outbox without the owner lock |
| `5697d5e` | With decode_step_pipeline, no boundary granted ahead while a prefill prepares (part of step 17) |
| `aefa2f0` | `capacity_recovery_all_decode`（基线即打开）：原版 MuxServe 在 KV 池满时的行为：暂停新 prefill，能放下的 decode 继续，抢占最年轻的空闲 decode。没有它开环 trace 无法跑完，所以属于基线而不是优化步骤。 |
