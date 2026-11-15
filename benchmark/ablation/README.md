# 补丁消融实验

量化每个优化补丁对吞吐和延迟的增量贡献。做法是**累加**：第 N 步在第 N−1 步的基础上
多打开一个提交引入的开关，其余设置在所有步骤中完全相同；第 27 步就是最终配置
`configs/a100_open_loop.json`。

| 文件 | 内容 |
|---|---|
| `steps.json` | 步骤定义：每步对应的提交、开关、类型（性能 / 原版对齐）、动机、对模型适配的影响；基线里常开的改动 |
| `make_configs.py` | 由 `steps.json` 和最终配置生成 `configs/ablation/sNN-*.json` 与 `RESULTS.md`，结果逐字节可复现 |
| `RESULTS.md` | 结果表（实测后填写）、每个补丁的动机与适配影响、基线常开改动清单 |
| `../../configs/ablation/` | 28 份完整服务配置，s00（全关）到 s27（最终） |

## 实验条件

- 负载：`benchmark/xbench/benches.toml` 的 case `2bdea08f`
  （论文 Figure 5 点：ShareGPT，alpha 0.7，rate scale 1.0，60 s，257 个请求）。
- 所有步骤相同：三个模型、TP4、KV 池（原版公式 0.05061，即每卡 2 GiB）、
  每模型最多 16 个运行中请求（原版 `max_num_seqs` 16）、输出上限、超时、端口。
- 第 0 步：全部开关关闭，`rpc_persistent_connections` 也关闭（每次 RPC 重新连接认证）。
  它仍包含没有开关的常开修复，以及原版的 KV 满载恢复行为 `capacity_recovery_all_decode`
  （没有它开环负载跑不完）。这些在 `RESULTS.md` 末尾单独列出，不计入单步增量。
- 类型“原版对齐”的步骤（16、18、22、25、27）是把行为改回原版 MuxServe 的做法，
  不是新优化；它们的增量说明原版机制在 SGLang 上的作用。

## 怎么跑

前提和 `benchmark/xbench/README.md` 一样：已 `uv sync`，`configs/local.json` 已填好，
CrossPool checkout 已 `uv sync --group dev`。每一步：

服务端（MuxServe 仓库根目录）：

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
export MUXSERVE_LOCAL_CONFIG=$PWD/configs/local.json
uv run muxserve configs/ablation/s00-baseline.json
```

客户端（CrossPool checkout 根目录，服务 ready 之后）：

```bash
uv run xbench run --catalog /absolute/path/to/MuxServe/benchmark/xbench/benches.toml --case 2bdea08f
```

跑完 Ctrl-C 停服务，换下一份配置（`s01-rpc-persistent.json` ……`s27-adbs-step-history.json`）重复。
`uv run xbench report RUN_ID` 给出吞吐、TTFT、TPOT 和失败数，填进 `RESULTS.md`。

规则：

- 每步一次运行，服务每步重新启动；不开 TRACE / PROFILE。
- 第 0、27 步建议各跑 3 轮，给出噪声范围；相邻两步的差小于这个范围时，不能说该补丁有贡献。
- 早期步骤（尤其第 0–2 步）可能很慢，部分请求可能超过 300 s 截止而失败；
  失败数照实填写，不要只用成功请求算吞吐。
- 累加的增量包含与前面补丁的交互；某一步需要单独说明时，再从最终配置只关掉该步的开关测一次（去除对照）。

修改 `steps.json` 或最终配置后重新生成：

```bash
uv run python benchmark/ablation/make_configs.py
```
