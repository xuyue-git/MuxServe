# MuxServe 的 xbench 基准配置（CrossPool 写法）

所有实验条件写在 `benches.toml` 里，服务端和客户端各用一条原生命令启动，
中间没有包装脚本。`benches.toml` 共 61 个可运行用例：

- 前 3 个是**默认三模型部署**（LLaMA-7B/13B/30B）的用例，保留原 case ID；
- 后 58 个是按 CrossPool catalogue 逐项迁移的双模型、单模型矩阵。

CrossPool 最后 3 个 ATN/FFN/Lane 拓扑用例在 `pending-topologies.toml`，明确标记为不可运行。

## 文件

| 文件 | 内容 |
|---|---|
| `benches.toml` | 61 个用例，每个独立声明：到达过程、到达率、时长、prompt、输出长度、采样、预热、并发上限、端点 |
| `pending-topologies.toml` | 3 个待实现拓扑用例（不可运行） |
| `../../configs/a100_xbench_poisson.toml`、`../../configs/a100_open_loop.toml` | 三模型服务配置 |
| `../../configs/xbench/*.toml` | 8 份双模型 / 单模型服务配置 |
| `../../configs/muxserve.example.toml`、`../../.env.example` | 本机设置模板（锁目录、模型路径、GPU），复制为 `configs/muxserve.local.toml`、`.env` |
| `inputs/`、`make_inputs.py` | 三模型用例的冻结输入和离线生成器 |
| `suites/serving/multi_model.py` | CrossPool 的 serving suite，未修改（MIT） |
| `CROSSPOOL-LICENSE` | 借用 catalogue 结构和 suite 的 MIT 授权 |

catalogue 写法参考 CrossPool `0d3267edecfbe67459c1e9edc62ee73ff1f1a63f`，
它的声明与严格 case schema 和客户端锁定版本 `3c1f976cdd1069756314ecb7142581fad2367272` 相同。
迁移后的用例换了模型，所以用新的 UUIDv4；每条声明前的注释写有 CrossPool 的原 case ID。

## 三模型用例（默认部署）

| case 前缀 | 负载 | 请求数 | 服务配置 |
|---|---|---|---|
| `55af3ef0` | 固定 512 输入 / 128 输出，xbench Poisson，每模型 0.5 req/s，600 s | 约 900 | `configs/a100_xbench_poisson.toml` |
| `279fe35b` | 同样的长度和到达率，原版 MuxServe `PoissonProcess` 生成的冻结 trace | 903（298 / 290 / 315） | `configs/a100_xbench_poisson.toml` |
| `2bdea08f` | 论文 Figure 5 点：ShareGPT，alpha 0.7，rate scale 1.0，60 s | 257（121 / 76 / 60） | `configs/a100_open_loop.toml` |

alpha 是幂律热度指数：第 i 个模型（按热度从 0 编号）到达率为
`2.0 × rate_scale / (i + 1)^alpha`，即 2.0 / 1.2311 / 0.9269 req/s。
这三个用例都是贪心解码、每个 token 一个 SSE 事件、忽略 EOS（输出恰好是请求长度），
全局最多 128 个活动 HTTP 请求，客户端超时 330 s（服务端截止 300 s）。
`55af3ef0` 和 `279fe35b` 每个模型预热 2 个请求，`2bdea08f` 不预热（与原版 Figure 5 回放一致）。

## CrossPool 矩阵（双模型 / 单模型）

| 原位置 | 数量 | 条件 |
|---|---:|---|
| 01–18 | 18 | 三个模型配对，各扫 0.1、0.2、0.5、1.0、2.0、4.0 req/s/model |
| 19–54 | 36 | 每个配对的两个模型分别单独运行，同样六档 |
| 55–56 | 2 | 7B/30B 流量偏斜：0.8/0.2、0.2/0.8 req/s |
| 57 | 1 | 13B/30B 长输入：输入 median_fraction 0.25，输出 0.03125 |
| 58 | 1 | 13B/30B 长输出：输入 median_fraction 0.125，输出 0.0625 |

| CrossPool 原模型 | MuxServe 模型 |
|---|---|
| Qwen2.5-0.5B、Qwen3-0.6B | llama-7b、llama-13b |
| Qwen3-14B、DeepSeek-V2-Lite-Chat | llama-7b、llama-30b |
| Qwen3-30B-A3B、GLM-4.7-Flash | llama-13b、llama-30b |

到达窗口 60 s、seed 0、输入/输出联合 lognormal 策略都与 CrossPool 相同：

```toml
output_tokens = { kind = "lognormal", median_fraction = 0.03125, sigma = 1.0, max_output_input_ratio = 2.0 }
prompts = { kind = "random", input_tokens = { kind = "lognormal", median_fraction = 0.125, sigma = 1.0 } }
```

客户端默认值沿用 CrossPool schema：`max_inflight` 128、每目标预热 1 个、统计桶 1 s、
不设客户端 HTTP 超时。随机输入最长可到 2047 token（上下文 2048 减最少 1 个输出），
所以这 8 份服务配置把 `max_input_tokens` 设为 2047，服务端不会拒绝合法请求。

| 原位置 | 服务配置 | 示例 case 前缀 |
|---|---|---|
| 01–06 | `configs/xbench/pair-7b-13b.toml` | `e81c17dd` |
| 07–12 | `configs/xbench/pair-7b-30b.toml` | `ec7e3f29` |
| 13–18、57–58 | `configs/xbench/pair-13b-30b.toml` | `3c154150`、`21db7577`、`d6a32a09` |
| 19–24、31–36 | `configs/xbench/isolated-7b.toml` | `7609e0a1`、`8c1cf4cb` |
| 25–30、43–48 | `configs/xbench/isolated-13b.toml` | `56d67c30`、`aac946e0` |
| 37–42、49–54 | `configs/xbench/isolated-30b.toml` | `841c8993`、`6af191c5` |
| 55 | `configs/xbench/pair-7b-30b-asymmetric-08-02.toml` | `39b65602` |
| 56 | `configs/xbench/pair-7b-30b-asymmetric-02-08.toml` | `d230a248` |

## 所有服务配置共同的条件

- FP16、TP4，四张 A100 40 GB；端点为原生 SGLang `/generate`（SSE），
  `http://127.0.0.1:18080/models/llama7b|llama13b|llama30b`。
- KV 池按原版 MuxServe 的公式定容：`kv_memory_fraction` 0.05061 ×
  torch 报告的整卡显存（`muxserve/muxsched/resource.py` 的 `num_gpu_blocks`）。
  在这批 A100（42,436,526,080 字节）上正好是每卡 2 GiB；启动时打印 `KV_POOL` 行，
  记录实际字节数。原版基线要用同一个 `gpu_memory_utilization=0.05061`。
- 机器相关的设置（个人锁目录、模型路径）不写在这些文件里，放在本机设置文件中，见下文。

## 怎么跑

做法和 CrossPool 的 quick-start 一样：仓库 checkout，机器相关的设置放在被 git 忽略的
本地文件里，每个终端 `export UV_ENV_FILE="$PWD/.env"`，然后 `uv sync` / `uv run`。

**0. 代码（只做一次）。** 三个仓库并排放在同一个目录（下面用 `~/xb`），固定到要测的 commit：

```bash
mkdir -p ~/xb && cd ~/xb
git clone -b Muxserve-SGLang https://github.com/xuyue-git/MuxServe.git
git clone https://github.com/xuyue-git/Muxserve-SGLang-Engine.git MuxServe-SGLang-Engine
git clone https://github.com/Coekjan/CrossPool.git
git -C CrossPool checkout 3c1f976cdd1069756314ecb7142581fad2367272
```

引擎目录名必须是 `MuxServe-SGLang-Engine`，且和 `MuxServe` 同级（`pyproject.toml` 按相对路径引用）。
需要 uv 0.12.17 以上和 CUDA Toolkit 13.2（`nvcc`），两个项目都用 uv 管理的 Python 3.12。

**1. MuxServe 服务端（只做一次）**，在 `~/xb/MuxServe`：

```bash
cp .env.example .env
cp configs/muxserve.example.toml configs/muxserve.local.toml
uv sync
```

在 `configs/muxserve.local.toml` 里填自己的绝对锁目录 `asset_lock_dir`，模型路径按本机修改；
这两个文件被 git 忽略。用哪四张卡写在启动命令里（`CUDA_VISIBLE_DEVICES=...`），不写进 `.env`。
本地设置文件只接受锁目录、运行目录和模型路径，写别的字段会被拒绝，所以改不了实验条件。
`uv run muxserve config dump CONFIG` 打印加上本地设置和全部默认值之后实际生效的配置
（对应 CrossPool 的 `uv run xpool config dump`）。

**2. CrossPool 客户端（只做一次）**，在 `~/xb/CrossPool`，照它 quick-start 的做法：

```bash
cp .env.example .env
cp configs/xpool.example.toml configs/xpool.local.toml
cp configs/xkit.example.toml configs/xkit.local.toml
uv sync --group dev
```

再把 `configs/xkit.local.toml` 里 `[xbench]` 的 `catalog` 改成本项目的目录（路径相对于这个文件）：

```toml
[xbench]
catalog = "../../MuxServe/benchmark/xbench/benches.toml"
```

这样客户端命令就和 CrossPool 自己的一样，不用每次写 `--catalog`。

**3. 每个 case：两个终端。**

终端 1，`~/xb/MuxServe`，启动 case 对应的服务配置：

```bash
export UV_ENV_FILE="$PWD/.env"
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run muxserve serve configs/a100_xbench_poisson.toml
```

看到 ready 后，`curl -s http://127.0.0.1:18080/health` 应返回正常。Ctrl-C 有序关停（排空后关 MPS）。

终端 2，`~/xb/CrossPool`：

```bash
export UV_ENV_FILE="$PWD/.env"
uv run xbench list
uv run xbench run --case 55af3ef0
uv run xbench report RUN_ID
```

规则：

- **一条命令只跑一个 case。** client 用例不占设备租约，同一条命令里写多个 `--case`
  会同时打到同一个服务上，负载叠加。
- 每个 case 用它对应的服务配置；换配置前先 Ctrl-C 停掉旧服务。不要用 `--all`。
- client 模式不会自动重启服务。要比较的多轮之间重启服务，避免 ADBS 历史、缓存、CUDA Graph 带到下一轮。
- 性能实验不开 `MXS_TRACE_DIR` / `PROFILE`。
- xbench 留在 CrossPool 自己的环境里，不装进 MuxServe 的环境，原因见
  `docs/crosspool-alignment.md`。

## 输入文件

```bash
uv run python benchmark/xbench/make_inputs.py --sharegpt-bundle /path/to/open-a07s1.json
```

固定 prompt 来自 `benchmark/paired/frozen-synthetic-512-128.toml`；固定 trace 来自原版
`PoissonProcess`，7B/13B/30B 分别用 seed 0/1/2；ShareGPT 输入来自冻结 bundle
`959024cce558`（source trace `d1d735cc3909`，原版 `sample_requests`，llama-7b tokenizer，seed 42）。
重新生成的文件逐字节相同。

## 与其他基线对齐

- 双方都用这份 `benches.toml` 的同一个 case，并保存 xbench run 目录里的 prepared workload
  （实际 prompt、到达时间、输出长度），而不只记录 seed：换模型会改变随机数的分域。
- 原版 MuxServe（vLLM）没有 HTTP 入口，不能直接用 client 用例测；它在自己的仓库和环境里
  回放同一份冻结输入（`inputs/`）。
- CrossPool 自己的用例是 owned 模式（xbench 负责启停服务，每次尝试都是新服务）。
  本项目是 client 模式，服务由上面的命令启动；对比时要写明这一点。

## 待实现的三个拓扑用例

| 原位置 | case 前缀 | 保留的维度 |
|---|---|---|
| 59 | `e7b57cc7` | 7B/13B：1 Lane 对 2 Lanes，4 req/s/model |
| 60 | `7aa80710` | 7B/13B：Attention TP2/DP1、两张 Attention 卡加一张 FFN 卡 |
| 61 | `4c4dbb63` | 7B/30B：同一双卡 FFN fleet 上 FFN TP1 对 TP2 |

MuxServe 没有 ATN/FFN 分离拓扑，这三项保留原声明结构，标记 `PENDING / NOT RUNNABLE`，
引用的模块和部署路径不存在，不能启动。
