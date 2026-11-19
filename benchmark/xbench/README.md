# MuxServe 的 xbench 基准配置（CrossPool 写法）

所有实验条件写在 `benches.toml` 里，服务端和客户端各用一条原生命令启动，
中间没有包装脚本。`benches.toml` 共 22 个用例，全部跑在默认的三模型部署上
（LLaMA-7B/13B/30B，四张 A100 40 GB）：

- 前 2 个是固定 512 输入 / 128 输出的负载，保留原 case ID；
- 后 20 个是论文 Figure 5 的网格。

## 文件

| 文件 | 内容 |
|---|---|
| `benches.toml` | 22 个用例，每个独立声明：到达过程、到达率、时长、prompt、输出长度、采样、预热、并发上限、端点 |
| `../../configs/a100_xbench_poisson.toml`、`../../configs/a100_open_loop.toml` | 三模型服务配置（前者给固定 512/128 用例，后者给 Figure 5 用例） |
| `../../configs/muxserve.example.toml`、`../../.env.example` | 本机设置模板（锁目录、模型路径、GPU），复制为 `configs/muxserve.local.toml`、`.env` |
| `inputs/fixed-512/` | 固定 512/128 用例的冻结 prompt 和原版生成器的到达 trace |
| `inputs/fig5-a<alpha>-s<scale>/`、`make_inputs.py` | Figure 5 每个点的冻结输入（`trace.jsonl` 和三个 `prompts-<model>.jsonl`）及离线生成器 |
| `suites/serving/multi_model.py` | CrossPool 的 serving suite，未修改（MIT） |
| `CROSSPOOL-LICENSE` | 借用 catalogue 结构和 suite 的 MIT 授权 |

catalogue 写法参考 CrossPool `0d3267edecfbe67459c1e9edc62ee73ff1f1a63f`，
它的声明与严格 case schema 和客户端锁定版本 `3c1f976cdd1069756314ecb7142581fad2367272` 相同。

## 固定 512/128 用例

| case 前缀 | 负载 | 请求数 | 服务配置 |
|---|---|---|---|
| `55af3ef0` | 固定 512 输入 / 128 输出，xbench Poisson，每模型 0.5 req/s，600 s | 约 900 | `configs/a100_xbench_poisson.toml` |
| `279fe35b` | 同样的长度和到达率，原版 MuxServe `PoissonProcess` 生成的冻结 trace | 903（298 / 290 / 315） | `configs/a100_xbench_poisson.toml` |

两个用例都是贪心解码、忽略 EOS（每个请求正好输出 128 个 token），每个模型预热 2 个请求，
全局最多 128 个活动 HTTP 请求，客户端超时 330 s。固定 prompt 来自
`inputs/fixed-512/frozen-synthetic-512-128.json`；`279fe35b` 的 trace 来自原版 `PoissonProcess`，
7B/13B/30B 分别用 seed 0/1/2。

## 论文 Figure 5 网格

论文 Figure 5 扫 alpha ∈ {0.7, 0.9, 1.3, 1.7, 2.1} × rate scale ∈ {1.0, 1.5, 2.0, 2.5}，共 20 个点。
这 20 条 trace 在 2026-11-05 已按论文的生成方式做好（ShareGPT，seed 42，每点 60 s，
基础速率 2.0 req/s，原版 MuxServe 的 `workload.json` 格式）。

**输入和输出长度。** 每个请求的输入是 `prompts-<model>.jsonl` 里的 `input_ids`（已分词），
输出长度是 `trace.jsonl` 里的 `max_new_tokens`。ShareGPT 请求按原版 `sample_requests` 过滤：
输入和输出都至少 4 个 token，输入不超过 1024，输入加输出不超过 2048。

**到达。** alpha 是幂律热度指数：第 i 个模型（按热度从 0 编号，即 7B、13B、30B）到达率为
`2.0 × rate_scale / (i + 1)^alpha`，各模型独立泊松到达；客户端按 `trace.jsonl` 里每个请求的
`arrival_seconds` 开环发送，不等前一个请求完成。

所有用例都是贪心解码、每个 token 一个 SSE 事件、忽略 EOS（输出恰好是请求长度），
全局最多 128 个活动 HTTP 请求，客户端超时 330 s（服务端截止 300 s），不预热（与原版 Figure 5 回放一致）。

**20 个点和对应用例**（请求数为 7B / 13B / 30B；服务端都用 `configs/a100_open_loop.toml`，
加表里的 `--alpha`、`--rate-scale`）：

| alpha | rate scale | 请求数 | case 前缀 |
|---|---|---|---|
| 0.7 | 1.0 | 257（121 / 76 / 60） | `2bdea08f` |
| 0.7 | 1.5 | 382（184 / 104 / 94） | `e9867899` |
| 0.7 | 2.0 | 499（229 / 146 / 124） | `401a4222` |
| 0.7 | 2.5 | 618（282 / 180 / 156） | `7c29c076` |
| 0.9 | 1.0 | 232（130 / 52 / 50） | `5883efb6` |
| 0.9 | 1.5 | 359（195 / 88 / 76） | `993fb53d` |
| 0.9 | 2.0 | 487（261 / 129 / 97） | `10f8e6ce` |
| 0.9 | 2.5 | 604（324 / 155 / 125） | `ed945f2f` |
| 1.3 | 1.0 | 191（117 / 48 / 26） | `a8559a20` |
| 1.3 | 1.5 | 282（173 / 70 / 39） | `d0d084ea` |
| 1.3 | 2.0 | 390（246 / 89 / 55） | `d20de0c3` |
| 1.3 | 2.5 | 491（309 / 114 / 68） | `d5c232a4` |
| 1.7 | 1.0 | 174（116 / 37 / 21） | `1a5a552f` |
| 1.7 | 1.5 | 263（179 / 52 / 32） | `f55bb9a1` |
| 1.7 | 2.0 | 357（240 / 77 / 40） | `35866028` |
| 1.7 | 2.5 | 442（296 / 97 / 49） | `bde94b52` |
| 2.1 | 1.0 | 161（122 / 30 / 9） | `115748fe` |
| 2.1 | 1.5 | 247（187 / 42 / 18） | `ec9f5989` |
| 2.1 | 2.0 | 314（238 / 53 / 23） | `b85f7ab5` |
| 2.1 | 2.5 | 386（293 / 68 / 25） | `3c884f3d` |

每条 trace 里记录的到达率都等于 `2.0 × rate_scale / (排名 + 1)^alpha`，所以服务端用这两个参数算出的
到达率就是生成 trace 时用的那组。

**重新生成用例（一般不需要）。** 把 20 个点的文件夹（`alpha-*-scale-*/workload.json`，服务器上在
`muxserve-legacy-a100-*/paper-fig5-inputs-20261105-ab7ce5115516/`）放在一个目录下：

```bash
python benchmark/xbench/make_inputs.py --fig5 DIR
```

它写出 `inputs/fig5-a<alpha>-s<scale>/`，并在 `benches.toml` 末尾的“Figure 5 grid”段里为另外 19 个点
各写一个用例（ID 由 alpha 和 rate scale 决定，重跑不变）。alpha 0.7 / 1.0 这一点必须逐字节重现
现有输入，否则报错，因为 `2bdea08f` 的结果就是在它上面测的。打印的表格列出每个点的请求数和 case 前缀；
`uv run xbench list` 里每个用例的说明也写着它的 alpha 和 rate scale。

**跑某一点。** 服务端用同一份配置，把这一点的 alpha 和 rate scale 写在启动命令里（ADBS 需要各模型到达率，
按 `2.0 × rate_scale / (排名 + 1)^alpha` 算，排名按配置里 `[[models]]` 的顺序；不写这两个参数就用配置里的
`policy_arrival_rates`）：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 uv run muxserve serve configs/a100_open_loop.toml --alpha 2.1 --rate-scale 2.0
```

客户端用这一点的用例：`uv run xbench run --case <case 前缀>`。原版 MuxServe 的启动命令同样加
`--alpha 2.1 --rate-scale 2.0`（见原版仓库 `Muxserve-change` 分支的 `examples/xbench/README.md`）。
`uv run muxserve config dump CONFIG --alpha 2.1 --rate-scale 2.0` 可以先看实际生效的到达率。

## 服务配置的条件

- FP16、TP4，四张 A100 40 GB；端点为原生 SGLang `/generate`（SSE），
  `http://127.0.0.1:18080/models/llama7b|llama13b|llama30b`。
- KV 池按原版 MuxServe 的公式定容：`kv_memory_fraction` 0.05061 ×
  torch 报告的整卡显存（`muxserve/muxsched/resource.py` 的 `num_gpu_blocks`）。
  在这批 A100（42,436,526,080 字节）上正好是每卡 2 GiB；启动时打印 `KV_POOL` 行，
  记录实际字节数。原版基线要用同一个 `gpu_memory_utilization=0.05061`。
- 机器相关的设置（个人锁目录、模型路径）不写在配置文件里，放在本机设置文件中，见下文。

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
服务器连不上 GitHub 时，可以在本机 clone 后用 `git bundle` 传过去，再从 bundle clone，commit 不变。

编译时可能遇到的两种情况（都只是编译选项，不改代码）：

- 机器上没有 Rust（`cargo`）：MuxServe 这边 `uv sync` 前加 `SGLANG_BUILD_RUST_EXTS=none`，跳过 SGLang 的
  Rust 扩展（gRPC、多模态、Rust HTTP 服务、Rust 前缀树）。这些在本项目里都不用：启动器固定
  `SGLANG_RUST_SERVER=0`，前缀树默认用 Python 实现。之后在 MuxServe 目录用 `uv` 时也带上这个变量，
  否则 uv 重新编译引擎时会再报错。
- CrossPool 编译下载 googletest 失败（连不上 GitHub）：`uv sync --group dev` 前加
  `SKBUILD_CMAKE_DEFINE="XPOOL_BUILD_CEXT_TESTS=OFF"`，只是不编 CrossPool 自己的 C++ 单元测试，xbench 不受影响。

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

终端 1，`~/xb/MuxServe`，用这一点的 alpha 和 rate scale 启动服务（下面是 `2bdea08f` 那一点）：

```bash
export UV_ENV_FILE="$PWD/.env"
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run muxserve serve configs/a100_open_loop.toml --alpha 0.7 --rate-scale 1.0
```

看到最后一行 `Mode: HTTP/event/ADBS serving` 后，`curl -s http://127.0.0.1:18080/health` 返回
`"ready":true`。

**停止：在另一个终端只给启动器发 SIGINT，不要在终端 1 按 Ctrl-C。**

```bash
pkill -INT -u $USER -f '\.venv/bin/muxserve serve configs/a100_open_loop\.toml'
```

启动器收到后排空请求、关闭各进程和它自己的 MPS，终端 1 回到提示符。在终端 1 按 Ctrl-C 时，
SIGINT 会发给整个前台进程组，网关和 rank 子进程跟着退出，启动器就把这当成故障，进入
`PROTECTED_FAILURE` 并一直保留 GPU 上的进程（为了保留现场）。遇到这种情况，用运行目录里的
`status.json`（`/tmp/mxs-<uid>-*/status.json`）找到启动器的进程组，对这个组发 TERM 再发 HUP，
然后对它的 `mps/` 管道发 `quit`；不要用 `kill -9`。

终端 2，`~/xb/CrossPool`：

```bash
export UV_ENV_FILE="$PWD/.env"
uv run xbench list
uv run xbench run --case 2bdea08f
uv run xbench report RUN_ID
```

规则：

- **一条命令只跑一个 case。** client 用例不占设备租约，同一条命令里写多个 `--case`
  会同时打到同一个服务上，负载叠加。
- 每个 case 用它对应的服务配置（固定 512/128 用 `configs/a100_xbench_poisson.toml`；Figure 5
  用 `configs/a100_open_loop.toml`，还要加这一点的 `--alpha`、`--rate-scale`）；换 case 前先按上面的方法
  停掉旧服务。不要用 `--all`。
- client 模式不会自动重启服务。要比较的多轮之间重启服务，避免 ADBS 历史、缓存、CUDA Graph 带到下一轮。
- 性能实验不开 `MXS_TRACE_DIR` / `PROFILE`。
- xbench 留在 CrossPool 自己的环境里，不装进 MuxServe 的环境，原因见
  `docs/crosspool-alignment.md`。

## 输入文件

```bash
uv run python benchmark/xbench/make_inputs.py --sharegpt-bundle /path/to/open-a07s1.json
uv run python benchmark/xbench/make_inputs.py --fig5 DIR
```

每次运行都会写出 `inputs/fixed-512/` 下固定 512/128 用例的 prompt 和 trace；
`--sharegpt-bundle` 写出 alpha 0.7 / 1.0 那一点，来自冻结 bundle `959024cce558`
（source trace `d1d735cc3909`，原版 `sample_requests`，llama-7b tokenizer，seed 42）；
`--fig5` 写出全部 20 个点（见上文）。重新生成的文件逐字节相同。

## 与原版对齐

- 两个版本都用这份 `benches.toml` 的同一个 case，并保存 xbench run 目录里的 prepared workload
  （实际 prompt、到达时间、输出长度），而不只记录 seed。
- 原版 MuxServe 在自己仓库的 `Muxserve-change` 分支加了 HTTP 入口（不改调度和执行），
  在自己的环境里启动，用同一个 client 用例测；它的启动命令同样加 `--alpha`、`--rate-scale`。
- CrossPool 自己的用例是 owned 模式（xbench 负责启停服务，每次尝试都是新服务）。
  本项目是 client 模式，服务由上面的命令启动；对比时要写明这一点。
