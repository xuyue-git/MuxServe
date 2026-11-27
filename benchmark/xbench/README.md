# 用 xbench 测 MuxServe

跑一次实验要两个程序：**MuxServe 服务端**负责部署模型、处理请求；**xbench 客户端**负责按时间发请求、记录结果。
两个程序各读各的配置，所以这里有两种格式：MuxServe 读 yaml，xbench 读 toml（xbench 来自 CrossPool，toml 是它规定的写法）。
服务端由 xbench 自己启动和停止（CrossPool 的 hosted 模式），不用另开终端。

| 文件 | 谁读 | 管什么 |
|---|---|---|
| `fig5-a0.7-s1.0.yaml` | MuxServe 服务端 | 怎么部署：三个 LLaMA 模型、TP4、MPS 100/40、`max_num_seqs` 16、KV 每卡 2 GiB（0.05061）。20 个负载点共用，点由启动命令的 `--alpha`、`--rate-scale` 决定 |
| `benches.toml` | xbench 客户端 | 论文 Figure 5 的 20 个用例。每个用例写明：用本目录的 `server.py` 启动服务端、这个点的 alpha 和 rate scale、到达率、问题和输出长度怎么生成、贪心解码、最多 128 个同时请求、超时 330 秒 |
| `server.py` | xbench / 手动 | 启动、停止服务端，只用 Python 标准库 |
| `.env.example` | `server.py` | 本机设置的样例：SGLang 环境和 CUDA 的路径。复制成 `.env` 再填（git 不收） |
| `CROSSPOOL-LICENSE` | — | `benches.toml` 写法的 MIT 授权 |

比较原版和 baseline 时，两边用同一份 yaml、同一份 `benches.toml`（原版仓库 `examples/xbench/benches.toml`，逐字节相同）、
同一个 xbench，只有执行引擎不同。

## 20 个点

| alpha | rate scale | 请求总数（按 rate scale 依次） | 用例前缀（按 rate scale 依次） |
|---|---|---|---|
| 0.7 | 1.0 / 1.5 / 2.0 / 2.5 | 234 / 372 / 497 / 624 | `596bb910` / `c02fa7c4` / `2868bf58` / `e8508945` |
| 0.9 | 1.0 / 1.5 / 2.0 / 2.5 | 216 / 344 / 453 / 564 | `d9dc6559` / `97cb9b08` / `3c0ec869` / `265f3d0f` |
| 1.3 | 1.0 / 1.5 / 2.0 / 2.5 | 183 / 289 / 399 / 484 | `5f5d389e` / `00167f40` / `0f63505d` / `fb14e827` |
| 1.7 | 1.0 / 1.5 / 2.0 / 2.5 | 163 / 259 / 353 / 435 | `6b5b616f` / `ccaeeca5` / `12dfe781` / `586cf3a9` |
| 2.1 | 1.0 / 1.5 / 2.0 / 2.5 | 149 / 233 / 319 / 392 | `95bf0bd1` / `28586f45` / `29f90517` / `fbeaf556` |

请求由 xbench 现场生成，和 CrossPool 自己的用例一样，不存输入文件：

- **到达**：每个点 60 秒，泊松到达，到达率是 `2.0 × rate_scale / (排名 + 1)^alpha`（7B、13B、30B 排名 0、1、2），
  和交给服务端 ADBS 的到达率相同；
- **问题**：随机 token，长度按对数正态抽，中位数约 220 个 token；
- **输出**：长度按对数正态抽，中位数约 13 个 token、平均约 32 个，最多是问题长度的 2 倍；贪心解码，不在 EOS 停。

所有用例的随机种子都是 0，所以两个版本收到的请求完全一样；表里的请求数是 xbench 用这个种子生成的数目。

2026-11-25/26 的 20 个点用的是旧版 CrossPool（3c1f976）和旧用例（ShareGPT 请求，存在 `inputs/` 里），
旧用例和输入在 git 历史里（a9f06bd 及以前）。新版 xbench 读不了旧结果，旧结果的报告要用旧版 CrossPool 出。

## 负载点参数

ADBS 启动时要知道各模型的到达率。用 `--alpha A --rate-scale S` 时，入口按论文的生成方式
（`benchmark/end_to_end/bench_end_to_end_muxserve.py` 的 `get_real_rate` × `gen_power_law_dis`）算出
`--base-rate × S / (排名 + 1)^A`，排名按 yaml 里 `models` 的顺序；把结果写进
`/tmp/mxo-$USER-load/workload-a<A>-s<S>-b<base>.json` 交给调度器（文件里只有到达率，没有请求）。
`--alpha` 和 `--workload-file` 只能二选一。用例的 `launch_settings` 把 alpha 和 rate scale 交给 `server.py`。

## 跑一个点

**第一次**（每台机器做一次）：

1. 在本仓库根目录，把本目录的 `.env.example` 复制成 `.env`（git 不收），填两项：
   `MUXSERVE_ENV` = 装了 SGLang 分支的 Python 环境，`CUDA_HOME` = CUDA 13.2。
   如果这个环境是和别的仓库共用的、装的不是 SGLang 分支，再填 `SGLANG_FORK` = SGLang 分支的目录
   （xuyue-git/sglang-engine 的 muxserve-v0.5.20），`server.py` 会让服务端从它的 `python/` 导入 SGLang，
   共用的环境不用动。启动时日志里的 `sglang from` 会写出实际用的是哪一份。
2. 编译扩展：`uv run benchmark/xbench/server.py build`
3. CrossPool 用 main `987b2d5` 或更新的版本。`XPOOL_CONFIG` 指向的配置里，`[vendor]` 的 `model_base_uri`
   设成模型目录（这台机器是 `/storage/crosspool-models`）：xbench 要从那里读分词器，生成随机问题。

**跑**：在 CrossPool 目录，一条命令跑一个用例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3     # xbench 从这几张卡里给服务端 4 张
uv run xbench run --catalog <本仓库>/benchmark/xbench/benches.toml --case 2868bf58
uv run xbench report --list | tail -1     # 最新的运行编号
uv run xbench report <运行编号>            # 生成 report.md 和图
```

xbench 会自己：运行 `server.py requirements` 问要几张卡、几个端口（4 张卡，三个模型共用一个端口）；
挑好端口，运行 `server.py start` 启动服务端；等 `GET /v1/models` 列出三个模型；发请求、测量；
测完给整个进程组发 SIGTERM，30 秒后还没退的发 SIGKILL。`server.py` 收到 SIGTERM 后等服务端退出，再关掉 MPS
（MPS 不在这个进程组里，xbench 管不到）。

服务端的输出在 xbench 运行目录的 `logs/server.log`，也在本仓库 `log/xbench/a<alpha>-s<scale>-<时间>/`。

**停止**：正常不用管。xbench 被中断、或者有进程、MPS 没停干净时，在本仓库根目录运行：

```bash
uv run benchmark/xbench/server.py stop
```

它先像 Ctrl-C 那样停服务端（等 30 秒，还没停再发 TERM、HUP，不发 kill -9），再关掉这次的 MPS。只停本仓库启动的进程。

**手动启动**（不测，只看服务端）：

```bash
uv run benchmark/xbench/server.py start --alpha 0.7 --rate-scale 2.0 --gpus 0,1,2,3
```

出现 `Server ready` 后，`http://127.0.0.1:18080` 上就能发 OpenAI 格式的请求（`GET /v1/models`、`POST /v1/completions`）。
`benches.toml` 的用例会自己启动服务端，跑用例前要先停掉手动起的这个（Ctrl-C 或 `stop`）。

### start 具体做了什么

和以前手动启动完全一样，只是不用自己敲：

1. GPU 换成 UUID：xbench 给的已经是 UUID；手动时把 `--gpus` 的编号换成 UUID。私有 MPS 也在
   `CUDA_VISIBLE_DEVICES` 下启动，写编号会被当成 MPS 那几张卡里的相对编号；
2. 在 `/tmp/mxo-$USER-mps` 起私有 MPS，并确认环境里看得到 4 张卡；
3. 用 `.env` 里的环境运行
   `python -m muxserve.launch --nproc_per_node 4 --master-port 39335 benchmark/xbench/fig5-a0.7-s1.0.yaml --server-port 29702 --flexstore-port 33651 --mps-dir /tmp/mxo-$USER-mps --schedule-approach adbs --max-num-batched-tokens 2048 --http-port P --alpha A --rate-scale S`，
   `P` 是 xbench 挑的端口，手动时是 18080；
4. xbench 启动时，服务端留在 xbench 管的进程组里（xbench 要求监听端口的进程在这个组里），
   并把命令写进 xbench 运行目录的 `launch/commands.json` 留作记录。

每次的日志在 `log/xbench/a<alpha>-s<scale>-<时间>/`：`server.log`（服务端输出）、`proc/`（各进程日志）、
`command.txt`（实际用的环境变量、命令和代码版本）。手动启动加 `--dry-run` 只打印这些，不启动。

### 请求格式

HTTP 入口是 `muxserve/muxsched/http_scheduler.py`，原版和 baseline 是同一份。模型名用 xbench 的写法：
yaml 里模型路径的最后两段，例如 `huggyllama/llama-7b`。xbench 发 OpenAI Completions 请求：
`prompt` 是 token id 列表，`max_tokens` 是输出长度，`temperature` 0、`ignore_eos` true、流式。
每生成一个 token 回一帧，带累计的 token 数；最后一帧 `finish_reason` 是 `length`，再跟一帧只带 token 数的帧和 `[DONE]`。
调度不动：请求进的还是原来那个等待队列。旧版 xbench 用的 `POST /models/<模型>/generate` 也还在。

- 原版用同样的方式：原版仓库（`Muxserve-change` 分支）的 `examples/xbench/` 下有同一份 `benches.toml` 和它自己的 `server.py`，
  `--catalog` 换成那份就是测原版，见那边的 `README.md`。原版的 yaml 和本目录的 `fig5-a0.7-s1.0.yaml`
  逐字节相同（sha256 `7134a3df…`）；这份 yaml 里的注释也是原版那份的原文，所以写的是 `examples/xbench` 和旧用例。
- 同一时间只能跑一个服务端：原版和 baseline 用同样的内部端口和 MPS 目录。另一个还在跑时，`start` 会提示先 `stop`。
