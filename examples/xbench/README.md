# 用 xbench 测原版 MuxServe

本目录放配置和启动脚本 `server.py`。原版代码只加了两处：HTTP 入口 `muxserve/muxsched/http_scheduler.py`，
以及 `muxserve/entrypoint.py` 里的 `--http-port` 和负载点参数 `--alpha`、`--rate-scale`、`--base-rate`。
调度逻辑没有改动。服务端由 xbench 自己启动和停止（CrossPool 的 hosted 模式），不用另开终端。

| 文件 | 内容 |
|---|---|
| `fig5-a0.7-s1.0.yaml` | 原版模型配置：三个 LLaMA 模型、TP4、MPS 100/40、`max_num_seqs` 16、KV 每卡 2 GiB（0.05061）。所有负载点共用，负载点由启动命令的 `--alpha`、`--rate-scale` 决定 |
| `benches.toml` | xbench 的 20 个 Figure 5 用例，和 baseline 仓库（`Muxserve-SGLang-Headwisecache` 分支）`benchmark/xbench/benches.toml` 逐字节相同；每个用例用本目录的 `server.py` 启动服务端 |
| `server.py` | 启动 / 停止服务端，xbench 调用，也能手动用（`uv run examples/xbench/server.py ...`），只用 Python 标准库 |
| `.env.example` | 本机设置的样例：原版环境和 CUDA 的路径。复制成 `.env` 再填（git 不收） |
| `CROSSPOOL-LICENSE` | `benches.toml` 写法的 MIT 授权 |

20 个点、用例前缀和请求怎么生成，见 baseline 那份 README。两边用同一份用例、同一个 xbench，
请求完全一样（随机种子都是 0）。

## 负载点参数

原版的 ADBS 启动时要知道各模型的到达率。用 `--alpha A --rate-scale S` 时，入口按论文的生成方式
（`benchmark/end_to_end/bench_end_to_end_muxserve.py` 的 `get_real_rate` × `gen_power_law_dis`）算出
`--base-rate × S / (排名 + 1)^A`，排名按 yaml 里 `models` 的顺序；把结果写进
`/tmp/mxo-$USER-load/workload-a<A>-s<S>-b<base>.json`，再按原来的方式交给原版（文件里只有到达率，没有请求）。
`--alpha` 和 `--workload-file` 只能二选一。alpha 0.7 / rate scale 1.0 时三个到达率是
2.0 / 1.2311 / 0.9269 req/s。用例的 `launch_settings` 把 alpha 和 rate scale 交给 `server.py`，
用例里 xbench 的到达率也按同一个公式算。

## 跑一个点

uv 只负责运行启动脚本 `server.py`；服务端本身还是在原版自己的 Python 3.9 / CUDA 11.8 环境里跑，
uv 不装、不改这个环境。

**第一次**（每台机器做一次）：

1. 把本目录的 `.env.example` 复制成 `.env`，填两项：
   `MUXSERVE_ENV` = 原版的 Python 环境（装了 MuxServe-vLLM 和本仓库，见根目录 README 的 Installation），
   `CUDA_HOME` = CUDA 11.8。
2. CrossPool 用 main `987b2d5` 或更新的版本。`XPOOL_CONFIG` 指向的配置里，`[vendor]` 的 `model_base_uri`
   设成模型目录（这台机器是 `/storage/crosspool-models`）：xbench 要从那里读分词器，生成随机问题。

**跑**：在 CrossPool 目录，一条命令跑一个用例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3     # xbench 从这几张卡里给服务端 4 张
uv run xbench run --catalog <原版仓库>/examples/xbench/benches.toml --case 2868bf58
uv run xbench report --list | tail -1     # 最新的运行编号
uv run xbench report <运行编号>            # 生成 report.md 和图
```

`--catalog` 换成 baseline 仓库的那份就是测 baseline。xbench 会自己：运行 `server.py requirements`
问要几张卡、几个端口（4 张卡，三个模型共用一个端口）；挑好端口，运行 `server.py start` 启动服务端；
等 `GET /v1/models` 列出三个模型；发请求、测量；测完给整个进程组发 SIGTERM，30 秒后还没退的发 SIGKILL。
`server.py` 收到 SIGTERM 后等服务端退出，再关掉 MPS（MPS 不在这个进程组里，xbench 管不到）。

服务端的输出在 xbench 运行目录的 `logs/server.log`，也在本仓库 `log/xbench/a<alpha>-s<scale>-<时间>/`。

**停止**：正常不用管。xbench 被中断、或者有进程、MPS 没停干净时，在本仓库根目录运行：

```bash
uv run examples/xbench/server.py stop
```

它先像 Ctrl-C 那样停服务端（等 30 秒，还没停再发 TERM、HUP，不发 kill -9），再关掉这次的 MPS。只停本仓库启动的进程。

**手动启动**（不测，只看服务端）：

```bash
uv run examples/xbench/server.py start --alpha 0.7 --rate-scale 2.0 --gpus 0,1,2,3
```

出现 `Server ready` 后，`http://127.0.0.1:18080` 上就能发 OpenAI 格式的请求（`GET /v1/models`、`POST /v1/completions`）。
`benches.toml` 的用例会自己启动服务端，跑用例前要先停掉手动起的这个（Ctrl-C 或 `stop`）。

### start 具体做了什么

和以前手动启动完全一样，只是不用自己敲：

1. 用 `.env` 里的环境：`PATH` 最前面是原版环境的 `bin`，去掉别的 CUDA（比如实验室的 CUDA 13.2），
   `CUDA_HOME`、`LD_LIBRARY_PATH` 指向 CUDA 11.8，`PYTHONNOUSERSITE=1`；
2. GPU 换成 UUID：xbench 给的已经是 UUID；手动时把 `--gpus` 的编号换成 UUID。
   **`CUDA_VISIBLE_DEVICES` 必须用 UUID，不能用编号**：私有 MPS 也是在这个变量下启动的，
   写编号会被当成 MPS 那几张卡里的相对编号，例如写 `2,3,4,5` 只剩两张卡可见，FlexStore 启动几秒就退出，其余进程一直空等；
3. 在 `/tmp/mxo-$USER-mps` 起私有 MPS，并确认环境里看得到 4 张卡；
4. 运行 `python -m muxserve.launch --nproc_per_node 4 --master-port 39335 examples/xbench/fig5-a0.7-s1.0.yaml --server-port 29702 --flexstore-port 33651 --mps-dir /tmp/mxo-$USER-mps --schedule-approach adbs --max-num-batched-tokens 2048 --http-port P --alpha A --rate-scale S`，
   `P` 是 xbench 挑的端口，手动时是 18080；
5. xbench 启动时，服务端留在 xbench 管的进程组里（xbench 要求监听端口的进程在这个组里），
   并把命令写进 xbench 运行目录的 `launch/commands.json` 留作记录。

每次的日志在 `log/xbench/a<alpha>-s<scale>-<时间>/`：`server.log`（服务端输出）、`proc/`（各进程日志）、
`command.txt`（实际用的环境变量、命令和代码版本）。手动启动加 `--dry-run` 只打印这些，不启动。

### 请求格式

模型名用 xbench 的写法：yaml 里模型路径的最后两段，例如 `huggyllama/llama-7b`。xbench 发 OpenAI Completions 请求：
`prompt` 是 token id 列表，`max_tokens` 是输出长度，`temperature` 0、`ignore_eos` true、流式。
每生成一个 token 回一帧，带累计的 token 数；最后一帧 `finish_reason` 是 `length`，再跟一帧只带 token 数的帧和 `[DONE]`。
请求进的还是原版那个等待队列。旧版 xbench 用的 `POST /models/<模型>/generate` 也还在。

同一时间只能跑一个服务端：原版和 baseline 用同样的内部端口和 MPS 目录。另一个还在跑时，`start` 会提示先 `stop`。
