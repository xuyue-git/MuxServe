# 用 xbench 测原版 MuxServe

本目录放配置和启动脚本 `server.py`。原版代码只加了两处：HTTP 入口 `muxserve/muxsched/http_scheduler.py`，
以及 `muxserve/entrypoint.py` 里的 `--http-port` 和负载点参数 `--alpha`、`--rate-scale`、`--base-rate`。
调度逻辑没有改动。

| 文件 | 内容 |
|---|---|
| `fig5-a0.7-s1.0.yaml` | 原版模型配置：三个 LLaMA 模型、TP4、MPS 100/40、`max_num_seqs` 16、KV 每卡 2 GiB（0.05061）。所有负载点共用，负载点由启动命令的 `--alpha`、`--rate-scale` 决定 |
| `server.py` | 启动 / 停止服务端（`uv run examples/xbench/server.py ...`），只用 Python 标准库 |
| `.env.example` | 本机设置的样例：原版环境和 CUDA 的路径。复制成 `.env` 再填（git 不收） |

请求由 xbench 发来：MuxServe-SGLang 仓库 `benchmark/xbench/benches.toml` 里的用例
（alpha 0.7 / 1.0 是 `2bdea08f`，其余 Figure 5 网格的点见那份 README）。两边用同一个用例、同一个 xbench 客户端。

## 负载点参数

原版的 ADBS 启动时要知道各模型的到达率。用 `--alpha A --rate-scale S` 时，入口按论文的生成方式
（`benchmark/end_to_end/bench_end_to_end_muxserve.py` 的 `get_real_rate` × `gen_power_law_dis`）算出
`--base-rate × S / (排名 + 1)^A`，排名按 yaml 里 `models` 的顺序；把结果写进
`/tmp/mxo-$USER-load/workload-a<A>-s<S>-b<base>.json`，再按原来的方式交给原版（文件里只有到达率，没有请求）。
`--alpha` 和 `--workload-file` 只能二选一。alpha 0.7 / rate scale 1.0 时三个到达率是
2.0 / 1.2311 / 0.9269 req/s，与之前 Figure 5 那次原版运行的完全相同。

## 跑一个点

用 uv 启动，和 CrossPool 一样。uv 只负责运行启动脚本 `server.py`；服务端本身还是在原版自己的
Python 3.9 / CUDA 11.8 环境里跑，uv 不装、不改这个环境。

**第一次**（每台机器做一次）：把本目录的 `.env.example` 复制成 `.env`，填两项：
`MUXSERVE_ENV` = 原版的 Python 环境（装了 MuxServe-vLLM 和本仓库，见根目录 README 的 Installation），
`CUDA_HOME` = CUDA 11.8。

**终端 1：服务端**，在本仓库根目录。`--alpha`、`--rate-scale` 选 20 个点中的哪一个
（`uv run examples/xbench/server.py points` 列出 20 个点和对应的 xbench 用例），`--gpus` 选哪四张卡：

```bash
uv run examples/xbench/server.py start --alpha 0.7 --rate-scale 2.0 --gpus 0,1,2,3
```

出现 `Server ready` 后，屏幕上会写出这个点对应的 xbench 命令。调度器空闲 240 秒会自己退出（原版行为），
所以就绪后要尽快开始测。

**终端 2：客户端**，在 CrossPool 目录：

```bash
uv run xbench run --case 401a4222
uv run xbench report --list | tail -1     # 最新的运行编号
uv run xbench report <运行编号>            # 生成 report.md 和图
```

**停止**：终端 1 按 Ctrl-C。终端关了或不在手边，就在本仓库根目录运行：

```bash
uv run examples/xbench/server.py stop
```

两种方式一样：先像 Ctrl-C 那样停服务端（等 30 秒，还没停再发 TERM、HUP，不发 kill -9），
再关掉这次的 MPS。只停本仓库启动的进程。

### start 具体做了什么

和以前手动启动完全一样，只是不用自己敲：

1. 用 `.env` 里的环境：`PATH` 最前面是原版环境的 `bin`，去掉别的 CUDA（比如实验室的 CUDA 13.2），
   `CUDA_HOME`、`LD_LIBRARY_PATH` 指向 CUDA 11.8，`PYTHONNOUSERSITE=1`；
2. 把 `--gpus` 的编号换成 GPU UUID。**`CUDA_VISIBLE_DEVICES` 必须用 UUID，不能用编号**：
   私有 MPS 也是在这个变量下启动的，写编号会被当成 MPS 那几张卡里的相对编号，
   例如写 `2,3,4,5` 只剩两张卡可见，FlexStore 启动几秒就退出，其余进程一直空等；
3. 在 `/tmp/mxo-$USER-mps` 起私有 MPS，并确认环境里看得到 4 张卡；
4. 运行 `python -m muxserve.launch --nproc_per_node 4 --master-port 39335 examples/xbench/fig5-a0.7-s1.0.yaml --server-port 29702 --flexstore-port 33651 --mps-dir /tmp/mxo-$USER-mps --schedule-approach adbs --max-num-batched-tokens 2048 --http-port 18080 --alpha A --rate-scale S`。

每次的日志在 `log/xbench/a<alpha>-s<scale>-<时间>/`：`server.log`（服务端输出）、`proc/`（各进程日志）、
`command.txt`（实际用的环境变量、命令和代码版本）。加 `--dry-run` 只打印这些，不启动。

同一时间只能跑一个服务端：原版和 baseline 用同样的端口和 MPS 目录。另一个还在跑时，`start` 会提示先 `stop`。
