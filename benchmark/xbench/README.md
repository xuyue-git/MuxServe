# 用 xbench 测 MuxServe

跑一次实验要两个程序：**MuxServe 服务端**负责部署模型、处理请求；**xbench 客户端**负责按时间发请求、记录结果。
两个程序各读各的配置，所以这里有两种格式：MuxServe 读 yaml，xbench 读 toml（xbench 来自 CrossPool，toml 是它规定的写法）。

| 文件 | 谁读 | 管什么 |
|---|---|---|
| `fig5-a0.7-s1.0.yaml` | MuxServe 服务端 | 怎么部署：三个 LLaMA 模型、TP4、MPS 100/40、`max_num_seqs` 16、KV 每卡 2 GiB（0.05061）。20 个负载点共用，点由启动命令的 `--alpha`、`--rate-scale` 决定 |
| `benches.toml` | xbench 客户端 | 发什么请求：论文 Figure 5 的 20 个用例。每个用例写明到达时间文件、prompt 文件、贪心解码、输出到规定长度为止、最多 128 个同时请求、超时 330 秒、发到 `127.0.0.1:18080` |
| `inputs/fig5-a<alpha>-s<scale>/` | xbench 客户端 | 每个点冻结好的请求：`trace.jsonl`（每个请求的到达时间和输出长度）、`prompts-<模型>.jsonl`（已分词的 prompt） |
| `suites/serving/multi_model.py` | xbench 客户端 | CrossPool 的 serving suite，未修改 |
| `CROSSPOOL-LICENSE` | — | `benches.toml` 写法和 suite 的 MIT 授权 |

比较原版和 baseline 时，两边用同一份 yaml、同样的启动参数、同一个用例，只有执行引擎不同。

## 20 个点

| alpha | rate scale | 请求总数（按 rate scale 依次） | 用例前缀（按 rate scale 依次） |
|---|---|---|---|
| 0.7 | 1.0 / 1.5 / 2.0 / 2.5 | 257 / 382 / 499 / 618 | `2bdea08f` / `e9867899` / `401a4222` / `7c29c076` |
| 0.9 | 1.0 / 1.5 / 2.0 / 2.5 | 232 / 359 / 487 / 604 | `5883efb6` / `993fb53d` / `10f8e6ce` / `ed945f2f` |
| 1.3 | 1.0 / 1.5 / 2.0 / 2.5 | 191 / 282 / 390 / 491 | `a8559a20` / `d0d084ea` / `d20de0c3` / `d5c232a4` |
| 1.7 | 1.0 / 1.5 / 2.0 / 2.5 | 174 / 263 / 357 / 442 | `1a5a552f` / `f55bb9a1` / `35866028` / `bde94b52` |
| 2.1 | 1.0 / 1.5 / 2.0 / 2.5 | 161 / 247 / 314 / 386 | `115748fe` / `ec9f5989` / `b85f7ab5` / `3c884f3d` |

每个点 60 秒，ShareGPT 请求，到达率是 `2.0 × rate_scale / (排名 + 1)^alpha`（7B、13B、30B 排名 0、1、2）。

## 负载点参数

ADBS 启动时要知道各模型的到达率。用 `--alpha A --rate-scale S` 时，入口按论文的生成方式
（`benchmark/end_to_end/bench_end_to_end_muxserve.py` 的 `get_real_rate` × `gen_power_law_dis`）算出
`--base-rate × S / (排名 + 1)^A`，排名按 yaml 里 `models` 的顺序；把结果写进
`/tmp/mxo-$USER-load/workload-a<A>-s<S>-b<base>.json` 交给调度器（文件里只有到达率，没有请求）。
`--alpha` 和 `--workload-file` 只能二选一。

## 跑一个点

用 uv 启动，和 CrossPool 一样。uv 只负责运行启动脚本 `server.py`（只用 Python 标准库）；
服务端本身还是在 `.env` 里写的那个 SGLang 环境里跑，uv 不装、不改这个环境。

**第一次**（每台机器做一次），在本仓库根目录：

1. 把本目录的 `.env.example` 复制成 `.env`（git 不收），填两项：
   `MUXSERVE_ENV` = 装了 SGLang 分支的 Python 环境，`CUDA_HOME` = CUDA 13.2。
2. 编译扩展：`uv run benchmark/xbench/server.py build`

**终端 1：服务端**，在本仓库根目录。`--alpha`、`--rate-scale` 选 20 个点中的哪一个（见上表，
或 `uv run benchmark/xbench/server.py points`），`--gpus` 选哪四张卡：

```bash
uv run benchmark/xbench/server.py start --alpha 0.7 --rate-scale 2.0 --gpus 0,1,2,3
```

出现 `Server ready` 后，屏幕上会写出这个点对应的 xbench 命令。调度器空闲 240 秒会自己退出（原版行为），
所以就绪后要尽快开始测。

**终端 2：客户端**，在 CrossPool 目录：

```bash
uv run xbench run --case 401a4222
uv run xbench report --list | tail -1     # 最新的运行编号
uv run xbench report <运行编号>            # 生成 report.md 和图
```

CrossPool 的 `configs/xkit.local.toml` 里 `[xbench]` 的 `catalog` 指向要用的 `benches.toml`。
2026-11-25/26 那次 20 个点的实验，catalog 指的是 `Muxserve-SGLang` 分支 `benchmark/xbench/benches.toml`，
其中这 20 个用例和输入与本目录逐字节相同。一条命令只跑一个用例。

**停止**：终端 1 按 Ctrl-C。终端关了或不在手边，就在本仓库根目录运行：

```bash
uv run benchmark/xbench/server.py stop
```

两种方式一样：先像 Ctrl-C 那样停服务端（等 30 秒，还没停再发 TERM、HUP，不发 kill -9），
再关掉这次的 MPS。只停本仓库启动的进程。

### start 具体做了什么

和以前手动启动完全一样，只是不用自己敲：

1. 把 `--gpus` 的编号换成 GPU UUID：私有 MPS 也在 `CUDA_VISIBLE_DEVICES` 下启动，写编号会被当成
   MPS 那几张卡里的相对编号；
2. 在 `/tmp/mxo-$USER-mps` 起私有 MPS，并确认环境里看得到 4 张卡；
3. 用 `.env` 里的环境运行
   `python -m muxserve.launch --nproc_per_node 4 --master-port 39335 benchmark/xbench/fig5-a0.7-s1.0.yaml --server-port 29702 --flexstore-port 33651 --mps-dir /tmp/mxo-$USER-mps --schedule-approach adbs --max-num-batched-tokens 2048 --http-port 18080 --alpha A --rate-scale S`。

每次的日志在 `log/xbench/a<alpha>-s<scale>-<时间>/`：`server.log`（服务端输出）、`proc/`（各进程日志）、
`command.txt`（实际用的环境变量、命令和代码版本）。加 `--dry-run` 只打印这些，不启动。

- 原版用同样的方式启动：在原版仓库（`Muxserve-change` 分支）运行 `uv run examples/xbench/server.py start ...`，
  见那边的 `examples/xbench/README.md`。原版的 yaml 放在它的 `examples/xbench/` 下，
  和本目录的 `fig5-a0.7-s1.0.yaml` 逐字节相同（sha256 `7134a3df…`）；这份 yaml 里的注释也是原版那份的原文，所以写的是 `examples/xbench`。
- 同一时间只能跑一个服务端：原版和 baseline 用同样的端口和 MPS 目录。另一个还在跑时，`start` 会提示先 `stop`。
