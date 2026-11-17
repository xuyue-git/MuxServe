# 用 xbench 测原版 MuxServe

本目录只放配置。原版代码只加了两处：HTTP 入口 `muxserve/muxsched/http_scheduler.py`，
以及 `muxserve/entrypoint.py` 里的 `--http-port` 和负载点参数 `--alpha`、`--rate-scale`、`--base-rate`。
调度逻辑没有改动。

| 文件 | 内容 |
|---|---|
| `fig5-a0.7-s1.0.yaml` | 原版模型配置：三个 LLaMA 模型、TP4、MPS 100/40、`max_num_seqs` 16、KV 每卡 2 GiB（0.05061）。所有负载点共用，负载点由启动命令的 `--alpha`、`--rate-scale` 决定 |

请求由 xbench 发来：MuxServe-SGLang 仓库 `benchmark/xbench/benches.toml` 里的用例
（alpha 0.7 / 1.0 是 `2bdea08f`，其余 Figure 5 网格的点见那份 README）。两边用同一个用例、同一个 xbench 客户端。

## 负载点参数

原版的 ADBS 启动时要知道各模型的到达率。用 `--alpha A --rate-scale S` 时，入口按论文的生成方式
（`benchmark/end_to_end/bench_end_to_end_muxserve.py` 的 `get_real_rate` × `gen_power_law_dis`）算出
`--base-rate × S / (排名 + 1)^A`，排名按 yaml 里 `models` 的顺序；把结果写进
`/tmp/mxo-$USER-load/workload-a<A>-s<S>-b<base>.json`，再按原来的方式交给原版（文件里只有到达率，没有请求）。
`--alpha` 和 `--workload-file` 只能二选一。alpha 0.7 / rate scale 1.0 时三个到达率是
2.0 / 1.2311 / 0.9269 req/s，与之前 Figure 5 那次原版运行的完全相同。

## 启动（在本仓库根目录，用原版自己的 Python 3.9 / CUDA 11.8 环境）

```bash
R=~/muxserve-legacy-a100-20261029-93431228ccf3
export PATH=$R/venv/bin:$(echo "$PATH" | tr : '\n' | grep -v cuda-13.2 | paste -sd:)
export CUDA_HOME=/usr/local/cuda-11.8 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 PYTHONNOUSERSITE=1
export MPS=/tmp/mxo-$USER-mps CUDA_MPS_PIPE_DIRECTORY=/tmp/mxo-$USER-mps/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/mxo-$USER-mps/nvidia-log
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i 2,3,4,5 --query-gpu=uuid --format=csv,noheader | paste -sd,)
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
python -c "import torch; assert torch.cuda.device_count() == 4"
python -m muxserve.launch --nproc_per_node 4 --master-port 39335 \
  examples/xbench/fig5-a0.7-s1.0.yaml --server-port 29702 --flexstore-port 33651 --mps-dir $MPS \
  --schedule-approach adbs --max-num-batched-tokens 2048 --http-port 18080 \
  --alpha 0.7 --rate-scale 1.0
```

- 用哪四张卡写在 `nvidia-smi -i` 后面。**`CUDA_VISIBLE_DEVICES` 必须用 GPU UUID，不能用编号**：
  私有 MPS 也是在这个变量下启动的，客户端写编号时会被当成 MPS 服务那几张卡里的相对编号，
  例如写 `2,3,4,5` 只剩两张卡可见，FlexStore 启动几秒就退出，其余进程一直空等。
- 换负载点只改最后一行的 `--alpha`、`--rate-scale`，客户端换成对应的用例。
- `PATH` 一行去掉了 CUDA 13.2：同一个终端里如果设过实验室的 CUDA 13.2 环境，原版要用 11.8。

日志出现 `MuxScheduler HTTP entry on http://127.0.0.1:18080` 后，在 CrossPool 目录运行
`uv run xbench run --case <用例前缀>`。调度器启动后先等 20 秒，空闲 240 秒会自己退出（原版行为），
所以入口开放后要尽快开始测。

## 停止

先在启动终端按 Ctrl-C（原版自己的停止方式），再执行：

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mxo-$USER-mps/nvidia-mps nvidia-cuda-mps-control
pgrep -u $USER -af "muxserve.entrypoint|muxserve_server.py|torch.distributed.launch"
```

第一条关掉本次的 MPS，第二条确认没有残留进程；
如有残留，只结束这些进程：`pkill -TERM -u $USER -f "muxserve.entrypoint|muxserve_server.py|torch.distributed.launch"`。
