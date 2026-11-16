# 用 xbench 测原版 MuxServe

本目录只放配置，原版代码没有改动（HTTP 入口见 `muxserve/muxsched/http_scheduler.py`）。

| 文件 | 内容 |
|---|---|
| `fig5-a0.7-s1.0.yaml` | 原版模型配置：三个 LLaMA 模型、TP4、MPS 100/40、`max_num_seqs` 16、KV 每卡 2 GiB（0.05061） |
| `fig5-a0.7-s1.0-workload.json` | 只给 ADBS 提供各模型到达率（和以前原版 Figure 5 那次相同），不回放请求 |

对应的 xbench 用例是 MuxServe-SGLang 仓库 `benchmark/xbench/benches.toml` 里的 `2bdea08f`，
两边用同一个用例、同一个 xbench 客户端。

## 启动（在本仓库根目录，用原版自己的 Python 3.9 / CUDA 11.8 环境）

```bash
R=~/muxserve-legacy-a100-20261029-93431228ccf3
export PATH=$R/venv/bin:$PATH CUDA_HOME=/usr/local/cuda-11.8 LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 PYTHONNOUSERSITE=1
export MPS=/tmp/mxo-$USER-mps CUDA_MPS_PIPE_DIRECTORY=/tmp/mxo-$USER-mps/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/mxo-$USER-mps/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
CUDA_VISIBLE_DEVICES=1,2,3,4 nvidia-cuda-mps-control -d
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m muxserve.launch --nproc_per_node 4 --master-port 39335 \
  examples/xbench/fig5-a0.7-s1.0.yaml --server-port 29702 --flexstore-port 33651 --mps-dir $MPS \
  --workload-file examples/xbench/fig5-a0.7-s1.0-workload.json --schedule-approach adbs \
  --max-num-batched-tokens 2048 --http-port 18080
```

两处 `CUDA_VISIBLE_DEVICES` 写同样的四张卡。日志出现 `MuxScheduler HTTP entry on http://127.0.0.1:18080`
后即可在 CrossPool 目录运行 `uv run xbench run --case 2bdea08f`。调度器启动后先等 20 秒，
空闲 240 秒会自己退出（原版行为），所以入口开放后要尽快开始测。

## 停止

```bash
echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/mxo-$USER-mps/nvidia-mps nvidia-cuda-mps-control
pgrep -u $USER -af "muxserve.entrypoint|muxserve_server.py"
```

先在启动终端按 Ctrl-C，再执行上面两条：第一条关掉本次的 MPS，第二条确认没有残留进程；
如有残留，只结束这些进程：`pkill -TERM -u $USER -f "muxserve.entrypoint|muxserve_server.py"`。
