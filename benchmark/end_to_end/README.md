# The paper's end-to-end tooling (original authors)

These files come unchanged from original MuxServe. They generate the paper's
Figure 5 configurations and workloads for its 32-GPU cluster (19 models,
alpha 0.7 / 0.9 / 1.3 / 1.7 / 2.1) and summarise its logs:

| File | Use |
|---|---|
| `bench_end_to_end_muxserve.py` | Placement configs and ShareGPT workloads per alpha / rate scale (paths at the top of the file are the authors') |
| `plot_alpha_rate_dist.py` | Figure 6, the cumulative rate distribution per alpha |
| `plot_p_latency.py` | Figure 11, P99 latency / TPOT / TTFT from the authors' logs |
| `process_log_file.ipynb` | The authors' log parsing |

The authors' shell launchers (`run_end_to_end.sh`, `srun_end_to_end.sh`, which
start MPS with a root password) are not part of this repository.

**In this repository the Figure 5 experiment is run with xbench.** Its rate formula
(`get_real_rate` x `gen_power_law_dis`, rate = base x scale / (rank + 1) ** alpha) is
the one behind `muxserve serve --alpha --rate-scale`, and the 20 Figure 5 points on
four A100s are xbench cases; see [`benchmark/xbench/README.md`](../xbench/README.md).
