#!/usr/bin/env python3
"""GPU check for prefill_break_slim: the slim graph-break path computes exactly what
the original one does.

    CUDA_VISIBLE_DEVICES=<one idle GPU> $PY -B acceptance/graph_break_slim_gpu_check.py

Needs torch with CUDA, SGLang and the built Head-wise extension (the serving venv).
Q/K/V are views of one projection output, as in the model. For several request
splits it compares
  * attention: _attention_into on the views vs _extend_attention on contiguous copies;
  * KV store: ops.store from the views vs from contiguous copies, into two pools.
Prints GRAPH_BREAK_SLIM_GPU_PASS, or the first difference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from muxserve_sglang import ops
    from muxserve_sglang.sglang_attention import HeadwiseAttentionBackend as B

    torch.manual_seed(0)
    device = torch.device("cuda")
    heads, scale = 13, 128 ** -0.5
    for lengths in ((1,), (37,), (512,), (20, 300), (1, 700, 3), (1019,)):
        rows = sum(lengths)
        qkv = torch.randn(rows, 3 * heads * 128, device=device, dtype=torch.float16)
        q, k, v = qkv.split(heads * 128, dim=1)  # strided views, like the model's split
        views = [B._head_view(t, rows, heads) for t in (q, k, v)]
        assert all(t.data_ptr() == s.data_ptr() for t, s in zip(views, (q, k, v))), "a view was copied"
        dense = [t.reshape(rows, heads, 128).contiguous() for t in (q, k, v)]

        reference = B._extend_attention(*dense, lengths, scale)
        out = torch.empty(rows, heads * 128, device=device, dtype=torch.float16)
        B._attention_into(*views, lengths, scale, out.view(rows, heads, 128))
        diff = (out.float() - reference.float()).abs().max().item()
        if diff > 2e-3:
            print("GRAPH_BREAK_SLIM_GPU_FAILED attention", lengths, diff)
            return 1

        blocks = (rows * heads) // 16 + 8
        slots = torch.randperm(blocks * 16, device=device)[:rows * heads].view(rows, heads).long().contiguous()
        pools = []
        for key, value in ((views[1], views[2]), (dense[1], dense[2])):
            pool_k = torch.zeros(blocks, 16, 16, 8, device=device, dtype=torch.float16)
            pool_v = torch.zeros(blocks, 128, 16, device=device, dtype=torch.float16)
            ops.store(key, value, pool_k, pool_v, slots)
            pools.append((pool_k, pool_v))
        torch.cuda.synchronize()
        if not (torch.equal(pools[0][0], pools[1][0]) and torch.equal(pools[0][1], pools[1][1])):
            print("GRAPH_BREAK_SLIM_GPU_FAILED store", lengths)
            return 1
        print("ok", lengths, "max attention diff %.2e" % diff)
    print("GRAPH_BREAK_SLIM_GPU_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
