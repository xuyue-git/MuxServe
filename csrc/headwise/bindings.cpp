// Python bindings of MuxServe-vLLM's head-wise KV ops, as MuxServe-vLLM binds
// them (csrc/attention.cpp and csrc/cache.cpp of MuxServe-vLLM eaca02e): the
// same two functions with the same arguments. The kernels are the copied files
// next to this one.
#include <c10/util/Optional.h>
#include <torch/extension.h>

void single_query_cached_kv_headwise_attention(torch::Tensor& out, torch::Tensor& query,
                                               torch::Tensor& key_cache, torch::Tensor& value_cache,
                                               torch::Tensor& head_mapping, float scale,
                                               torch::Tensor& block_tables, torch::Tensor& context_lens,
                                               int block_size, int max_context_len,
                                               const c10::optional<torch::Tensor>& alibi_slopes);

void headwise_reshape_and_cache(torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache,
                                torch::Tensor& value_cache, torch::Tensor& slot_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("single_query_cached_kv_headwise_attention", &single_query_cached_kv_headwise_attention,
          "Compute the attention between an input query and the cached key/value tensors");
    m.def("headwise_reshape_and_cache", &headwise_reshape_and_cache,
          "Reshape the key and value tensors and cache them");
}
