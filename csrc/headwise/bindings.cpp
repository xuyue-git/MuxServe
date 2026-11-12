#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void headwise_reshape_and_cache(torch::Tensor&, torch::Tensor&, torch::Tensor&,
                               torch::Tensor&, torch::Tensor&);
void single_query_cached_kv_headwise_attention(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&,
    torch::Tensor&, float, torch::Tensor&, torch::Tensor&, int, int,
    const c10::optional<torch::Tensor>&);

static void pool_check(const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(k.is_cuda() && v.device() == k.device(), "CUDA pool device mismatch");
  TORCH_CHECK(k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
              "Initial Head-wise pool requires FP16");
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous(), "Pool must be contiguous");
  TORCH_CHECK(k.dim() == 4 && k.size(1) == 16 && k.size(2) == 16 && k.size(3) == 8,
              "K layout must be [blocks,16,16,8]");
  TORCH_CHECK(v.dim() == 3 && v.size(0) == k.size(0) && v.size(1) == 128 && v.size(2) == 16,
              "V layout must be [blocks,128,16]");
}

static void store(torch::Tensor key, torch::Tensor value, torch::Tensor k,
                  torch::Tensor v, torch::Tensor slots) {
  pool_check(k, v);
  const c10::cuda::CUDAGuard guard(k.device());
  TORCH_CHECK(key.device() == k.device() && value.device() == k.device() &&
              slots.device() == k.device(), "Writer device mismatch");
  TORCH_CHECK(key.scalar_type() == at::kHalf && value.scalar_type() == at::kHalf,
              "Writer requires FP16");
  TORCH_CHECK(key.dim() == 3 && key.sizes() == value.sizes() && key.size(2) == 128,
              "Writer requires [tokens,heads,128]");
  TORCH_CHECK(key.stride(2) == 1 && key.stride(1) == 128 &&
              value.stride(2) == 1 && value.stride(1) == 128, "Invalid head stride");
  TORCH_CHECK(slots.scalar_type() == at::kLong && slots.is_contiguous() &&
              slots.dim() == 2 && slots.size(0) == key.size(0) &&
              slots.size(1) == key.size(1), "Invalid per-head slots");
  if (key.size(0) == 0) return;
  headwise_reshape_and_cache(key, value, k, v, slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static torch::Tensor decode(torch::Tensor query, torch::Tensor k, torch::Tensor v,
                            torch::Tensor table, torch::Tensor lengths, double scale,
                            int max_context) {
  pool_check(k, v);
  const c10::cuda::CUDAGuard guard(k.device());
  TORCH_CHECK(query.device() == k.device() && table.device() == k.device() &&
              lengths.device() == k.device(), "Decode device mismatch");
  TORCH_CHECK(query.scalar_type() == at::kHalf && query.dim() == 3 &&
              query.size(2) == 128 && query.is_contiguous(), "Invalid decode query");
  TORCH_CHECK(table.scalar_type() == at::kInt && table.is_contiguous() &&
              table.dim() == 3 && table.size(0) == query.size(0) &&
              table.size(1) == query.size(1), "Invalid Head-wise block table");
  TORCH_CHECK(lengths.scalar_type() == at::kInt && lengths.is_contiguous() &&
              lengths.dim() == 1 && lengths.size(0) == query.size(0), "Invalid context lengths");
  TORCH_CHECK(max_context > 0 && max_context <= 2048 && table.size(2) * 16 >= max_context,
              "Invalid context capacity");
  auto output = torch::empty_like(query);
  if (query.size(0) == 0) return output;
  auto head_map = torch::arange(query.size(1), table.options());
  single_query_cached_kv_headwise_attention(output, query, k, v, head_map,
      static_cast<float>(scale), table, lengths, 16, max_context, c10::nullopt);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("store", &store);
  module.def("decode", &decode);
}
