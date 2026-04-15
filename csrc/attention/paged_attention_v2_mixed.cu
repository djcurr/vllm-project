/*
 * Adapted from paged_attention_v2.cu for mixed block size support.
 */

#include "attention_kernels_mixed.cuh"
#include "../cuda_compat.h"
#include <c10/cuda/CUDAException.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

template <typename T, typename CACHE_T, int HEAD_SIZE, int KERNEL_BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_mixed_launcher_impl(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    torch::Tensor& block_size_multipliers, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_token_stride = key_cache.stride(1);
  int kv_head_stride = key_cache.stride(2);

  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  int* block_size_multipliers_ptr = block_size_multipliers.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());

  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int v_smem_size = NUM_WARPS * KERNEL_BLOCK_SIZE * head_size * sizeof(CACHE_T);

  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size + v_smem_size, outputs_size);
  
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vllm::paged_attention_v2_mixed_kernel<T, CACHE_T, HEAD_SIZE, KERNEL_BLOCK_SIZE,
                                        NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE,
                                        PARTITION_SIZE>
      <<<grid, block, shared_mem_size, stream>>>(
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr,
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr, seq_lens_ptr,
          block_size_multipliers_ptr, max_num_blocks_per_seq, alibi_slopes_ptr,
          q_stride, kv_block_stride, kv_token_stride, kv_head_stride, k_scale_ptr,
          v_scale_ptr,
          tp_rank, blocksparse_local_blocks, blocksparse_vert_stride,
          blocksparse_block_size, blocksparse_head_sliding_step);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  vllm::paged_attention_mixed_reduce_kernel<T, HEAD_SIZE, NUM_THREADS>
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,
          max_num_partitions);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, typename CACHE_T, int HEAD_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_mixed_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    torch::Tensor& block_size_multipliers, int max_seq_len,
    int kernel_block_size, const std::optional<torch::Tensor>& alibi_slopes,
    torch::Tensor& k_scale, torch::Tensor& v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  switch (kernel_block_size) {
    case 8:
      paged_attention_v2_mixed_launcher_impl<T, CACHE_T, HEAD_SIZE, 8, KV_DTYPE,
                                              IS_BLOCK_SPARSE>(
          out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,
          num_kv_heads, scale, block_tables, seq_lens, block_size_multipliers,
          max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,
          blocksparse_local_blocks, blocksparse_vert_stride,
          blocksparse_block_size, blocksparse_head_sliding_step);
      break;
    case 16:
      paged_attention_v2_mixed_launcher_impl<T, CACHE_T, HEAD_SIZE, 16, KV_DTYPE,
                                              IS_BLOCK_SPARSE>(
          out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,
          num_kv_heads, scale, block_tables, seq_lens, block_size_multipliers,
          max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,
          blocksparse_local_blocks, blocksparse_vert_stride,
          blocksparse_block_size, blocksparse_head_sliding_step);
      break;
    default:
      TORCH_CHECK(false, "Unsupported kernel block size for mixed attention v2: ",
                  kernel_block_size);
      break;
  }
}

#define CALL_V2_MIXED_LAUNCHER(T, CACHE_T, HEAD_SIZE, KV_DTYPE, IS_BLOCK_SPARSE) \
  paged_attention_v2_mixed_launcher<T, CACHE_T, HEAD_SIZE, KV_DTYPE,            \
                                    IS_BLOCK_SPARSE>(                            \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,         \
      num_kv_heads, scale, block_tables, seq_lens, block_size_multipliers,       \
      max_seq_len, kernel_block_size, alibi_slopes, k_scale, v_scale, tp_rank,   \
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size, \
      blocksparse_head_sliding_step);

#define CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, HEAD_SIZE, IS_FP8_KV_CACHE) \
  if (is_block_sparse) {                                                        \
    CALL_V2_MIXED_LAUNCHER(T, CACHE_T, HEAD_SIZE, IS_FP8_KV_CACHE, true);       \
  } else {                                                                      \
    CALL_V2_MIXED_LAUNCHER(T, CACHE_T, HEAD_SIZE, IS_FP8_KV_CACHE, false);      \
  }

void paged_attention_v2_mixed(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    torch::Tensor& block_size_multipliers, int64_t max_seq_len,
    int64_t kernel_block_size,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);
  int head_size = query.size(2);

#define DISPATCH_HEAD_SIZE(T, CACHE_T, KV_DTYPE)                              \
  switch (head_size) {                                                        \
    case 32:                                                                  \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);              \
      break;                                                                  \
    case 64:                                                                  \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 64, KV_DTYPE);              \
      break;                                                                  \
    case 80:                                                                  \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 80, KV_DTYPE);              \
      break;                                                                  \
    case 96:                                                                  \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 96, KV_DTYPE);              \
      break;                                                                  \
    case 112:                                                                 \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 112, KV_DTYPE);             \
      break;                                                                  \
    case 120:                                                                 \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 120, KV_DTYPE);             \
      break;                                                                  \
    case 128:                                                                 \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 128, KV_DTYPE);             \
      break;                                                                  \
    case 192:                                                                 \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 192, KV_DTYPE);             \
      break;                                                                  \
    case 256:                                                                 \
      CALL_V2_MIXED_LAUNCHER_SPARSITY(T, CACHE_T, 256, KV_DTYPE);             \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false, "Unsupported head size for mixed attention v2: ",    \
                  head_size);                                                 \
      break;                                                                  \
  }

  if (query.dtype() == torch::kFloat16) {
    if (kv_cache_dtype == "auto") {
      DISPATCH_HEAD_SIZE(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
      DISPATCH_HEAD_SIZE(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported KV cache dtype for mixed attention v2: ",
                  kv_cache_dtype);
    }
  } else if (query.dtype() == torch::kBFloat16) {
    if (kv_cache_dtype == "auto") {
      DISPATCH_HEAD_SIZE(__nv_bfloat16, __nv_bfloat16,
                         vllm::Fp8KVCacheDataType::kAuto);
    } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
      DISPATCH_HEAD_SIZE(__nv_bfloat16, __nv_bfloat16,
                         vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported KV cache dtype for mixed attention v2: ",
                  kv_cache_dtype);
    }
  } else {
    TORCH_CHECK(false, "Unsupported query dtype for mixed attention v2: ",
                query.dtype());
  }

#undef DISPATCH_HEAD_SIZE
#undef CALL_V2_MIXED_LAUNCHER_SPARSITY
#undef CALL_V2_MIXED_LAUNCHER
}

#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
