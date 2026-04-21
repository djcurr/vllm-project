/*
 * Adapted from attention_kernels.cuh for mixed block size support.
 * Copyright (c) 2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#include "../cuda_compat.h"

#ifndef USE_ROCM
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#else
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
  typedef __hip_bfloat16 __nv_bfloat16;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum_mixed(float* red_smem, float sum) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  if (lane == 0) {
    red_smem[warp] = sum;
  }

  __syncthreads();

  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  return VLLM_SHFL_SYNC(sum, 0);
}

// Mixed block size paged attention kernel.
// KERNEL_BLOCK_SIZE is the compile-time underlying kernel page size.
// Logical block size can vary per sequence via block_size_multipliers:
//   logical_block_size = kernel_block_size * block_size_multiplier
// The kernel_block_size template parameter allows the same vectorization
// and thread grouping optimizations as the original fixed-size kernel.
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int KERNEL_BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>
__device__ void paged_attention_mixed_kernel_impl(
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_size_multipliers,
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_token_stride,
    const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    return;
  }

  const int block_size_multiplier = block_size_multipliers[seq_idx];
  if (block_size_multiplier <= 0) {
    // Defensive: avoid division by zero if caller passes invalid multiplier.
    return;
  }
  const int logical_block_size = KERNEL_BLOCK_SIZE * block_size_multiplier;
  const bool is_multiplier_one = block_size_multiplier == 1;
  const bool is_multiplier_two = block_size_multiplier == 2;

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, logical_block_size);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / logical_block_size : num_seq_blocks;

  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  const int start_token_idx = start_block_idx * logical_block_size;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * logical_block_size, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / KERNEL_BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(KERNEL_BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();

  // Memory planning.
  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  __shared__ float red_smem[2 * NUM_WARPS];

  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  // Iterate over the key blocks.
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {

    const int block_token_start = block_idx * logical_block_size;
    const int block_token_end = MIN(block_token_start + logical_block_size, seq_len);
    const int tokens_in_block = block_token_end - block_token_start;
    const int num_sub_blocks = is_multiplier_one
                                   ? 1
                                   : is_multiplier_two
                                         ? (tokens_in_block > KERNEL_BLOCK_SIZE ? 2 : 1)
                                         : MIN(block_size_multiplier,
                                               DIVIDE_ROUND_UP(tokens_in_block, KERNEL_BLOCK_SIZE));

    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * logical_block_size / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int sub_block = 0; sub_block < num_sub_blocks; ++sub_block) {
          const int token_base_in_block = sub_block * KERNEL_BLOCK_SIZE;
          for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset =
                (thread_group_idx + i * WARP_SIZE) % KERNEL_BLOCK_SIZE;
            const int token_idx =
                block_idx * logical_block_size + token_base_in_block +
                physical_block_offset;
            if (thread_group_offset == 0 && token_idx < end_token_idx) {
              logits[token_idx - start_token_idx] = -FLT_MAX;
            }
          }
        }
        continue;
      }
    }

    // For mixed block sizes, the logical block maps to block_size_multiplier
    // consecutive physical kernel blocks.
    const int physical_block_start = is_multiplier_one
                                         ? block_idx
                                         : is_multiplier_two
                                               ? (block_idx << 1)
                                               : block_idx * block_size_multiplier;

    // Load all physical sub-blocks that make up this logical block.
    for (int sub_block = 0; sub_block < num_sub_blocks; ++sub_block) {
      const int token_base_in_block = sub_block * KERNEL_BLOCK_SIZE;
      const int64_t physical_block_number =
          static_cast<int64_t>(block_table[physical_block_start + sub_block]);

      for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
        const int physical_block_offset =
            (thread_group_idx + i * WARP_SIZE) % KERNEL_BLOCK_SIZE;
        const int token_idx =
            block_idx * logical_block_size + token_base_in_block +
            physical_block_offset;

        K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
        for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
          const cache_t* k_ptr =
              k_cache + physical_block_number * kv_block_stride +
              physical_block_offset * kv_token_stride +
              kv_head_idx * kv_head_stride;
          const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;

          if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
            k_vecs[j] = *reinterpret_cast<const K_vec*>(
                k_ptr + vec_idx * VEC_SIZE);
          } else {
            Quant_vec k_quant_vec = *reinterpret_cast<const Quant_vec*>(
                k_ptr + vec_idx * VEC_SIZE);
            k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
                k_quant_vec, *k_scale);
          }
        }

        float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                              q_vecs[thread_group_offset], k_vecs);

        if (alibi_slopes != nullptr) {
          qk += alibi_slope * (token_idx - seq_len + 1);
        }

        if (thread_group_offset == 0) {
          const bool mask = token_idx >= end_token_idx;
          logits[token_idx - start_token_idx] = mask ? 0.f : qk;
          qk_max = mask ? qk_max : fmaxf(qk_max, qk);
        }
      }
    }
  }

  // Match the original kernel's max reduction before softmax.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum_mixed<NUM_WARPS>(red_smem, exp_sum);
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);

  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), KERNEL_BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = KERNEL_BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    const int block_token_start = block_idx * logical_block_size;
    const int block_token_end = MIN(block_token_start + logical_block_size, seq_len);
    const int tokens_in_block = block_token_end - block_token_start;
    const int num_sub_blocks = is_multiplier_one
                                   ? 1
                                   : is_multiplier_two
                                         ? (tokens_in_block > KERNEL_BLOCK_SIZE ? 2 : 1)
                                         : MIN(block_size_multiplier,
                                               DIVIDE_ROUND_UP(tokens_in_block, KERNEL_BLOCK_SIZE));

    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * logical_block_size / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }

    const int physical_block_start = is_multiplier_one
                                         ? block_idx
                                         : is_multiplier_two
                                               ? (block_idx << 1)
                                               : block_idx * block_size_multiplier;

    // Iterate over physical sub-blocks within this logical block.
    for (int sub_block = 0; sub_block < num_sub_blocks; ++sub_block) {
      const int token_base_in_block = sub_block * KERNEL_BLOCK_SIZE;
      const int token_idx = block_idx * logical_block_size + token_base_in_block;

      const int64_t physical_block_number =
          static_cast<int64_t>(block_table[physical_block_start + sub_block]);
      const cache_t* v_base = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;

      const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
      const int actual_token_idx = token_idx + physical_block_offset;

      // Load logits for this group of tokens.
      L_vec logits_vec;
      from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(
                                  logits + actual_token_idx - start_token_idx));

#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE) {
          V_vec v_vec;
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);

#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            int token_offset = physical_block_offset + j;
            if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
              v_vec_ptr[j] =
                  v_base[token_offset * kv_token_stride + row_idx];
            } else {
              v_vec_ptr[j] = fp8::scaled_convert<scalar_t, cache_t, KV_DTYPE>(
                  v_base[token_offset * kv_token_stride + row_idx], *v_scale);
            }
          }
          
          // Zero out out-of-context tokens in the last block.
          if (block_idx == num_seq_blocks - 1) {
#pragma unroll
            for (int j = 0; j < V_VEC_SIZE; j++) {
              v_vec_ptr[j] = actual_token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
            }
          }
          
          accs[i] += dot(logits_vec, v_vec);
        }
      }
    }
  }

  // Match the original kernel's warp-local and cross-warp reduction.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  __syncthreads();

  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int j = 0; j < NUM_ROWS_PER_THREAD; j++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[j];
        }
      }
    }
    __syncthreads();

    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int j = 0; j < NUM_ROWS_PER_THREAD; j++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[j] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE +
                        head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Reduce kernel for V2 mixed.
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__global__ void paged_attention_mixed_reduce_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const scalar_t* __restrict__ tmp_out,
    const int* __restrict__ seq_lens,
    const int max_num_partitions) {
  
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int seq_len = seq_lens[seq_idx];
  
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, 512);  // PARTITION_SIZE = 512
  
  __shared__ float shared_max_logit;
  __shared__ float shared_exp_sum;
  
  if (threadIdx.x == 0) {
    float max_logit = -FLT_MAX;
    for (int i = 0; i < num_partitions; ++i) {
      max_logit = MAX(max_logit, max_logits[seq_idx * num_heads * max_num_partitions +
                                            head_idx * max_num_partitions + i]);
    }
    shared_max_logit = max_logit;
    
    float exp_sum = 0.f;
    for (int i = 0; i < num_partitions; ++i) {
      float partition_max = max_logits[seq_idx * num_heads * max_num_partitions +
                                      head_idx * max_num_partitions + i];
      float partition_exp_sum = exp_sums[seq_idx * num_heads * max_num_partitions +
                                         head_idx * max_num_partitions + i];
      exp_sum += partition_exp_sum * expf(partition_max - max_logit);
    }
    shared_exp_sum = exp_sum;
  }
  
  __syncthreads();
  
  const float global_max_logit = shared_max_logit;
  const float global_exp_sum_inv = 1.f / (shared_exp_sum + 1e-6f);
  
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  
  for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
    float acc = 0.f;
    
    for (int p = 0; p < num_partitions; ++p) {
      float partition_max = max_logits[seq_idx * num_heads * max_num_partitions +
                                      head_idx * max_num_partitions + p];
      scalar_t partition_out = tmp_out[seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                                       head_idx * max_num_partitions * HEAD_SIZE +
                                       p * HEAD_SIZE + i];
      acc += static_cast<float>(partition_out) * expf(partition_max - global_max_logit);
    }
    
    out_ptr[i] = static_cast<scalar_t>(acc * global_exp_sum_inv);
  }
}

// Kernel wrappers for launch

// V1 wrapper: grid = (num_heads, num_seqs, 1)
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int KERNEL_BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_mixed_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_size_multipliers,
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_token_stride,
    const int kv_head_stride, const float* k_scale, const float* v_scale,
    const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  paged_attention_mixed_kernel_impl<scalar_t, cache_t, HEAD_SIZE, KERNEL_BLOCK_SIZE,
                                     NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>(
      /*exp_sums=*/nullptr, /*max_logits=*/nullptr, out, q, k_cache, v_cache,
      num_kv_heads, scale, block_tables, seq_lens, block_size_multipliers,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_token_stride, kv_head_stride, k_scale, v_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

// V2 wrapper: grid = (num_heads, num_seqs, max_num_partitions)
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int KERNEL_BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_mixed_kernel(
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    scalar_t* __restrict__ tmp_out,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_size_multipliers,
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_token_stride,
    const int kv_head_stride, const float* k_scale, const float* v_scale,
    const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  paged_attention_mixed_kernel_impl<scalar_t, cache_t, HEAD_SIZE, KERNEL_BLOCK_SIZE,
                                     NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE,
                                     PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, block_size_multipliers, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_token_stride, kv_head_stride,
      k_scale, v_scale, tp_rank, blocksparse_local_blocks, blocksparse_vert_stride,
      blocksparse_block_size, blocksparse_head_sliding_step);
}

}  // namespace vllm

#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
