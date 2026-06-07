#pragma once

#include <hip/hip_runtime.h>
#include "common.hpp"

namespace iso {

__device__ inline void block_reduce_256(int* sm_sum, int tid) {
    sm_sum[tid] = sm_sum[tid];
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sm_sum[tid] += sm_sum[tid + stride];
        __syncthreads();
    }
}

__device__ inline void block_reduce_900(int* sm_sum, int tid) {
    __syncthreads();
    for (int stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other = tid + stride;
            if (other < BLOCK30) sm_sum[tid] += sm_sum[other];
        }
        __syncthreads();
    }
}

__device__ inline int neighbor_sum_u8(const uint8_t* sm, int stride, int ly, int lx) {
    return sm[(ly - 1) * stride + lx - 1] + sm[(ly - 1) * stride + lx] +
           sm[(ly - 1) * stride + lx + 1] + sm[ly * stride + lx - 1] +
           sm[ly * stride + lx + 1] + sm[(ly + 1) * stride + lx - 1] +
           sm[(ly + 1) * stride + lx] + sm[(ly + 1) * stride + lx + 1];
}

__device__ inline int neighbor_sum_u32_row(const uint32_t* rows, int ly, int lx) {
    auto bit = [&](int y, int x) -> int {
        return static_cast<int>((rows[y] >> x) & 1u);
    };
    return bit(ly - 1, lx - 1) + bit(ly - 1, lx) + bit(ly - 1, lx + 1) + bit(ly, lx - 1) +
           bit(ly, lx + 1) + bit(ly + 1, lx - 1) + bit(ly + 1, lx) + bit(ly + 1, lx + 1);
}

// --- Baselines (notebook methods 1, 1b, 2) ---

__global__ void kernel_interview_if_atomic(const uint8_t* grid, int N, int* out) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    if (grid[i * N + j] == 1 && (i == 0 || j == 0 || grid[(i - 1) * N + j - 1] == 0) &&
        (i == 0 || grid[(i - 1) * N + j] == 0) &&
        (i == 0 || j == N - 1 || grid[(i - 1) * N + j + 1] == 0) &&
        (j == 0 || grid[i * N + j - 1] == 0) &&
        (j == N - 1 || grid[i * N + j + 1] == 0) &&
        (i == N - 1 || j == 0 || grid[(i + 1) * N + j - 1] == 0) &&
        (i == N - 1 || grid[(i + 1) * N + j] == 0) &&
        (i == N - 1 || j == N - 1 || grid[(i + 1) * N + j + 1] == 0)) {
        atomicAdd(out, 1);
    }
}

__global__ void kernel_divergent_early_return(const uint8_t* grid, int N, int* out) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    if (grid[i * N + j] != 1) return;
    if (i > 0 && grid[(i - 1) * N + j] == 1) return;
    if (i < N - 1 && grid[(i + 1) * N + j] == 1) return;
    if (j > 0 && grid[i * N + j - 1] == 1) return;
    if (j < N - 1 && grid[i * N + j + 1] == 1) return;
    if (i > 0 && j > 0 && grid[(i - 1) * N + j - 1] == 1) return;
    if (i > 0 && j < N - 1 && grid[(i - 1) * N + j + 1] == 1) return;
    if (i < N - 1 && j > 0 && grid[(i + 1) * N + j - 1] == 1) return;
    if (i < N - 1 && j < N - 1 && grid[(i + 1) * N + j + 1] == 1) return;
    atomicAdd(out, 1);
}

__global__ void kernel_uniform_block_reduce(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int isolated = 0;
    if (i <= N && j <= N) {
        uint8_t v = padded[i * P + j];
        int n = padded[(i - 1) * P + j - 1] + padded[(i - 1) * P + j] + padded[(i - 1) * P + j + 1] +
              padded[i * P + j - 1] + padded[i * P + j + 1] + padded[(i + 1) * P + j - 1] +
              padded[(i + 1) * P + j] + padded[(i + 1) * P + j + 1];
        isolated = (v == 1 && n == 0) ? 1 : 0;
    }
    __shared__ int sm[256];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    sm[tid] = isolated;
    block_reduce_256(sm, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm[0];
}

// --- Method 5: tile30 halo (notebook) ---

__global__ void kernel_tile30_halo(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * ROW_STEP;
    int tc = blockIdx.x * TILE;
    int bdx = blockDim.x;
    int tid = threadIdx.y * bdx + threadIdx.x;
    int nt = bdx * blockDim.y;

    __shared__ uint8_t sm[SM_HALO];
    for (int idx = tid; idx < HALO * HALO; idx += nt) {
        int ly = idx / HALO;
        int lx = idx % HALO;
        int sr = tr + ly;
        int sc = tc + lx;
        int si = ly * SM_STRIDE + lx;
        sm[si] = (sr < P && sc < P) ? padded[sr * P + sc] : 0;
    }
    __syncthreads();

    int local = 0;
    if (tr < N && tc < N) {
        int lim_r = (tr + TILE > N) ? N - tr : TILE;
        int lim_c = (tc + TILE > N) ? N - tc : TILE;
        int ncells = lim_r * lim_c;
        for (int idx = tid; idx < ncells; idx += nt) {
            int a = idx / lim_c;
            int b = idx % lim_c;
            int ly = a + 1;
            int lx = b + 1;
            uint8_t v = sm[ly * SM_STRIDE + lx];
            if (v == 1 && neighbor_sum_u8(sm, SM_STRIDE, ly, lx) == 0) ++local;
        }
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5a: same kernel but row stride = HALO (no bank padding) — A/B vs SM_STRIDE
__global__ void kernel_tile30_halo_nopad(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * ROW_STEP;
    int tc = blockIdx.x * TILE;
    int bdx = blockDim.x;
    int tid = threadIdx.y * bdx + threadIdx.x;
    int nt = bdx * blockDim.y;

    __shared__ uint8_t sm[HALO * HALO];
    for (int idx = tid; idx < HALO * HALO; idx += nt) {
        int ly = idx / HALO;
        int lx = idx % HALO;
        int sr = tr + ly;
        int sc = tc + lx;
        int si = ly * HALO + lx;
        sm[si] = (sr < P && sc < P) ? padded[sr * P + sc] : 0;
    }
    __syncthreads();

    int local = 0;
    if (tr < N && tc < N) {
        int lim_r = (tr + TILE > N) ? N - tr : TILE;
        int lim_c = (tc + TILE > N) ? N - tc : TILE;
        int ncells = lim_r * lim_c;
        for (int idx = tid; idx < ncells; idx += nt) {
            int a = idx / lim_c;
            int b = idx % lim_c;
            int ly = a + 1;
            int lx = b + 1;
            if (sm[ly * HALO + lx] == 1 && neighbor_sum_u8(sm, HALO, ly, lx) == 0) ++local;
        }
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5e: uint32 row pack — 32 halo rows x 32 bits (128 B LDS vs 1056 B)
__global__ void kernel_tile30_u32pack(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * ROW_STEP;
    int tc = blockIdx.x * TILE;
    int bdx = blockDim.x;
    int tid = threadIdx.y * bdx + threadIdx.x;
    int nt = bdx * blockDim.y;

    __shared__ uint32_t rows[HALO];
    for (int ly = tid; ly < HALO; ly += nt) {
        uint32_t w = 0;
        int sr = tr + ly;
        if (sr < P) {
            int base = sr * P + tc;
            for (int lx = 0; lx < HALO; ++lx) {
                int sc = tc + lx;
                if (sc < P && padded[base + lx]) w |= (1u << lx);
            }
        }
        rows[ly] = w;
    }
    __syncthreads();

    int local = 0;
    if (tr < N && tc < N) {
        int lim_r = (tr + TILE > N) ? N - tr : TILE;
        int lim_c = (tc + TILE > N) ? N - tc : TILE;
        int ncells = lim_r * lim_c;
        for (int idx = tid; idx < ncells; idx += nt) {
            int a = idx / lim_c;
            int b = idx % lim_c;
            int ly = a + 1;
            int lx = b + 1;
            if (((rows[ly] >> lx) & 1u) && neighbor_sum_u32_row(rows, ly, lx) == 0) ++local;
        }
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5b: tile16 halo
__global__ void kernel_tile16_halo(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * TILE16;
    int tc = blockIdx.x * TILE16;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * TILE16 + tx;

    __shared__ uint8_t sm[SM16];
    for (int idx = tid; idx < SM16; idx += 256) {
        int ly = idx / HALO16;
        int lx = idx % HALO16;
        int sr = tr + ly;
        int sc = tc + lx;
        sm[idx] = (sr < P && sc < P) ? padded[sr * P + sc] : 0;
    }
    __syncthreads();

    int local = 0;
    int r = tr + ty;
    int c = tc + tx;
    if (r < N && c < N) {
        int ly = ty + 1;
        int lx = tx + 1;
        if (sm[ly * HALO16 + lx] == 1 && neighbor_sum_u8(sm, HALO16, ly, lx) == 0) local = 1;
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5f: tile16 + __shfl for horizontal neighbors (16-wide subgroup per row)
__global__ void kernel_tile16_shfl(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * TILE16;
    int tc = blockIdx.x * TILE16;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * TILE16 + tx;

    __shared__ uint8_t sm[SM16];
    for (int idx = tid; idx < SM16; idx += 256) {
        int ly = idx / HALO16;
        int lx = idx % HALO16;
        int sr = tr + ly;
        int sc = tc + lx;
        sm[idx] = (sr < P && sc < P) ? padded[sr * P + sc] : 0;
    }
    __syncthreads();

    int local = 0;
    int r = tr + ty;
    int c = tc + tx;
    if (r < N && c < N) {
        int ly = ty + 1;
        int lx = tx + 1;
        int si = ly * HALO16 + lx;
        uint8_t v = sm[si];
        if (v == 1) {
            int center = 1;
            int left = static_cast<int>(sm[si - 1]);
            int right = static_cast<int>(sm[si + 1]);
#if defined(__HIP_DEVICE_COMPILE__)
            left = __shfl(center, tx - 1, TILE16);
            right = __shfl(center, tx + 1, TILE16);
            if (tx == 0) left = static_cast<int>(sm[si - 1]);
            if (tx == TILE16 - 1) right = static_cast<int>(sm[si + 1]);
#endif
            int up = sm[(ly - 1) * HALO16 + lx - 1] + sm[(ly - 1) * HALO16 + lx] +
                     sm[(ly - 1) * HALO16 + lx + 1];
            int down = sm[(ly + 1) * HALO16 + lx - 1] + sm[(ly + 1) * HALO16 + lx] +
                       sm[(ly + 1) * HALO16 + lx + 1];
            int n = left + right + up + down;
            if (n == 0) local = 1;
        }
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5c: neighbor-sum software cache
__global__ void kernel_tile30_nbrcache(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * ROW_STEP;
    int tc = blockIdx.x * TILE;
    int bdx = blockDim.x;
    int tid = threadIdx.y * bdx + threadIdx.x;
    int nt = bdx * blockDim.y;

    __shared__ uint8_t sm[SM_HALO];
    __shared__ uint8_t sm_nbr[SM_HALO];
    for (int idx = tid; idx < HALO * HALO; idx += nt) {
        int ly = idx / HALO;
        int lx = idx % HALO;
        int sr = tr + ly;
        int sc = tc + lx;
        int si = ly * SM_STRIDE + lx;
        sm[si] = (sr < P && sc < P) ? padded[sr * P + sc] : 0;
    }
    __syncthreads();

    int local = 0;
    if (tr < N && tc < N) {
        int lim_r = (tr + TILE > N) ? N - tr : TILE;
        int lim_c = (tc + TILE > N) ? N - tc : TILE;
        int ncells = lim_r * lim_c;
        for (int idx = tid; idx < ncells; idx += nt) {
            int a = idx / lim_c;
            int b = idx % lim_c;
            int ly = a + 1;
            int lx = b + 1;
            int si = ly * SM_STRIDE + lx;
            sm_nbr[si] = static_cast<uint8_t>(neighbor_sum_u8(sm, SM_STRIDE, ly, lx));
        }
    }
    __syncthreads();

    if (tr < N && tc < N) {
        int lim_r = (tr + TILE > N) ? N - tr : TILE;
        int lim_c = (tc + TILE > N) ? N - tc : TILE;
        int ncells = lim_r * lim_c;
        for (int idx = tid; idx < ncells; idx += nt) {
            int a = idx / lim_c;
            int b = idx % lim_c;
            int ly = a + 1;
            int lx = b + 1;
            int si = ly * SM_STRIDE + lx;
            if (sm[si] == 1 && sm_nbr[si] == 0) ++local;
        }
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5d: 30x30 tile == 30x30 block
__global__ void kernel_tile30_block30(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * ROW_STEP;
    int tc = blockIdx.x * TILE;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * TILE + tx;

    __shared__ uint8_t sm[SM_HALO];
    for (int idx = tid; idx < HALO * HALO; idx += BLOCK30) {
        int ly = idx / HALO;
        int lx = idx % HALO;
        int sr = tr + ly;
        int sc = tc + lx;
        int si = ly * SM_STRIDE + lx;
        sm[si] = (sr < P && sc < P) ? padded[sr * P + sc] : 0;
    }
    __syncthreads();

    int local = 0;
    int r = tr + ty;
    int c = tc + tx;
    if (r < N && c < N) {
        int ly = ty + 1;
        int lx = tx + 1;
        if (sm[ly * SM_STRIDE + lx] == 1 && neighbor_sum_u8(sm, SM_STRIDE, ly, lx) == 0) local = 1;
    }

    __shared__ int sm_sum[SM_SUM1024];
    sm_sum[tid] = local;
    block_reduce_900(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

// 5g: tile30 u32 rows + 32-wide shuffle for left/right on interior row walks
__global__ void kernel_tile30_u32pack_shfl(const uint8_t* padded, int N, int* block_out) {
    int P = N + 2;
    int tr = blockIdx.y * ROW_STEP;
    int tc = blockIdx.x * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nt = blockDim.x * blockDim.y;

    __shared__ uint32_t rows[HALO];
    for (int ly = tid; ly < HALO; ly += nt) {
        uint32_t w = 0;
        int sr = tr + ly;
        if (sr < P) {
            int base = sr * P + tc;
            for (int lx = 0; lx < HALO; ++lx) {
                int sc = tc + lx;
                if (sc < P && padded[base + lx]) w |= (1u << lx);
            }
        }
        rows[ly] = w;
    }
    __syncthreads();

    int local = 0;
    if (tr < N && tc < N) {
        int lim_r = (tr + TILE > N) ? N - tr : TILE;
        int lim_c = (tc + TILE > N) ? N - tc : TILE;
        for (int ar = ty; ar < lim_r; ar += blockDim.y) {
            int ly = ar + 1;
            int lx = tx + 1;
            if (lx > lim_c) continue;
            uint32_t row0 = rows[ly];
            int center = static_cast<int>((row0 >> lx) & 1u);
            if (!center) continue;
            int left = static_cast<int>((row0 >> (lx - 1)) & 1u);
            int right = static_cast<int>((row0 >> (lx + 1)) & 1u);
#if defined(__HIP_DEVICE_COMPILE__)
            left = __shfl(center, tx - 1, HALO);
            right = __shfl(center, tx + 1, HALO);
#endif
            if (lx == 1) left = static_cast<int>((row0 >> 0) & 1u);
            if (lx == lim_c) right = static_cast<int>((row0 >> (lx + 1)) & 1u);
            uint32_t row_m = rows[ly - 1];
            uint32_t row_p = rows[ly + 1];
            int n = left + right + static_cast<int>((row_m >> (lx - 1)) & 1u) +
                    static_cast<int>((row_m >> lx) & 1u) + static_cast<int>((row_m >> (lx + 1)) & 1u) +
                    static_cast<int>((row_p >> (lx - 1)) & 1u) + static_cast<int>((row_p >> lx) & 1u) +
                    static_cast<int>((row_p >> (lx + 1)) & 1u);
            if (n == 0) ++local;
        }
    }

    __shared__ int sm_sum[256];
    sm_sum[tid] = local;
    block_reduce_256(sm_sum, tid);
    if (tid == 0) block_out[blockIdx.y * gridDim.x + blockIdx.x] = sm_sum[0];
}

}  // namespace iso
