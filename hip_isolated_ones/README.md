# HIP isolated-ones benchmark (method 5 family)

C++/HIP port of the Colab notebook kernels for counting **isolated 1s** (8-neighbor, boundary = 0) in `N×N` binary matrices.

Designed to run on a **remote AMD GPU** with ROCm (Linux).

## Methods

| Key | Description |
|-----|-------------|
| `1_interview_if_atomic` | Interview baseline — divergent `if (&&)` + global atomic |
| `1b_divergent_early_return` | Early-return variant |
| `2_uniform_block_reduce` | Uniform neighbor-sum + block reduce |
| **`5_tile30_halo`** | Notebook method 5 — 30×30 tile, 32×32 halo, `SM_STRIDE=33` |
| `5a_tile30_nopad` | Same as 5, row stride 32 (bank-conflict A/B) |
| `5e_tile30_u32pack` | Halo rows packed as `uint32[32]` (~128 B LDS) |
| `5g_tile30_u32pack_shfl` | u32 rows + `__shfl` for horizontal neighbors, block `(32,8)` |
| `5b_tile16_halo` | 16×16 tile = 16×16 block (1 thread/cell) |
| `5f_tile16_shfl` | 5b + `__shfl` for left/right neighbors |
| `5c_tile30_nbrcache` | Software cache: precompute 8-neighbor sum for all cells |
| `5d_tile30_block30` | 30×30 tile = 30×30 block (900 threads) |

## Build (Linux + ROCm)

```bash
# On the AMD machine (ROCm installed, e.g. /opt/rocm)
cd hip_isolated_ones
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
cmake --build . -j

# Or one-liner without CMake:
/opt/rocm/bin/hipcc -O3 -std=c++17 -I include src/main.cpp -o hip_isolated_ones
```

Set GPU arch if needed (check with `rocminfo`):

```bash
hipcc -O3 --offload-arch=gfx1030 -std=c++17 -I include src/main.cpp -o hip_isolated_ones
```

## Run

```bash
# Quick correctness (N=64,256)
./hip_isolated_ones --quick

# Full benchmark — matches notebook: N=1024,2048 × sparse/medium/dense
./hip_isolated_ones

# Method 5 family only
./hip_isolated_ones --methods 5

# Baselines + method 5
./hip_isolated_ones --methods all

# Custom sizes
./hip_isolated_ones --sizes 512,1024,2048 --repeats 30
```

Output: correctness vs CPU reference, then **median kernel time (ms)** per method (hipEvents, same scope as notebook: kernel + sync).

## Copy to remote

From your Windows machine:

```bash
scp -r hip_isolated_ones user@amd-host:~/
ssh user@amd-host
cd ~/hip_isolated_ones && hipcc -O3 -std=c++17 -I include src/main.cpp -o hip_isolated_ones
./hip_isolated_ones --quick && ./hip_isolated_ones --methods 5
```

## Interview notes

- **LDS** = AMD term for CUDA shared memory (`__shared__` / `sm` arrays).
- **SM_STRIDE = HALO + 1** avoids 32-bank conflicts on row walks (matmul padding lesson).
- **uint32 row pack** shrinks LDS and uses bit tests instead of 9 byte loads.
- **`__shfl`** passes horizontal neighbors between lanes — best on **5b/5f** where one row maps cleanly to 16-wide subgroups.
- On AMD, `prop.warpSize` is often **64**; shuffle width parameters still work for 16/32 subgroups.
- Compare `5a` vs `5` on your GPU to quantify bank-padding benefit.

## Windows

This project targets **ROCm on Linux**. For local Windows dev without an AMD GPU, use the Colab notebook; build and run on the remote ROCm host.
