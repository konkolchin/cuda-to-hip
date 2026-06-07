#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "kernels.hpp"

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            std::cerr << "HIP error " << hipGetErrorString(err) << " at "       \
                      << __FILE__ << ":" << __LINE__ << "\n";                   \
            std::exit(1);                                                     \
        }                                                                       \
    } while (0)

namespace iso {

struct MethodSpec {
    const char* name;
    const char* launch;
    int (*run)(const uint8_t* d_grid, const uint8_t* d_padded, int N, int* d_scalar,
               int* d_block_out, int nblocks);
};

int grid_2d(int N) { return div_up(N, TPB); }

int grid_tile30_x(int N) { return div_up(N, TILE); }
int grid_tile30_y(int N) { return div_up(N, ROW_STEP); }
int grid_tile30_blocks(int N) { return grid_tile30_x(N) * grid_tile30_y(N); }

int sum_block_out(int* d_block_out, int nblocks) {
    std::vector<int> h(nblocks);
    HIP_CHECK(hipMemcpy(h.data(), d_block_out, nblocks * sizeof(int), hipMemcpyDeviceToHost));
    long long s = 0;
    for (int v : h) s += v;
    return static_cast<int>(s);
}

int run_m1(const uint8_t* d_grid, const uint8_t*, int N, int* d_scalar, int*, int) {
    int zero = 0;
    HIP_CHECK(hipMemcpy(d_scalar, &zero, sizeof(int), hipMemcpyHostToDevice));
    int bx = grid_2d(N);
    dim3 grid(bx, bx);
    dim3 block(TPB, TPB);
    kernel_interview_if_atomic<<<grid, block>>>(d_grid, N, d_scalar);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    int out = 0;
    HIP_CHECK(hipMemcpy(&out, d_scalar, sizeof(int), hipMemcpyDeviceToHost));
    return out;
}

int run_m1b(const uint8_t* d_grid, const uint8_t*, int N, int* d_scalar, int*, int) {
    int zero = 0;
    HIP_CHECK(hipMemcpy(d_scalar, &zero, sizeof(int), hipMemcpyHostToDevice));
    int bx = grid_2d(N);
    dim3 grid(bx, bx);
    dim3 block(TPB, TPB);
    kernel_divergent_early_return<<<grid, block>>>(d_grid, N, d_scalar);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    int out = 0;
    HIP_CHECK(hipMemcpy(&out, d_scalar, sizeof(int), hipMemcpyDeviceToHost));
    return out;
}

int run_m2(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out, int nblocks) {
    int bx = grid_2d(N);
    dim3 grid(bx, bx);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_uniform_block_reduce<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile30(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out, int nblocks) {
    int bx = grid_tile30_x(N);
    int by = grid_tile30_y(N);
    dim3 grid(bx, by);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile30_halo<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile30_nopad(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out,
                     int nblocks) {
    int bx = grid_tile30_x(N);
    int by = grid_tile30_y(N);
    dim3 grid(bx, by);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile30_halo_nopad<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile30_u32pack(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out,
                       int nblocks) {
    int bx = grid_tile30_x(N);
    int by = grid_tile30_y(N);
    dim3 grid(bx, by);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile30_u32pack<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile30_u32pack_shfl(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out,
                            int nblocks) {
    int bx = grid_tile30_x(N);
    int by = grid_tile30_y(N);
    dim3 grid(bx, by);
    dim3 block(HALO, 8);  // 32x8 = 256 threads; one column per lane in row walks
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile30_u32pack_shfl<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile16(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out, int nblocks) {
    int bx = grid_2d(N);
    dim3 grid(bx, bx);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile16_halo<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile16_shfl(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out,
                    int nblocks) {
    int bx = grid_2d(N);
    dim3 grid(bx, bx);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile16_shfl<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile30_nbrcache(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out,
                        int nblocks) {
    int bx = grid_tile30_x(N);
    int by = grid_tile30_y(N);
    dim3 grid(bx, by);
    dim3 block(TPB, TPB);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile30_nbrcache<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

int run_tile30_block30(const uint8_t*, const uint8_t* d_padded, int N, int*, int* d_block_out,
                       int nblocks) {
    int bx = grid_tile30_x(N);
    int by = grid_tile30_y(N);
    dim3 grid(bx, by);
    dim3 block(TILE, TILE);
    HIP_CHECK(hipMemset(d_block_out, 0, nblocks * sizeof(int)));
    kernel_tile30_block30<<<grid, block>>>(d_padded, N, d_block_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    return sum_block_out(d_block_out, nblocks);
}

const std::vector<MethodSpec>& all_methods() {
    static const std::vector<MethodSpec> m = {
        {"1_interview_if_atomic", "grid (ceil(N/16))^2 block (16,16)", run_m1},
        {"1b_divergent_early_return", "grid (ceil(N/16))^2 block (16,16)", run_m1b},
        {"2_uniform_block_reduce", "grid (ceil(N/16))^2 block (16,16)", run_m2},
        {"5_tile30_halo", "grid (ceil(N/30), ceil(N/30)) block (16,16) SM_STRIDE=33", run_tile30},
        {"5a_tile30_nopad", "grid (ceil(N/30), ceil(N/30)) block (16,16) stride=32 A/B", run_tile30_nopad},
        {"5e_tile30_u32pack", "grid (ceil(N/30), ceil(N/30)) block (16,16) uint32 rows", run_tile30_u32pack},
        {"5g_tile30_u32pack_shfl", "grid (ceil(N/30), ceil(N/30)) block (32,8) u32+shfl", run_tile30_u32pack_shfl},
        {"5b_tile16_halo", "grid (ceil(N/16))^2 block (16,16) tile==block", run_tile16},
        {"5f_tile16_shfl", "grid (ceil(N/16))^2 block (16,16) __shfl horiz", run_tile16_shfl},
        {"5c_tile30_nbrcache", "grid (ceil(N/30), ceil(N/30)) block (16,16) nbr cache", run_tile30_nbrcache},
        {"5d_tile30_block30", "grid (ceil(N/30), ceil(N/30)) block (30,30)", run_tile30_block30},
    };
    return m;
}

struct BenchStats {
    double median_ms = 0;
    double mean_ms = 0;
    double std_ms = 0;
};

BenchStats benchmark_method(const MethodSpec& spec, const uint8_t* d_grid, const uint8_t* d_padded,
                            int N, int* d_scalar, int* d_block_out, int nblocks_tile30,
                            int nblocks_2d, int warmup, int repeats) {
    int nblocks = (spec.run == run_m1 || spec.run == run_m1b) ? 1
                  : (spec.run == run_m2 || spec.run == run_tile16 || spec.run == run_tile16_shfl)
                        ? nblocks_2d
                        : nblocks_tile30;

    for (int i = 0; i < warmup; ++i) spec.run(d_grid, d_padded, N, d_scalar, d_block_out, nblocks);

    std::vector<double> times;
    times.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        spec.run(d_grid, d_padded, N, d_scalar, d_block_out, nblocks);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times.push_back(static_cast<double>(ms));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    std::sort(times.begin(), times.end());
    BenchStats s;
    s.median_ms = times[times.size() / 2];
    double sum = 0;
    for (double t : times) sum += t;
    s.mean_ms = sum / times.size();
    double var = 0;
    for (double t : times) var += (t - s.mean_ms) * (t - s.mean_ms);
    s.std_ms = std::sqrt(var / times.size());
    return s;
}

void print_device_info() {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "HIP device: " << prop.name << "\n";
    std::cout << "  arch: " << prop.major << "." << prop.minor
              << "  wavefront/warp size: " << prop.warpSize << "\n";
    std::cout << "  LDS (shared) / CU: " << prop.sharedMemPerMultiprocessor / 1024 << " KB\n";
    std::cout << "  max threads/block: " << prop.maxThreadsPerBlock << "\n";
}

std::vector<int> parse_sizes(const char* arg) {
    std::vector<int> sizes;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) sizes.push_back(std::stoi(item));
    }
    return sizes;
}

bool method_enabled(const std::string& filter, const char* name) {
    if (filter == "all") return true;
    if (filter == "5" && name[0] == '5') return true;
    if (filter == "baseline" && (name[0] == '1' || name[0] == '2')) return true;
    return filter == name;
}

}  // namespace iso

int main(int argc, char** argv) {
    using namespace iso;

    bool quick = false;
    std::string method_filter = "all";
    std::vector<int> sizes = {1024, 2048};
    int warmup = 5;
    int repeats = 20;
    uint32_t seed = 7;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") {
            quick = true;
            sizes = {64, 256};
            repeats = 5;
        } else if (a == "--methods" && i + 1 < argc) {
            method_filter = argv[++i];
        } else if (a == "--sizes" && i + 1 < argc) {
            sizes = parse_sizes(argv[++i]);
        } else if (a == "--repeats" && i + 1 < argc) {
            repeats = std::stoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: hip_isolated_ones [options]\n"
                << "  --quick              N=64,256 only; fast correctness check\n"
                << "  --methods all|5|baseline|<exact_name>\n"
                << "  --sizes 1024,2048\n"
                << "  --repeats 20\n"
                << "  --seed 7\n";
            return 0;
        }
    }

    print_device_info();
    std::cout << "\nMethods (filter=" << method_filter << "):\n";
    for (const auto& m : all_methods()) {
        if (method_enabled(method_filter, m.name)) std::cout << "  " << m.name << " — " << m.launch << "\n";
    }
    std::cout << "\n";

    int* d_scalar = nullptr;
    HIP_CHECK(hipMalloc(&d_scalar, sizeof(int)));

    for (int N : sizes) {
        int nblocks_2d = grid_2d(N) * grid_2d(N);
        int nblocks_tile30 = grid_tile30_blocks(N);
        int max_blocks = std::max(nblocks_2d, nblocks_tile30);
        int* d_block_out = nullptr;
        HIP_CHECK(hipMalloc(&d_block_out, max_blocks * sizeof(int)));

        for (const auto& scenario : default_scenarios()) {
            std::cout << "=== N=" << N << " density=" << scenario.density << " (" << scenario.label
                      << ") ===\n";
            auto grid = make_grid(N, scenario.density, seed + N + static_cast<int>(scenario.density * 1000));
            auto padded = pad_grid(grid.data(), N);
            int ref = count_isolated_cpu(grid.data(), N);
            int n_ones = 0;
            for (uint8_t v : grid) n_ones += v;
            std::cout << "  ones: " << n_ones << "  isolated (ref): " << ref << "\n";

            uint8_t* d_grid = nullptr;
            uint8_t* d_padded = nullptr;
            HIP_CHECK(hipMalloc(&d_grid, grid.size()));
            HIP_CHECK(hipMalloc(&d_padded, padded.size()));
            HIP_CHECK(hipMemcpy(d_grid, grid.data(), grid.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_padded, padded.data(), padded.size(), hipMemcpyHostToDevice));

            for (const auto& spec : all_methods()) {
                if (!method_enabled(method_filter, spec.name)) continue;
                int nblocks = (spec.run == run_m1 || spec.run == run_m1b)
                                  ? 1
                                  : (spec.run == run_m2 || spec.run == run_tile16 ||
                                     spec.run == run_tile16_shfl)
                                        ? nblocks_2d
                                        : nblocks_tile30;
                int got = spec.run(d_grid, d_padded, N, d_scalar, d_block_out, nblocks);
                if (got != ref) {
                    std::cerr << "  MISMATCH " << spec.name << " got=" << got << " ref=" << ref << "\n";
                    return 1;
                }
                if (!quick || N <= 256) {
                    auto stats = benchmark_method(spec, d_grid, d_padded, N, d_scalar, d_block_out,
                                                    nblocks_tile30, nblocks_2d, warmup, repeats);
                    std::cout << "  " << std::left << std::setw(28) << spec.name
                              << "  median " << std::fixed << std::setprecision(2) << stats.median_ms
                              << " ms  mean " << stats.mean_ms << " ms"
                              << "  (std " << stats.std_ms << ")\n";
                } else {
                    std::cout << "  " << std::left << std::setw(28) << spec.name << "  OK\n";
                }
            }

            auto t0 = std::chrono::steady_clock::now();
            int cpu_ref = count_isolated_cpu(grid.data(), N);
            auto t1 = std::chrono::steady_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  CPU reference: " << cpu_ms << " ms (count=" << cpu_ref << ")\n\n";

            HIP_CHECK(hipFree(d_grid));
            HIP_CHECK(hipFree(d_padded));
        }

        HIP_CHECK(hipFree(d_block_out));
    }

    HIP_CHECK(hipFree(d_scalar));
    std::cout << "All checks passed.\n";
    return 0;
}
