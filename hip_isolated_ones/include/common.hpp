#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace iso {

constexpr int TPB = 16;
constexpr int TILE = 30;
constexpr int HALO = 32;
constexpr int SM_STRIDE = HALO + 1;
constexpr int SM_HALO = HALO * SM_STRIDE;
constexpr int ROW_STEP = TILE;

constexpr int TILE16 = 16;
constexpr int HALO16 = 18;
constexpr int SM16 = HALO16 * HALO16;

constexpr int BLOCK30 = TILE * TILE;
constexpr int SM_SUM1024 = 1024;

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline int count_isolated_cpu(const uint8_t* grid, int N) {
    int c = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (grid[i * N + j] != 1) continue;
            bool neighbor = false;
            for (int di = -1; di <= 1 && !neighbor; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di, nj = j + dj;
                    if (0 <= ni && ni < N && 0 <= nj && nj < N && grid[ni * N + nj] == 1) {
                        neighbor = true;
                        break;
                    }
                }
            }
            if (!neighbor) ++c;
        }
    }
    return c;
}

inline std::vector<uint8_t> make_grid(int N, float density, uint32_t seed) {
    std::vector<uint8_t> g(static_cast<size_t>(N) * N);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : g) v = dist(rng) < density ? 1 : 0;
    return g;
}

inline std::vector<uint8_t> pad_grid(const uint8_t* grid, int N) {
    int P = N + 2;
    std::vector<uint8_t> p(static_cast<size_t>(P) * P, 0);
    for (int i = 0; i < N; ++i)
        std::memcpy(&p[(i + 1) * P + 1], &grid[i * N], N);
    return p;
}

struct DensityScenario {
    float density;
    const char* label;
};

inline const std::vector<DensityScenario>& default_scenarios() {
    static const std::vector<DensityScenario> s = {
        {0.01f, "sparse"},
        {0.10f, "medium"},
        {0.90f, "dense"},
    };
    return s;
}

}  // namespace iso
