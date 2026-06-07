cmake_minimum_required(VERSION 3.21)
project(hip_isolated_ones LANGUAGES CXX)

# ROCm / HIP — set HIP_PATH or install rocm-dev on Linux
if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(DEFINED ENV{HIPCC})
        set(CMAKE_CXX_COMPILER "$ENV{HIPCC}")
    elseif(EXISTS "/opt/rocm/bin/hipcc")
        set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")
    endif()
endif()

find_program(HIPCC hipcc)
if(HIPCC)
    message(STATUS "Using hipcc: ${HIPCC}")
    set(CMAKE_CXX_COMPILER "${HIPCC}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hip_isolated_ones src/main.cpp)
target_include_directories(hip_isolated_ones PRIVATE include)

# hipcc links HIP runtime automatically; explicit link for cmake+ninja builds
find_package(hip QUIET CONFIG PATHS /opt/rocm /opt/rocm/hip)
if(hip_FOUND)
    target_link_libraries(hip_isolated_ones PRIVATE hip::host)
else()
    target_link_libraries(hip_isolated_ones PRIVATE amdhip64)
endif()

target_compile_options(hip_isolated_ones PRIVATE -O3 -ffast-math)

# Optional: target a specific AMD GPU arch, e.g. gfx1030 (RDNA2), gfx1100 (RDNA3), gfx942 (MI300)
# target_compile_options(hip_isolated_ones PRIVATE --offload-arch=gfx1030)

install(TARGETS hip_isolated_ones RUNTIME DESTINATION bin)
