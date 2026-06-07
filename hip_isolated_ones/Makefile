# Default: ROCm hipcc on Linux. Override: make HIPCC=/path/to/hipcc ARCH=gfx1030
HIPCC ?= /opt/rocm/bin/hipcc
ARCH ?=
ARCH_FLAG := $(if $(ARCH),--offload-arch=$(ARCH),)

CXXFLAGS = -O3 -std=c++17 -I include $(ARCH_FLAG)
TARGET = hip_isolated_ones
SRC = src/main.cpp

.PHONY: all clean run-quick run-bench

all: $(TARGET)

$(TARGET): $(SRC) include/common.hpp include/kernels.hpp
	$(HIPCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run-quick: $(TARGET)
	./$(TARGET) --quick

run-bench: $(TARGET)
	./$(TARGET) --methods 5
