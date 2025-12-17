# Makefile for rdep CUDA extension
#
# Builds rdep.so for Python import
# Requires: CUDA 12.x, sm_100a (B200), pybind11
# Optional: NVSHMEM for multi-node support (auto-detected)
#
# Usage:
#   make          - Build with NVSHMEM if available
#   make info     - Show build configuration
#   make clean    - Remove build artifacts

CUDA_HOME ?= /usr/local/cuda
PYTHON := python3

# NVSHMEM paths (apt package installs to /usr/include/nvshmem_12 and /usr/lib/x86_64-linux-gnu/nvshmem/12)
NVSHMEM_INCLUDE ?= /usr/include/nvshmem_12
NVSHMEM_LIB ?= /usr/lib/x86_64-linux-gnu/nvshmem/12

# Get Python/pybind11 paths
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE := $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())")
PYTHON_EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Auto-detect NVSHMEM
HAS_NVSHMEM := $(shell test -f $(NVSHMEM_INCLUDE)/nvshmem.h && echo 1 || echo 0)

# Compilers
NVCC := $(CUDA_HOME)/bin/nvcc
CXX := g++

# Base flags
NVCC_FLAGS := -std=c++17 -O3 \
    -gencode arch=compute_100a,code=sm_100a \
    -rdc=true \
    -Xcompiler -fPIC \
    --expt-relaxed-constexpr \
    -DNMOE_ENABLE_PTX_E2M1=1 \
    -I$(CUDA_HOME)/include

CXX_FLAGS := -std=c++17 -O3 -fPIC \
    -I$(CUDA_HOME)/include \
    -I$(PYTHON_INCLUDE) \
    -I$(PYBIND11_INCLUDE)

LDFLAGS := -shared -L$(CUDA_HOME)/lib64 -lcublas -lcublasLt -lcudart -lcudadevrt -Wl,-rpath,$(CUDA_HOME)/lib64

# NVSHMEM support (if available)
ifeq ($(HAS_NVSHMEM), 1)
    NVCC_FLAGS += -DWITH_NVSHMEM -I$(NVSHMEM_INCLUDE)
    CXX_FLAGS += -DWITH_NVSHMEM -I$(NVSHMEM_INCLUDE)
    LDFLAGS += -L$(NVSHMEM_LIB) -lnvshmem_host -Wl,-rpath,$(NVSHMEM_LIB)
    # Device library for nvlink (static lib)
    NVSHMEM_DEVICE_LIB := $(NVSHMEM_LIB)/libnvshmem_device.a
    NVSHMEM_SRCS := rdep_nvshmem.cu
    NVSHMEM_STATUS := enabled
else
    NVSHMEM_SRCS :=
    NVSHMEM_DEVICE_LIB :=
    NVSHMEM_STATUS := disabled (NVSHMEM not found at $(NVSHMEM_INCLUDE))
endif

# FlashMLA (required for SM100 MLA backward) and its CUTLASS headers.
FLASHMLA_HOME ?= $(abspath ../../third_party/flashmla)
FLASHMLA_CSRC ?= $(FLASHMLA_HOME)/csrc
CUTLASS_INCLUDE ?= $(FLASHMLA_CSRC)/cutlass/include
CUTLASS_UTIL_INCLUDE ?= $(FLASHMLA_CSRC)/cutlass/tools/util/include
HAS_FLASHMLA := $(shell test -f $(FLASHMLA_CSRC)/sm100/prefill/dense/interface.h && test -f $(CUTLASS_INCLUDE)/cutlass/cutlass.h && echo 1 || echo 0)

ifeq ($(HAS_FLASHMLA), 1)
    FLASHMLA_INCLUDE_FLAGS := -I$(FLASHMLA_CSRC) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL_INCLUDE)
    FLASHMLA_STATUS := enabled (FLASHMLA_HOME=$(FLASHMLA_HOME))
else
    $(error FlashMLA not found. Clone https://github.com/deepseek-ai/FlashMLA into third_party/flashmla (with submodules) or set FLASHMLA_HOME=/path/to/FlashMLA)
endif

# Targets (.so files stay in csrc/, imported as nmoe.csrc.*)
RDEP_TARGET := rdep$(PYTHON_EXT_SUFFIX)
GPU_TARGET := gpu$(PYTHON_EXT_SUFFIX)
# Place muon under opt/ subpackage
OPT_DIR := opt
MUON_TARGET := $(OPT_DIR)/muon$(PYTHON_EXT_SUFFIX)
FLASHMLA_TARGET := flashmla_sm100$(PYTHON_EXT_SUFFIX)

# RDEP sources
RDEP_CUDA_SRCS := rdep.cu quant.cu ptx.cu adamw.cu gemm.cu $(NVSHMEM_SRCS)
# Compile bindings.cpp with BUILD_RDEP into a dedicated object to avoid flag clashes
RDEP_CPP_OBJS := bindings_rdep.o

# GPU telemetry sources (standalone, no CUDA)
GPU_CPP_SRCS := gpu.cpp

# Muon orthogonalizer sources (CUDA + small pybind module)
# Compile bindings.cpp with BUILD_MUON into a dedicated object
MUON_CUDA_SRCS := muon.cu
MUON_OBJS := $(MUON_CUDA_SRCS:.cu=.o) bindings_muon.o

# FlashMLA SM100 dense prefill backward (CUDA)
FLASHMLA_BWD_SRC := $(FLASHMLA_CSRC)/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu
FLASHMLA_BWD_OBJ := flashmla_bwd_sm100.cuda.o
FLASHMLA_CPP_OBJS := flashmla_sm100_bindings.o

TORCH_INCLUDE_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-isystem '+p for p in include_paths()]))")
TORCH_LIB_DIR := $(shell $(PYTHON) -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

FLASHMLA_CXX_FLAGS := -DTORCH_EXTENSION_NAME=flashmla_sm100 -DTORCH_API_INCLUDE_EXTENSION_H \
	-std=c++17 -O3 -fPIC \
	$(FLASHMLA_INCLUDE_FLAGS) \
	$(TORCH_INCLUDE_FLAGS) \
	-isystem $(CUDA_HOME)/include \
	-isystem $(PYTHON_INCLUDE) \
	-I$(PYBIND11_INCLUDE)

FLASHMLA_NVCC_FLAGS := -DTORCH_EXTENSION_NAME=flashmla_sm100 -DTORCH_API_INCLUDE_EXTENSION_H \
	-std=c++17 -O3 --use_fast_math -lineinfo \
	-gencode arch=compute_100a,code=sm_100a \
	--expt-relaxed-constexpr \
	--compiler-options '-fPIC' \
	$(FLASHMLA_INCLUDE_FLAGS) \
	$(TORCH_INCLUDE_FLAGS) \
	-isystem $(CUDA_HOME)/include \
	-isystem $(PYTHON_INCLUDE)

FLASHMLA_LDFLAGS := -shared \
	-L$(TORCH_LIB_DIR) -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python \
	-L$(CUDA_HOME)/lib64 -lcudart \
	-Wl,-rpath,$(TORCH_LIB_DIR) -Wl,-rpath,$(CUDA_HOME)/lib64

# Objects
RDEP_CUDA_OBJS := rdep.o quant.o ptx.o adamw.o gemm.o $(NVSHMEM_SRCS:.cu=.o)
# Explicit binding object for RDEP pybind module
RDEP_CPP_OBJS := bindings_rdep.o
DLINK_OBJ := dlink.o
GPU_OBJS := $(GPU_CPP_SRCS:.cpp=.o)
MUON_DLINK_OBJ := muon_dlink.o

.PHONY: all clean info

all: $(RDEP_TARGET) $(GPU_TARGET) $(MUON_TARGET) $(FLASHMLA_TARGET)

# ========== RDEP module (CUDA + NVSHMEM) ==========

# Device link step (required for -rdc=true)
$(DLINK_OBJ): $(RDEP_CUDA_OBJS)
	$(NVCC) $(NVCC_FLAGS) -dlink -o $@ $^ $(NVSHMEM_DEVICE_LIB) -lcudadevrt

$(RDEP_TARGET): $(RDEP_CUDA_OBJS) $(DLINK_OBJ) $(RDEP_CPP_OBJS)
	$(CXX) -o $@ $^ $(NVSHMEM_DEVICE_LIB) $(LDFLAGS)
	@echo ""
	@echo "Built $(RDEP_TARGET)"
	@echo "  NVSHMEM: $(NVSHMEM_STATUS)"
	@echo ""

# ========== GPU telemetry module (NVML only, no CUDA) ==========

$(GPU_TARGET): $(GPU_OBJS)
	$(CXX) -o $@ $^ -shared -lnvidia-ml \
		-L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64
	@echo ""
	@echo "Built $(GPU_TARGET)"
	@echo ""

# ========== MUON module (CUDA) ==========

$(MUON_DLINK_OBJ): muon.o
	$(NVCC) $(NVCC_FLAGS) -dlink -o $@ $^ -lcudadevrt

$(MUON_TARGET): $(MUON_OBJS) $(MUON_DLINK_OBJ)
	@mkdir -p $(OPT_DIR)
	$(CXX) -o $@ $^ -shared \
		-L$(CUDA_HOME)/lib64 -lcublas -lcublasLt -lcudart -lcudadevrt -Wl,-rpath,$(CUDA_HOME)/lib64
	@echo ""
	@echo "Built $(MUON_TARGET)"
	@echo ""

# ========== FlashMLA SM100 dense prefill bwd (CUDA) ==========

$(FLASHMLA_TARGET): $(FLASHMLA_BWD_OBJ) $(FLASHMLA_CPP_OBJS)
	$(CXX) -o $@ $^ $(FLASHMLA_LDFLAGS)
	@echo ""
	@echo "Built $(FLASHMLA_TARGET)"
	@echo ""

rdep.o: rdep.cu ptx.cu rdep_nvshmem.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

quant.o: quant.cu ptx.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

ptx.o: ptx.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

rdep_nvshmem.o: rdep_nvshmem.cu rdep_nvshmem.cuh ptx.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(FLASHMLA_BWD_OBJ): $(FLASHMLA_BWD_SRC)
	$(NVCC) $(FLASHMLA_NVCC_FLAGS) -c $< -o $@

flashmla_sm100_bindings.o: flashmla_sm100_bindings.cpp
	$(CXX) $(FLASHMLA_CXX_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

bindings_rdep.o: bindings.cpp
	$(CXX) $(CXX_FLAGS) -DBUILD_RDEP -c $< -o $@

bindings_muon.o: bindings.cpp
	$(CXX) $(CXX_FLAGS) -DBUILD_MUON -c $< -o $@

clean:
	rm -f $(RDEP_CUDA_OBJS) $(DLINK_OBJ) $(RDEP_CPP_OBJS) $(RDEP_TARGET) \
		$(GPU_OBJS) $(GPU_TARGET) \
		$(MUON_TARGET) $(MUON_CUDA_SRCS:.cu=.o) bindings_rdep.o bindings_muon.o $(MUON_DLINK_OBJ) \
		$(FLASHMLA_BWD_OBJ) $(FLASHMLA_CPP_OBJS) $(FLASHMLA_TARGET)

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: all

# Print config
info:
	@echo "nmoe C++ Extensions Build Configuration"
	@echo "========================================"
	@echo "CUDA_HOME:        $(CUDA_HOME)"
	@echo "NVSHMEM_INCLUDE:  $(NVSHMEM_INCLUDE)"
	@echo "NVSHMEM_LIB:      $(NVSHMEM_LIB)"
	@echo "NVSHMEM:          $(NVSHMEM_STATUS)"
	@echo "FlashMLA:         $(FLASHMLA_STATUS)"
	@echo "PYTHON:           $(PYTHON)"
	@echo "PYTHON_INCLUDE:   $(PYTHON_INCLUDE)"
	@echo "PYBIND11_INCLUDE: $(PYBIND11_INCLUDE)"
	@echo ""
	@echo "Targets:"
	@echo "  RDEP:  $(RDEP_TARGET)"
	@echo "  GPU:   $(GPU_TARGET)"
	@echo "  MUON:  $(MUON_TARGET)"
	@echo "  FlashMLA: $(FLASHMLA_TARGET)"
	@echo ""
	@echo "RDEP Sources:"
	@echo "  CUDA: $(RDEP_CUDA_SRCS)"
	@echo "  CPP:  bindings.cpp (compiled as bindings_rdep.o with BUILD_RDEP)"
	@echo ""
	@echo "GPU Sources:"
	@echo "  CPP:  $(GPU_CPP_SRCS)"
