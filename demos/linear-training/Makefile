NVCC ?= nvcc

ifndef GPU
GPU = H100
endif

TARGET = linear_training
SRC    = linear_training.cu

ifndef PYTHON_VERSION
PYTHON_VERSION = 3.12
endif
PYTHON_BIN    ?= python${PYTHON_VERSION}
PYTHON_CONFIG ?= python${PYTHON_VERSION}-config

# {opt, debug, quick}
MODE ?= opt

ENABLE_TIMINGS ?= 0
ENABLE_DEBUG ?= 0

# Torch include/lib paths from the active Python (use venv python if available)
TORCH_INCLUDES = $(shell ${PYTHON_BIN} -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
TORCH_LIB_DIR  = $(shell ${PYTHON_BIN} -c "import torch; from torch.utils.cpp_extension import library_paths; paths = library_paths(); print(paths[0] if paths else '')")
TORCH_LIBS     = -L${TORCH_LIB_DIR} -Wl,-rpath,${TORCH_LIB_DIR}

###############################################################################
# Common low-cost flags
###############################################################################
NVCCFLAGS_COMMON = \
	-Xcompiler=-fPIE \
	--expt-extended-lambda \
	--expt-relaxed-constexpr \
	-Xcompiler=-Wno-psabi \
	-Xcompiler=-fno-strict-aliasing \
	-forward-unknown-to-host-compiler \
	-std=c++20 -x cu \
	-lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static \
	-I${THUNDERKITTENS_ROOT}/include \
	-I${THUNDERKITTENS_ROOT}/prototype \
	-I${MEGAKERNELS_ROOT}/include \
	$(TORCH_INCLUDES) $(TORCH_LIBS) \
	$(shell ${PYTHON_BIN} -m pybind11 --includes) \
	$(shell ${PYTHON_CONFIG} --ldflags) \
	-shared -fPIC -Wl,--no-as-needed \
	--threads=0 \
	-lpython${PYTHON_VERSION} -ltorch -ltorch_python -lc10

###############################################################################
# Mode-specific flags
###############################################################################

# 1. default optimize flags
NVCCFLAGS_OPT = \
	-DNDEBUG \
	-O3 \
	--use_fast_math \
	-Xnvlink=--verbose \
	-Xptxas=--verbose \
	-Xptxas=--warn-on-spills \
	-lineinfo

# 2. debug builds
NVCCFLAGS_DEBUG = \
	-O0 \
	-g -G \
	-lineinfo \
	-Xptxas=-O0

# 3. impatient builds
NVCCFLAGS_QUICK = \
	-O1 \
	-g \
	-Xptxas=-O1 \
	-rdc=false

###############################################################################
# Mode switch
###############################################################################
ifeq ($(MODE),opt)
    NVCCFLAGS_MODE = $(NVCCFLAGS_OPT)
else ifeq ($(MODE),debug)
    NVCCFLAGS_MODE = $(NVCCFLAGS_DEBUG)
else ifeq ($(MODE),quick)
    NVCCFLAGS_MODE = $(NVCCFLAGS_QUICK)
else
    $(error Invalid MODE: $(MODE). Expected one of {opt, debug, quick})
endif

NVCCFLAGS = $(NVCCFLAGS_COMMON) $(NVCCFLAGS_MODE)
NVCCFLAGS += -DLINEAR_DEMO_TIMING_ENABLED=$(ENABLE_TIMINGS) -DLINEAR_DEMO_DEBUG_ENABLED=$(ENABLE_DEBUG)

###############################################################################
# GPU architecture selection
###############################################################################
ifeq ($(GPU),4090)
	NVCCFLAGS += -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
	NVCCFLAGS += -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
	NVCCFLAGS += -DKITTENS_HOPPER -arch=sm_90a
else
	NVCCFLAGS += -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

###############################################################################
# Build targets
###############################################################################

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell ${PYTHON_CONFIG} --extension-suffix)

clean:
	rm -f $(TARGET)$(shell ${PYTHON_CONFIG} --extension-suffix)

# Convenience shortcuts
opt:
	$(MAKE) MODE=opt

debug:
	$(MAKE) MODE=debug

quick:
	$(MAKE) MODE=quick
