# C++ GPU Scheduler Implementation

## Overview

This directory contains a high-performance C++ implementation of GPU instruction scheduling algorithms, ported from the Python implementation in the parent directory. The C++ version provides significant performance improvements (typically 20-50x faster) while maintaining exact functional equivalence with the Python implementation.

## Architecture

### File Structure

```
cpp_src/
├── globs.hpp              # Global configuration struct (mirrors Python globs.py)
├── instructions.hpp       # Instruction class hierarchy (mirrors Python instructions.py)
├── scheduling.hpp         # Scheduling function declarations
├── scheduling.cpp         # Scheduling algorithm implementations (mirrors Python scheduler.py)
├── bindings.cpp          # Python bindings using pybind11
└── CLAUDE.md             # This documentation
```

### Key Components

1. **Globals Configuration (`globs.hpp`)**
   - Contains model parameters, batch sizes, device configuration
   - Mirrors the Python `Globals` class structure
   - Includes derived properties like `local_batch_size()`, `num_batch_blocks()`, etc.

2. **Instruction Hierarchy (`instructions.hpp`)**
   - Base `Instruction` class with virtual methods for opcode, serialization
   - `ComputeInstruction` and `MemoryInstruction` base classes
   - Specific instruction types:
     - Normalization: `AttnNorm`, `MLP_Norm`, `LM_Head_Norm`
     - Matrix operations: `QKV_RopeAppend`, `GateSilu`, `UpMatMul`, `DownProjResidual`, etc.
     - Attention: `AttentionDecode`, `AttentionPrefill`
     - Control: `IncBarrier`

3. **Scheduling Algorithms (`scheduling.hpp/cpp`)**
   - `create_instruction_tensor()`: Single device tensor creation
   - `create_all_instruction_tensors()`: Multi-device parallel tensor creation
   - Layer scheduling, norm scheduling, matmul scheduling
   - Wave interleaving for overlapping compute/memory operations
   - SM (Streaming Multiprocessor) assignment strategies

## Recent Updates (January 2025)

### Major Additions

1. **IncBarrier Instruction**
   - Added new `IncBarrier` instruction type (opcode 12)
   - Inserted at the start of model execution
   - Matches Python implementation

2. **Multi-Device Support**
   - Added `create_all_instruction_tensors()` function
   - Parallel tensor creation for all devices using `std::async`
   - Significant performance improvements for multi-GPU setups

3. **Modular Loading**
   - Moved C++ extension loading to separate `cpp_scheduler.py` module
   - Prevents automatic compilation when importing from tp_throughput
   - Optional loading pattern for better development workflow

4. **Globals Reorganization**
   - Moved `Globals` struct to separate `globs.hpp` file
   - Matches Python structure where Globals is in `globs.py`
   - Better code organization and separation of concerns

5. **Bug Fixes**
   - Fixed `local_batch_size()` and `local_batch_blocks()` to accept device_idx parameter
   - Fixed attention decode scheduling to handle partial groups
   - Updated schedule_norm functions to pass device_idx correctly

## API Reference

### Main Functions

#### create_instruction_tensor()
```cpp
torch::Tensor create_instruction_tensor(
    const Globals& globs,
    int device_idx,
    int layer_limit = -1,
    bool interleave_waves = false,
    int interleave_buffer_size = -1,
    bool move_to_gpu = true,
    const std::string& stop_after_op = "",
    bool zero_init = true,
    int num_threads = 1
);
```

Creates instruction tensor for a single device.

#### create_all_instruction_tensors()
```cpp
std::vector<torch::Tensor> create_all_instruction_tensors(
    const Globals& globs,
    int layer_limit = -1,
    bool interleave_waves = false,
    int interleave_buffer_size = -1,
    bool move_to_gpu = true,
    const std::string& stop_after_op = "",
    bool zero_init = true,
    int num_threads = 1
);
```

Creates instruction tensors for all devices in parallel.

## Usage

### Python Import Pattern

The C++ scheduler is now in a separate submodule to avoid automatic loading:

```python
# Old way (automatic loading) - NO LONGER WORKS
# from megakernels.demos.tp_throughput import scheduler_cpp

# New way (explicit loading)
from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp

# Or check availability first
from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp, is_available

if is_available():
    result = scheduler_cpp.create_instruction_tensor(...)
else:
    # Fall back to Python implementation
    pass
```

### Environment Variables

- `USE_CPP_SCHEDULER`: Set to "0" to disable C++ scheduler (default: "1")

## Performance

### Benchmarks

Typical performance improvements over Python implementation:

| Configuration | Speedup | Notes |
|--------------|---------|-------|
| Single device, 1 thread | ~20-25x | Basic speedup from C++ |
| Single device, 8 threads | ~40-50x | Multi-threaded serialization |
| All devices, 1 thread | ~20x | Near-linear scaling |
| All devices, 8 threads | ~40x | Parallel device + serialization |

### Multi-Threading

The implementation uses multi-threading in several areas:

1. **Layer Scheduling**: Parallel scheduling across transformer layers
2. **Instruction Serialization**: Parallel conversion to tensor format
3. **Multi-Device Creation**: Parallel tensor creation across devices

## Testing

### Test Files

- `test_cpp_scheduling.py`: Comprehensive correctness tests
  - Tests both single device and multi-device functions
  - Parameter sweeps across different configurations
  - Verifies exact match with Python implementation

- `bench_cpp_scheduling.py`: Performance benchmarking
  - Supports `all_gpu` flag for multi-device benchmarking
  - Measures speedup vs Python implementation
  - Includes correctness verification

### Running Tests

```bash
# Run all tests
python test_cpp_scheduling.py

# Run benchmarks (single device)
python bench_cpp_scheduling.py

# Run benchmarks (all devices)
python bench_cpp_scheduling.py all_gpu=True num_threads=8
```

## Implementation Details

### Memory Management

- Uses `std::unique_ptr` for automatic memory management
- Raw arrays (`new int[]`) for performance-critical sections
- Move semantics to avoid unnecessary copies

### Serialization

- Instructions serialize to fixed-size arrays (32 integers)
- Zero-padding handled automatically
- Efficient batch serialization with optional multi-threading

### Wave Interleaving

Complex algorithm for overlapping compute and memory operations:
- Partitions instruction waves based on buffer sizes
- Interleaves waves of different types (compute vs memory)
- Maintains correctness while improving GPU utilization

## Building

The C++ extension is built automatically using PyTorch's JIT compilation:
- No manual build steps required
- Caches compiled extension for fast subsequent imports
- Automatically rebuilds when source files change

Requirements:
- C++17 compatible compiler
- PyTorch with C++ API
- pybind11 (automatically installed)
- ninja build system (recommended)

## Future Improvements

1. **Further Optimizations**
   - SIMD instructions for serialization
   - Better cache locality
   - GPU-side instruction generation

2. **Extended Functionality**
   - More instruction types
   - Dynamic scheduling strategies
   - Profiling integration

3. **Robustness**
   - Better error handling
   - Compile-time configuration options
   - Cross-platform support improvements