#pragma once
#include "kittens.cuh"

namespace megakernel {

struct linear_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];

    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // Warps and threads
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS     = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS   = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS    = 1;
    static constexpr int CLUSTER_BLOCKS = 1;

    // Keep total shared memory under the default per-block limit (~48 KB) to avoid invalid launch.
    static constexpr int MAX_SHARED_MEMORY = 49'152;

    // Same formula used by default_config
    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
              (SCRATCH_BYTES +
               (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 +
               DYNAMIC_SEMAPHORES * 8);

    // Remaining shared memory = page allocator pool
    static constexpr int DYNAMIC_SHARED_MEMORY =
        MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 16'384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES >= 1, "NUM_PAGES must be >= 1");

    #ifndef LINEAR_DEMO_TIMING_ENABLED
    #define LINEAR_DEMO_TIMING_ENABLED 0
    #endif

    #ifndef LINEAR_DEMO_DEBUG_ENABLED
    #define LINEAR_DEMO_DEBUG_ENABLED 0
    #endif

    static constexpr bool TIMING_RECORD_ENABLED = LINEAR_DEMO_TIMING_ENABLED != 0;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

// Re-export the layouts
template <typename C>
using instruction_layout = kittens::gl<int, 1, -1, -1, C::INSTRUCTION_WIDTH>;

template <typename C>
using timing_layout = kittens::gl<int, 1, -1, -1, C::TIMING_WIDTH>;

} // namespace megakernel
