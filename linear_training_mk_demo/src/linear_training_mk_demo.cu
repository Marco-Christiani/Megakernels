#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "config.cuh"
#include "kittens.cuh"
#include "megakernel.cuh"
#include "pyutils/torch_helpers.cuh"

#include <iostream>
#include <vector>
#include <cstdio>

constexpr int IN_DIM = 128;
constexpr int OUT_DIM = 64;

constexpr int ROW_TILE = 16;
constexpr int COL_TILE = 32;


using config = linear_training_mk_demo_config;
using state_t = megakernel::state<config>;

template <typename Config, typename Globals>
struct linear_fwd_pipeline {
    using state_t = megakernel::state<Config>;

    // Single logical page reserved for weights.
    static constexpr int WEIGHTS_LID = 0;
    static constexpr int SEM_WEIGHTS_READY = 0;
    static constexpr int SEM_COUNT = 1;

    __device__ static int weights_pid(state_t &s) {
        return s.pid(WEIGHTS_LID);
    }

    __device__ static kittens::semaphore &weights_ready(state_t &s) {
        return s.semaphores()[SEM_WEIGHTS_READY];
    }

    __device__ static kittens::st_fl<ROW_TILE, COL_TILE> &weights_tile(state_t &s) {
        // Use the VM page as backing storage for the tile.
        return *reinterpret_cast<kittens::st_fl<ROW_TILE, COL_TILE> *>(
            s.pages[weights_pid(s)].ptr()
        );
    }

    static __device__ int init_semaphores(state_t &s) {
        if (kittens::laneid() == 0) {
            kittens::init_semaphore(weights_ready(s), 1);
        }
        return SEM_COUNT;
    }

    // Loader:
    //  - wait for the weights page
    //  - fill the 16x32 tile
    //  - signal consumers via a single semaphore (parity fixed at 0)
    static __device__ void loader_fill(state_t &s, const Globals &g) {
        const int lane = kittens::laneid();
        const auto &inst = s.instruction();
        const int row_block = inst[2];
        const int col_block = inst[3];
        const int pid = weights_pid(s);

        if (lane == 0) {
            printf("[linear-training loader] worker=%u inst=%d waiting for weights page pid=%d row_block=%d col_block=%d\n",
                   (unsigned)megakernel::get_worker_id(),
                   s.instruction_index,
                   pid,
                   row_block,
                   col_block);
        }

        s.wait_page_ready(pid);
        __syncwarp();

        auto &tile = weights_tile(s);

        // Fill the 16x32 tile from global weights.
        for (int linear = lane; linear < ROW_TILE * COL_TILE; linear += kittens::WARP_THREADS) {
            int tile_row = linear / COL_TILE;
            int tile_col = linear % COL_TILE;
            int row      = row_block * ROW_TILE + tile_row;
            int col      = col_block * COL_TILE + tile_col;

            float val = 0.f;
            if (row < OUT_DIM && col < IN_DIM) {
                val = g.weights[{0, 0, row, col}];
            }
            tile[make_int2(tile_row, tile_col)] = val;
        }
        __syncwarp();  // ensure the tile is fully populated before consumers read it

        if (kittens::laneid() == 0) {
            kittens::arrive(weights_ready(s));
            printf("[linear-training loader] worker=%u inst=%d arrived weights_ready pid=%d\n",
                   (unsigned)megakernel::get_worker_id(),
                   s.instruction_index,
                   pid);
        }
    }

    // Called once per warp after it finishes using the weights tile.
    static __device__ void release_weights_page(state_t &s) {
        const int lane = kittens::laneid();
        if (lane == 0) {
            // One arrival per warp. Config::NUM_CONSUMER_WARPS arrivals total.
            int weights_pid_val = weights_pid(s);
            s.warp_finish_page(weights_pid_val, 1);
            printf("[linear-training release] worker=%u inst=%d warp=%d released weights pid=%d\n",
                   (unsigned)megakernel::get_worker_id(),
                   s.instruction_index,
                   kittens::warpid(),
                   weights_pid_val);
        }
    }
};


struct linear_training_globals {
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;
    using input_t = kittens::gl<float, -1, 1, 1, IN_DIM>;
    using target_t = kittens::gl<float, -1, 1, 1, OUT_DIM>;
    using output_t = kittens::gl<float, -1, 1, 1, OUT_DIM>;
    using grad_out_t = kittens::gl<float, -1, 1, 1, OUT_DIM>;
    using weights_t = kittens::gl<float, 1, 1, OUT_DIM, IN_DIM>;
    using grad_w_t = kittens::gl<float, 1, 1, OUT_DIM, IN_DIM>;

    instruction_layout instructions;
    timing_layout timings;
    input_t input;
    target_t target;
    output_t output;
    grad_out_t grad_out;
    weights_t weights;
    grad_w_t grad_w;
    float lr;
    int batch_size;
    int debug_vis;

    dim3 grid() const { return dim3(1); }
    dim3 block() const { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() const { return config::DYNAMIC_SHARED_MEMORY; }
};

template <typename C = config>
struct LinearFwd {
    static constexpr int opcode = 1;
    using pipeline = linear_fwd_pipeline<C, linear_training_globals>;

    // Design note: this op uses a single logical page (WEIGHTS_LID) and a single
    // dynamic semaphore (weights_ready, parity fixed at 0). The loader warp alone
    // waits for the weights page, fills the shared tile, and arrives the semaphore.
    // All NUM_CONSUMER_WARPS consumer warps wait on that semaphore before reading
    // the tile and each calls warp_finish_page once, no block-wide syncs exist in
    // the loader path, so only the participating warp ever stalls.
    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            // Simple round-robin logical ID assignment, as before.
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            pipeline::loader_fill(s, g);
        }
    };

    struct launcher {
        static __device__ void run(const linear_training_globals &, state_t &s) {
#ifdef KITTENS_BLACKWELL
            if (kittens::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, C::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer {
        static __device__ void run(const linear_training_globals &, state_t &) {}
    };

    struct consumer {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto &inst = s.instruction();
            const int out_block = inst[2];
            const int in_block  = inst[3];

            const int wid  = kittens::warpid();    // 0..15
            const int lane = kittens::laneid();    // 0..31

            const int row = ROW_TILE * out_block + wid;
            const int col = COL_TILE * in_block  + lane;

            if (lane == 0) {
                printf("[linear-training fwd-consumer] worker=%u inst=%d wid=%d row=%d col_block=%d active=%d\n",
                       (unsigned)megakernel::get_worker_id(),
                       s.instruction_index,
                       wid,
                       row,
                       in_block,
                       (int)(row < OUT_DIM));
            }

            if (g.debug_vis && lane == 0 && s.instruction_index == 0 && wid == 0) {
                printf("[linear-training-mk-demo] worker_id=%u opcode=%d instruction_rows=%d\n",
                       (unsigned)megakernel::get_worker_id(),
                       inst[0],
                       (int)g.instructions.rows());
            }

            kittens::wait(pipeline::weights_ready(s), 0);
            __syncwarp();

            auto &weights_tile = pipeline::weights_tile(s);

            // Some warps may not correspond to a valid output row.
            const bool active = (row < OUT_DIM);

            if (active) {
                float w_lane = weights_tile[make_int2(wid, lane)];

                for (int b = 0; b < g.batch_size; ++b) {
                    float x_val = g.input[{b, 0, 0, col}];
                    float partial = x_val * w_lane;

                    float dot = partial;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        dot += __shfl_down_sync(0xffffffff, dot, offset);
                    }

                    if (lane == 0) {
                        printf("[linear-training fwd-consumer] worker=%u inst=%d wid=%d batch=%d partial=%f\n",
                               (unsigned)megakernel::get_worker_id(),
                               s.instruction_index,
                               wid,
                               b,
                               dot);
                        float old = (in_block == 0) ? 0.0f : g.output[{b, 0, 0, row}];
                        g.output[{b, 0, 0, row}] = old + dot;
                    }
                }
            }

            // Every warp must contribute to releasing the weights page.
            pipeline::release_weights_page(s);
        }
    };
};

template <typename C = config> struct LossGrad {
    static constexpr int opcode = 2;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &) { return 0; }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &, state_t &s) {
            const int lane = kittens::laneid();
            if (lane >= 1 && lane < C::NUM_PAGES) {
                int pid = s.pid(lane);
                s.wait_page_ready(pid);
                s.finish_page(pid, C::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_training_globals &, state_t &s) {
#ifdef KITTENS_BLACKWELL
            if (kittens::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, C::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_training_globals &, state_t &) {} };

    struct consumer {
        // LossGrad
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto& inst = s.instruction();
            const int row_block = inst[2];
            const int wid  = kittens::warpid();   // row 0..15
            const float scale = 2.f / float(g.batch_size);
            const int lane = kittens::laneid();

            if (lane == 0) {
                printf("[linear-training lossgrad] worker=%u inst=%d row_block=%d wid=%d\n",
                       (unsigned)megakernel::get_worker_id(),
                       s.instruction_index,
                       row_block,
                       wid);
            }

            if (wid >= ROW_TILE) return;
            const int row = row_block * ROW_TILE + wid;

            for (int b = 0; b < g.batch_size; ++b) {
                float out   = g.output[{b,0,0,row}];
                float tgt   = g.target[{b,0,0,row}];
                g.grad_out[{b,0,0,row}] = (out - tgt) * scale;
                if (lane == 0) {
                    printf("[linear-training lossgrad] worker=%u inst=%d wid=%d batch=%d grad=%f\n",
                           (unsigned)megakernel::get_worker_id(),
                           s.instruction_index,
                           wid,
                           b,
                           g.grad_out[{b,0,0,row}]);
                }
            }
        }

    };
};

template <typename C = config> struct LinearBwdWeight {
    static constexpr int opcode = 3;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &) { return 0; }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &, state_t &s) {
            const int lane = kittens::laneid();
            if (lane >= 1 && lane < C::NUM_PAGES) {
                int pid = s.pid(lane);
                s.wait_page_ready(pid);
                s.finish_page(pid, C::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_training_globals &, state_t &s) {
#ifdef KITTENS_BLACKWELL
            if (kittens::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, C::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_training_globals &, state_t &) {} };

    struct consumer {
        // LinearBwdWeight
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto& inst = s.instruction();
            const int row_block = inst[2];
            const int col_block = inst[3];

            const int wid  = kittens::warpid();   // row
            const int lane = kittens::laneid();   // col
            if (wid >= ROW_TILE || lane >= COL_TILE) return;

            const int row = row_block * ROW_TILE + wid;
            const int col = col_block * COL_TILE + lane;

            if (lane == 0) {
                printf("[linear-training bwd-weight] worker=%u inst=%d row_block=%d col_block=%d wid=%d row=%d\n",
                       (unsigned)megakernel::get_worker_id(),
                       s.instruction_index,
                       row_block,
                       col_block,
                       wid,
                       row);
            }

            float acc = 0.f;

            for (int b = 0; b < g.batch_size; ++b) {
                float go = g.grad_out[{b,0,0,row}]; // grad wrt output (scalar per row)
                float x  = g.input[{b,0,0,col}];    // one input element
                acc += go * x;
                if (lane == 0) {
                    printf("[linear-training bwd-weight] worker=%u inst=%d wid=%d batch=%d partial=%f\n",
                           (unsigned)megakernel::get_worker_id(),
                           s.instruction_index,
                           wid,
                           b,
                           acc);
                }
            }

            g.grad_w[{0,0,row,col}] = acc / float(g.batch_size);
            if (lane == 0) {
                printf("[linear-training bwd-weight] worker=%u inst=%d wid=%d row=%d col=%d grad=%f\n",
                       (unsigned)megakernel::get_worker_id(),
                       s.instruction_index,
                       wid,
                       row,
                       col,
                       g.grad_w[{0,0,row,col}]);
            }
        }
    };
};

template <typename C = config> struct SgdUpdate {
    static constexpr int opcode = 4;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &) { return 0; }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &, state_t &s) {
            const int lane = kittens::laneid();
            if (lane >= 1 && lane < C::NUM_PAGES) {
                int pid = s.pid(lane);
                s.wait_page_ready(pid);
                s.finish_page(pid, C::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_training_globals &, state_t &s) {
#ifdef KITTENS_BLACKWELL
            if (kittens::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, C::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_training_globals &, state_t &) {} };

    struct consumer {
        // SgdUpdate
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto& inst = s.instruction();
            const int row_block = inst[2];
            const int col_block = inst[3];

            const int wid  = kittens::warpid();
            const int lane = kittens::laneid();
            if (wid >= ROW_TILE || lane >= COL_TILE) return;

            const int row = row_block * ROW_TILE + wid;
            const int col = col_block * COL_TILE + lane;

            if (lane == 0) {
                printf("[linear-training sgd] worker=%u inst=%d row=%d col_block=%d wid=%d\n",
                       (unsigned)megakernel::get_worker_id(),
                       s.instruction_index,
                       row,
                       col_block,
                       wid);
            }

            float w   = g.weights[{0,0,row,col}];
            float grad= g.grad_w[{0,0,row,col}];

            g.weights[{0,0,row,col}] = w - g.lr * grad;
            if (lane == 0) {
                printf("[linear-training sgd] worker=%u inst=%d row=%d col=%d w_old=%f grad=%f w_new=%f\n",
                       (unsigned)megakernel::get_worker_id(),
                       s.instruction_index,
                       row,
                       col,
                       w,
                       grad,
                       g.weights[{0,0,row,col}]);
            }
        }
    };
};

std::vector<torch::Tensor> run_linear_training(torch::Tensor instructions_tensor,
                                               torch::Tensor input,
                                               torch::Tensor target,
                                               torch::Tensor weights,
                                               double lr,
                                               bool debug_vis) {
    CHECK_INPUT(instructions_tensor);
    CHECK_INPUT(input);
    CHECK_INPUT(target);
    CHECK_INPUT(weights);

    TORCH_CHECK(instructions_tensor.scalar_type() == torch::kInt32, "instructions must be int32");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "input must be float32");
    TORCH_CHECK(target.scalar_type() == torch::kFloat, "target must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be float32");
    TORCH_CHECK(instructions_tensor.dim() == 3, "instructions must be [sm_count, num_instructions, 32]");
    TORCH_CHECK(instructions_tensor.size(2) == config::INSTRUCTION_WIDTH,
                "expected instruction width ", config::INSTRUCTION_WIDTH);
    TORCH_CHECK(input.dim() == 2 && input.size(1) == IN_DIM, "input must be [batch, ", IN_DIM, "]");
    TORCH_CHECK(target.dim() == 2 && target.size(1) == OUT_DIM, "target must be [batch, ", OUT_DIM, "]");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == OUT_DIM && weights.size(1) == IN_DIM,
                "weights must be [", OUT_DIM, ", ", IN_DIM, "]");

    auto batch_size = static_cast<int>(input.size(0));

    auto output = torch::zeros({batch_size, OUT_DIM}, input.options());
    auto grad_out = torch::zeros_like(output);
    auto grad_w = torch::zeros_like(weights);
    auto timings = torch::zeros(
        {instructions_tensor.size(0), instructions_tensor.size(1), config::TIMING_WIDTH},
        torch::TensorOptions().dtype(torch::kInt32).device(instructions_tensor.device()));

    typename linear_training_globals::instruction_layout inst_layout =
        kittens::make_gl<typename linear_training_globals::instruction_layout>(
            reinterpret_cast<uint64_t>(instructions_tensor.data_ptr<int>()),
            1,
            static_cast<int>(instructions_tensor.size(0)),
            static_cast<int>(instructions_tensor.size(1)),
            static_cast<int>(instructions_tensor.size(2)));

    typename linear_training_globals::timing_layout timing_layout =
        kittens::make_gl<typename linear_training_globals::timing_layout>(
            reinterpret_cast<uint64_t>(timings.data_ptr<int>()),
            1,
            static_cast<int>(timings.size(0)),
            static_cast<int>(timings.size(1)),
            static_cast<int>(timings.size(2)));

    typename linear_training_globals::input_t input_layout =
        kittens::make_gl<typename linear_training_globals::input_t>(
            reinterpret_cast<uint64_t>(input.data_ptr<float>()),
            batch_size,
            1,
            1,
            IN_DIM);

    typename linear_training_globals::target_t target_layout =
        kittens::make_gl<typename linear_training_globals::target_t>(
            reinterpret_cast<uint64_t>(target.data_ptr<float>()),
            batch_size,
            1,
            1,
            OUT_DIM);

    typename linear_training_globals::output_t output_layout =
        kittens::make_gl<typename linear_training_globals::output_t>(
            reinterpret_cast<uint64_t>(output.data_ptr<float>()),
            batch_size,
            1,
            1,
            OUT_DIM);

    typename linear_training_globals::grad_out_t grad_out_layout =
        kittens::make_gl<typename linear_training_globals::grad_out_t>(
            reinterpret_cast<uint64_t>(grad_out.data_ptr<float>()),
            batch_size,
            1,
            1,
            OUT_DIM);

    typename linear_training_globals::weights_t weights_layout =
        kittens::make_gl<typename linear_training_globals::weights_t>(
            reinterpret_cast<uint64_t>(weights.data_ptr<float>()),
            1,
            1,
            OUT_DIM,
            IN_DIM);

    typename linear_training_globals::grad_w_t grad_w_layout =
        kittens::make_gl<typename linear_training_globals::grad_w_t>(
            reinterpret_cast<uint64_t>(grad_w.data_ptr<float>()),
            1,
            1,
            OUT_DIM,
            IN_DIM);

    linear_training_globals g{
        inst_layout,
        timing_layout,
        input_layout,
        target_layout,
        output_layout,
        grad_out_layout,
        weights_layout,
        grad_w_layout,
        static_cast<float>(lr),
        batch_size,
        debug_vis ? 1 : 0};

    std::cout << "[linear-training host] launching kernel with batch_size=" << batch_size
              << " lr=" << lr
              << " debug=" << (debug_vis ? "true" : "false") << std::endl;

    megakernel::mk<config,
                   linear_training_globals,
                   LinearFwd<config>,
                   LossGrad<config>,
                   LinearBwdWeight<config>,
                   SgdUpdate<config>><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    cudaError_t err = cudaDeviceSynchronize();
    std::cout << "[linear-training host] kernel completed err=" << cudaGetErrorString(err) << std::endl;
    TORCH_CHECK(err == cudaSuccess, "linear training kernel launch failed: ", cudaGetErrorString(err));

    return {output, grad_out, grad_w, timings};
}

PYBIND11_MODULE(linear_training_mk_demo, m) {
    m.def(
        "run",
        &run_linear_training,
        "Run linear training megakernel",
        pybind11::arg("instructions"),
        pybind11::arg("input"),
        pybind11::arg("target"),
        pybind11::arg("weights"),
        pybind11::arg("lr"),
        pybind11::arg("debug") = false);
}
