#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "config.cuh"
#include "kittens.cuh"
#include "megakernel.cuh"
#include "pyutils/torch_helpers.cuh"

#include <iostream>
#include <vector>

// Tile geometry for all MMA paths
constexpr int M_TILE = 16;
constexpr int N_TILE = 16;
constexpr int K_TILE = 16;

using activ_tile_t = kittens::st_fl<M_TILE, K_TILE>;
using weight_tile_fwd_t = kittens::st_fl<N_TILE, K_TILE>;   // weights for forward (N x K)
using weight_tile_bwd_t = kittens::st_fl<K_TILE, N_TILE>;   // weights/activations for K x N tiles
using output_frag_t = kittens::rt_fl<M_TILE, N_TILE>;
using activ_frag_t = kittens::rt_bf<M_TILE, K_TILE>;
using weight_frag_fwd_t = kittens::rt_bf<N_TILE, K_TILE>;
using weight_frag_row_t = kittens::rt_bf<K_TILE, N_TILE>;
using weight_frag_col_t = kittens::rt_bf<K_TILE, N_TILE, kittens::ducks::rt_layout::col>;

using config = linear_training_mk_demo_config;
using state_t = megakernel::state<config>;

enum OpCode : int {
    LINEAR_FWD = 1,
    LOSS_GRAD = 2,
    LINEAR_BWD_WEIGHT = 3,
    LINEAR_BWD_INPUT = 4,
    SGD_UPDATE = 5
};

__host__ __device__ constexpr int div_up(int x, int y) {
    return (x + y - 1) / y;
}

struct linear_training_globals {
    using instruction_layout = megakernel::instruction_layout<config>;
    using timing_layout = megakernel::timing_layout<config>;

    // activations / grad_activations: (layer, 1, batch, dim)
    using activations_t = kittens::gl<float, -1, 1, -1, -1, activ_tile_t, weight_tile_bwd_t>;
    // weights / grad_w: (layer, 1, out_dim, in_dim)
    using weights_t = kittens::gl<float, -1, 1, -1, -1, weight_tile_fwd_t, weight_tile_bwd_t>;
    using grad_w_t = kittens::gl<float, -1, 1, -1, -1>;
    // targets: (1, 1, batch, dim)
    using targets_t = kittens::gl<float, 1, 1, -1, -1>;
    // dims arrays: (1, 1, 1, num_layers)
    using dims_t = kittens::gl<int, 1, 1, 1, -1>;

    instruction_layout instructions;
    timing_layout timings;
    activations_t activations;
    activations_t grad_activations;
    weights_t weights;
    grad_w_t grad_w;
    targets_t targets;
    dims_t layer_in_dims;
    dims_t layer_out_dims;
    float lr;
    int batch_size;
    int padded_batch;
    int max_dim;
    int num_layers;
    int debug_vis;

    dim3 grid() const { return dim3(1); }
    dim3 block() const { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() const { return config::DYNAMIC_SHARED_MEMORY; }
};

template <typename Config, typename Globals>
struct two_page_pipeline {
    using state_t   = megakernel::state<Config>;
    using tile_a_t  = activ_tile_t;
    using tile_b_t  = weight_tile_fwd_t;

    // Two pages per operand (ping–pong)
    static constexpr int A0_LID = 0;
    static constexpr int A1_LID = 1;
    static constexpr int B0_LID = 2;
    static constexpr int B1_LID = 3;

    // Two pipeline stages (stage == parity)
    static constexpr int INPUT_PIPELINE_STAGES = 2;

    // Semaphore layout:
    //   0,1 : A arrived [stage 0/1]
    //   2,3 : B arrived [stage 0/1]
    //   4,5 : A finished [stage 0/1]
    //   6,7 : B finished [stage 0/1]
    static constexpr int SEM_A_ARR_0 = 0;
    static constexpr int SEM_A_ARR_1 = 1;
    static constexpr int SEM_B_ARR_0 = 2;
    static constexpr int SEM_B_ARR_1 = 3;
    static constexpr int SEM_A_FIN_0 = 4;
    static constexpr int SEM_A_FIN_1 = 5;
    static constexpr int SEM_B_FIN_0 = 6;
    static constexpr int SEM_B_FIN_1 = 7;
    static constexpr int SEM_COUNT   = 8;

    // --- page helpers -----------------------------------------------------
    __device__ static int a_pid(state_t &s, int stage) {
        return s.pid(stage ? A1_LID : A0_LID);
    }
    __device__ static int b_pid(state_t &s, int stage) {
        return s.pid(stage ? B1_LID : B0_LID);
    }

    __device__ static tile_a_t &a_tile(state_t &s, int stage) {
        return *reinterpret_cast<tile_a_t *>(s.pages[a_pid(s, stage)].ptr());
    }
    __device__ static tile_b_t &b_tile(state_t &s, int stage) {
        return *reinterpret_cast<tile_b_t *>(s.pages[b_pid(s, stage)].ptr());
    }

    // --- semaphore helpers ------------------------------------------------
    __device__ static kittens::semaphore &a_arrived(state_t &s, int stage) {
        return s.semaphores()[stage ? SEM_A_ARR_1 : SEM_A_ARR_0];
    }
    __device__ static kittens::semaphore &b_arrived(state_t &s, int stage) {
        return s.semaphores()[stage ? SEM_B_ARR_1 : SEM_B_ARR_0];
    }
    __device__ static kittens::semaphore &a_finished(state_t &s, int stage) {
        return s.semaphores()[stage ? SEM_A_FIN_1 : SEM_A_FIN_0];
    }
    __device__ static kittens::semaphore &b_finished(state_t &s, int stage) {
        return s.semaphores()[stage ? SEM_B_FIN_1 : SEM_B_FIN_0];
    }

    static __device__ int init_semaphores(state_t &s) {
        if (kittens::laneid() == 0) {
            // ready / arrived semaphores start with 1 token (TK pattern)
            kittens::init_semaphore(a_arrived(s, 0), 1);
            kittens::init_semaphore(a_arrived(s, 1), 1);
            kittens::init_semaphore(b_arrived(s, 0), 1);
            kittens::init_semaphore(b_arrived(s, 1), 1);

            // finished semaphores throttle the loader (one credit per consumer warp)
            kittens::init_semaphore(a_finished(s, 0), Config::NUM_CONSUMER_WARPS);
            kittens::init_semaphore(a_finished(s, 1), Config::NUM_CONSUMER_WARPS);
            kittens::init_semaphore(b_finished(s, 0), Config::NUM_CONSUMER_WARPS);
            kittens::init_semaphore(b_finished(s, 1), Config::NUM_CONSUMER_WARPS);
        }
        return SEM_COUNT;
    }

    // Bit pattern copied from matvec_pipeline:
    //  - loader waits on *_finished with bit = (iter % (2*STAGES)) < STAGES
    //  - consumer waits on *_arrived with bit = (iter % (2*STAGES)) >= STAGES
    __device__ static int loader_wait_bit(int iter) {
        return (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES;
    }
    __device__ static int consumer_wait_bit(int iter) {
        return (iter % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES;
    }

    // We no longer use global wait/release; handled per-tile
    static __device__ void wait_pages(state_t &) {}
    static __device__ void release_pages(state_t &) {}
};

__device__ inline int get_in_dim(const linear_training_globals &g, int layer) {
    return g.layer_in_dims[{0, 0, 0, layer}];
}

__device__ inline int get_out_dim(const linear_training_globals &g, int layer) {
    return g.layer_out_dims[{0, 0, 0, layer}];
}

template <typename C = config>
struct LinearFwd {
    static constexpr int opcode = OpCode::LINEAR_FWD;
    using pipeline = two_page_pipeline<C, linear_training_globals>;
    using state_t  = megakernel::state<C>;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }

        static __device__ int release_lid(
            const linear_training_globals &, typename C::instruction_t &, int &query
        ) {
            if (query == 0) return pipeline::A0_LID;
            if (query == 1) return pipeline::A1_LID;
            if (query == 2) return pipeline::B0_LID;
            if (query == 3) return pipeline::B1_LID;
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const int lane   = kittens::laneid();
            const auto &inst = s.instruction();
            const int layer  = inst[1];
            const int m_block = inst[2];
            const int n_block = inst[3];

            const int in_dim  = get_in_dim(g, layer);
            const int k_tiles = div_up(in_dim, K_TILE);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int stage       = kt % pipeline::INPUT_PIPELINE_STAGES;
                const int k_start     = kt * K_TILE;
                const int loader_bit  = pipeline::loader_wait_bit(kt);

                if (lane == 0) {
                    // Throttle: don’t reuse stage until consumer marks finished.
                    kittens::wait(pipeline::a_finished(s, stage), loader_bit);
                    kittens::wait(pipeline::b_finished(s, stage), loader_bit);

                    // TK-style page credit: only during initial pipeline fill.
                    if (kt < pipeline::INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(pipeline::a_pid(s, stage));
                        s.wait_page_ready(pipeline::b_pid(s, stage));
                    }

                    const size_t bytes_a = sizeof(float) * M_TILE * K_TILE;
                    const size_t bytes_b = sizeof(float) * N_TILE * K_TILE;

                    kittens::tma::expect_bytes(pipeline::a_arrived(s, stage), bytes_a);
                    kittens::tma::expect_bytes(pipeline::b_arrived(s, stage), bytes_b);

                    kittens::tma::load_async(
                        pipeline::a_tile(s, stage),
                        g.activations,
                        kittens::coord<>{layer, 0, m_block * M_TILE, k_start},
                        pipeline::a_arrived(s, stage)
                    );

                    kittens::tma::load_async(
                        pipeline::b_tile(s, stage),
                        g.weights,
                        kittens::coord<>{layer, 0, n_block * N_TILE, k_start},
                        pipeline::b_arrived(s, stage)
                    );

                    if (g.debug_vis) {
                        printf("[fwd-loader] layer=%d m_blk=%d n_blk=%d kt=%d stage=%d pidA=%d pidB=%d\n",
                               layer, m_block, n_block, kt, stage,
                               pipeline::a_pid(s, stage), pipeline::b_pid(s, stage));
                    }
                }

                __syncwarp();
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

    struct storer {
        static __device__ void run(const linear_training_globals &, state_t &) {}
    };

    struct consumer {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto &inst = s.instruction();
            const int layer   = inst[1];
            const int m_block = inst[2];
            const int n_block = inst[3];

            const int m_start = m_block * M_TILE;
            const int n_start = n_block * N_TILE;
            const int in_dim  = get_in_dim(g, layer);
            const int k_tiles = div_up(in_dim, K_TILE);

            output_frag_t c_frag;
            kittens::warp::zero(c_frag);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int stage        = kt % pipeline::INPUT_PIPELINE_STAGES;
                const int consumer_bit = pipeline::consumer_wait_bit(kt);

                // Wait for TMA to finish for this stage.
                kittens::wait(pipeline::a_arrived(s, stage), consumer_bit);
                kittens::wait(pipeline::b_arrived(s, stage), consumer_bit);
                __syncwarp();

                activ_frag_t        a_frag;
                weight_frag_fwd_t   b_frag;
                kittens::warp::load(a_frag, pipeline::a_tile(s, stage));
                kittens::warp::load(b_frag, pipeline::b_tile(s, stage));
                kittens::warp::mma_ABt(c_frag, a_frag, b_frag, c_frag);
                __syncwarp();

                // Tell loader this stage is finished and return page credit.
                kittens::warp::arrive(pipeline::a_finished(s, stage));
                kittens::warp::arrive(pipeline::b_finished(s, stage));

                // TK-style page release: only in final pipeline iterations.
                if (kt >= k_tiles - pipeline::INPUT_PIPELINE_STAGES) {
                    if (kittens::laneid() == 0) {
                        s.warp_finish_page(pipeline::a_pid(s, stage), 1);
                        s.warp_finish_page(pipeline::b_pid(s, stage), 1);
                    }
                }
            }

            // Store entire tile (padding ensures safe bounds).
            kittens::warp::store(
                g.activations,
                c_frag,
                kittens::coord<>{layer + 1, 0, m_start, n_start}
            );

            if (g.debug_vis && kittens::laneid() == 0 && kittens::warpid() == 0) {
                printf("[fwd-consumer] layer=%d m_blk=%d n_blk=%d wrote tile\n",
                       layer, m_block, n_block);
            }
        }
    };
};

template <typename C = config>
struct LossGrad {
    static constexpr int opcode = OpCode::LOSS_GRAD;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &) { return 0; }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &, state_t &) {}
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
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto &inst = s.instruction();
            const int layer = inst[1]; // final layer index
            const int m_block = inst[2];
            const int n_block = inst[3];

            const int out_dim = get_out_dim(g, layer);
            const int row_start = m_block * M_TILE;
            const int col_start = n_block * N_TILE;
            const float scale = 2.f / float(g.batch_size);

            for (int linear = kittens::laneid(); linear < M_TILE * N_TILE; linear += kittens::WARP_THREADS) {
                const int mi = linear / N_TILE;
                const int ni = linear % N_TILE;
                const int row = row_start + mi;
                const int col = col_start + ni;
                if (row < g.batch_size && col < out_dim) {
                    const float out_val = g.activations[{layer + 1, 0, row, col}];
                    const float tgt = g.targets[{0, 0, row, col}];
                    const float grad = (out_val - tgt) * scale;
                    g.grad_activations[{layer + 1, 0, row, col}] = grad;
                }
            }

            if (g.debug_vis && kittens::laneid() == 0 && kittens::warpid() == 0) {
                printf("[lossgrad] layer=%d m_blk=%d n_blk=%d\n", layer, m_block, n_block);
            }
        }
    };
};

template <typename C = config>
struct LinearBwdWeight {
    static constexpr int opcode = OpCode::LINEAR_BWD_WEIGHT;
    using pipeline = two_page_pipeline<C, linear_training_globals>;
    using state_t  = megakernel::state<C>;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }

        static __device__ int release_lid(
            const linear_training_globals &, typename C::instruction_t &, int &query
        ) {
            if (query == 0) return pipeline::A0_LID;
            if (query == 1) return pipeline::A1_LID;
            if (query == 2) return pipeline::B0_LID;
            if (query == 3) return pipeline::B1_LID;
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const int lane   = kittens::laneid();
            const auto &inst = s.instruction();
            const int layer     = inst[1];
            const int out_block = inst[2];
            const int in_block  = inst[3];

            const int batch_tiles = div_up(g.batch_size, K_TILE);

            for (int kt = 0; kt < batch_tiles; ++kt) {
                const int stage       = kt % pipeline::INPUT_PIPELINE_STAGES;
                const int batch_start = kt * K_TILE;
                const int loader_bit  = pipeline::loader_wait_bit(kt);

                if (lane == 0) {
                    kittens::wait(pipeline::a_finished(s, stage), loader_bit);
                    kittens::wait(pipeline::b_finished(s, stage), loader_bit);

                    if (kt < pipeline::INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(pipeline::a_pid(s, stage));
                        s.wait_page_ready(pipeline::b_pid(s, stage));
                    }

                    const size_t bytes = sizeof(float) * K_TILE * N_TILE;

                    kittens::tma::expect_bytes(pipeline::a_arrived(s, stage), bytes);
                    kittens::tma::expect_bytes(pipeline::b_arrived(s, stage), bytes);

                    // A tile: grad_out chunk (batch, out)
                    kittens::tma::load_async(
                        reinterpret_cast<weight_tile_bwd_t &>(pipeline::a_tile(s, stage)),
                        g.grad_activations,
                        kittens::coord<>{layer + 1, 0, batch_start, out_block * N_TILE},
                        pipeline::a_arrived(s, stage)
                    );

                    // B tile: activations chunk (batch, in)
                    kittens::tma::load_async(
                        reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s, stage)),
                        g.activations,
                        kittens::coord<>{layer, 0, batch_start, in_block * N_TILE},
                        pipeline::b_arrived(s, stage)
                    );

                    if (g.debug_vis) {
                        printf("[bwd-w loader] layer=%d out_blk=%d in_blk=%d kt=%d stage=%d pidA=%d pidB=%d\n",
                               layer, out_block, in_block, kt, stage,
                               pipeline::a_pid(s, stage), pipeline::b_pid(s, stage));
                    }
                }

                __syncwarp();
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

    struct storer {
        static __device__ void run(const linear_training_globals &, state_t &) {}
    };

    struct consumer {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto &inst = s.instruction();
            const int layer     = inst[1];
            const int out_block = inst[2];
            const int in_block  = inst[3];

            const int batch_tiles = div_up(g.batch_size, K_TILE);

            output_frag_t grad_tile;
            kittens::warp::zero(grad_tile);

            for (int kt = 0; kt < batch_tiles; ++kt) {
                const int stage         = kt % pipeline::INPUT_PIPELINE_STAGES;
                const int consumer_bit = pipeline::consumer_wait_bit(kt);

                kittens::wait(pipeline::a_arrived(s, stage), consumer_bit);
                kittens::wait(pipeline::b_arrived(s, stage), consumer_bit);
                __syncwarp();

                weight_frag_row_t grad_out_row;
                weight_frag_row_t activ_row;
                weight_frag_col_t grad_out_frag;
                weight_frag_col_t activ_frag;

                kittens::warp::load(
                    grad_out_row,
                    reinterpret_cast<weight_tile_bwd_t &>(pipeline::a_tile(s, stage))
                );
                kittens::warp::load(
                    activ_row,
                    reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s, stage))
                );
                kittens::warp::swap_layout(grad_out_frag, grad_out_row);
                kittens::warp::swap_layout(activ_frag, activ_row);

                kittens::warp::mma_AtB(grad_tile, grad_out_frag, activ_frag, grad_tile);
                __syncwarp();

                kittens::warp::arrive(pipeline::a_finished(s, stage));
                kittens::warp::arrive(pipeline::b_finished(s, stage));

                if (kt >= batch_tiles - pipeline::INPUT_PIPELINE_STAGES) {
                    if (kittens::laneid() == 0) {
                        s.warp_finish_page(pipeline::a_pid(s, stage), 1);
                        s.warp_finish_page(pipeline::b_pid(s, stage), 1);
                    }
                }
            }

            const float scale = 1.f / float(g.batch_size);
            kittens::warp::mul(grad_tile, grad_tile, scale);
            kittens::warp::store(
                g.grad_w,
                grad_tile,
                kittens::coord<>{layer, 0, out_block * N_TILE, in_block * N_TILE}
            );
        }
    };
};

template <typename C = config>
struct LinearBwdInput {
    static constexpr int opcode = OpCode::LINEAR_BWD_INPUT;
    using pipeline = two_page_pipeline<C, linear_training_globals>;
    using state_t  = megakernel::state<C>;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }

        static __device__ int release_lid(
            const linear_training_globals &, typename C::instruction_t &, int &query
        ) {
            if (query == 0) return pipeline::A0_LID;
            if (query == 1) return pipeline::A1_LID;
            if (query == 2) return pipeline::B0_LID;
            if (query == 3) return pipeline::B1_LID;
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const int lane   = kittens::laneid();
            const auto &inst = s.instruction();
            const int layer   = inst[1];
            const int m_block = inst[2]; // batch tile
            const int n_block = inst[3]; // input tile

            const int out_dim  = get_out_dim(g, layer);
            const int k_tiles  = div_up(out_dim, K_TILE);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int stage       = kt % pipeline::INPUT_PIPELINE_STAGES;
                const int k_start     = kt * K_TILE;
                const int loader_bit  = pipeline::loader_wait_bit(kt);

                if (lane == 0) {
                    kittens::wait(pipeline::a_finished(s, stage), loader_bit);
                    kittens::wait(pipeline::b_finished(s, stage), loader_bit);

                    if (kt < pipeline::INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(pipeline::a_pid(s, stage));
                        s.wait_page_ready(pipeline::b_pid(s, stage));
                    }

                    const size_t bytes_a = sizeof(float) * M_TILE * K_TILE;
                    const size_t bytes_b = sizeof(float) * N_TILE * K_TILE;

                    kittens::tma::expect_bytes(pipeline::a_arrived(s, stage), bytes_a);
                    kittens::tma::expect_bytes(pipeline::b_arrived(s, stage), bytes_b);

                    // A tile: grad_out (batch, out_dim)
                    kittens::tma::load_async(
                        pipeline::a_tile(s, stage),
                        g.grad_activations,
                        kittens::coord<>{layer + 1, 0, m_block * M_TILE, k_start},
                        pipeline::a_arrived(s, stage)
                    );

                    // B tile: weights (out_dim, in_dim)
                    kittens::tma::load_async(
                        reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s, stage)),
                        g.weights,
                        kittens::coord<>{layer, 0, k_start, n_block * N_TILE},
                        pipeline::b_arrived(s, stage)
                    );

                    if (g.debug_vis) {
                        printf("[bwd-in loader] layer=%d m_blk=%d n_blk=%d kt=%d stage=%d pidA=%d pidB=%d\n",
                               layer, m_block, n_block, kt, stage,
                               pipeline::a_pid(s, stage), pipeline::b_pid(s, stage));
                    }
                }

                __syncwarp();
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

    struct storer {
        static __device__ void run(const linear_training_globals &, state_t &) {}
    };

    struct consumer {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const auto &inst = s.instruction();
            const int layer   = inst[1];
            const int m_block = inst[2];
            const int n_block = inst[3];

            const int out_dim  = get_out_dim(g, layer);
            const int k_tiles  = div_up(out_dim, K_TILE);

            output_frag_t grad_frag;
            kittens::warp::zero(grad_frag);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int stage         = kt % pipeline::INPUT_PIPELINE_STAGES;
                const int consumer_bit  = pipeline::consumer_wait_bit(kt);

                kittens::wait(pipeline::a_arrived(s, stage), consumer_bit);
                kittens::wait(pipeline::b_arrived(s, stage), consumer_bit);
                __syncwarp();

                activ_frag_t       grad_out_frag;
                weight_frag_col_t  w_col;

                // grad_out tile: (M_TILE x K_TILE), row layout
                kittens::warp::load(grad_out_frag, pipeline::a_tile(s, stage));
                // weights tile: [K_TILE x N_TILE], column layout.
                kittens::warp::load(
                    w_col,
                    reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s, stage))
                );

                // grad_in = grad_out @ W, where W is (out_dim x in_dim) and this tile is (K_TILE x N_TILE)
                kittens::warp::mma_AB(grad_frag, grad_out_frag, w_col, grad_frag);
                __syncwarp();

                kittens::warp::arrive(pipeline::a_finished(s, stage));
                kittens::warp::arrive(pipeline::b_finished(s, stage));

                if (kt >= k_tiles - pipeline::INPUT_PIPELINE_STAGES) {
                    if (kittens::laneid() == 0) {
                        s.warp_finish_page(pipeline::a_pid(s, stage), 1);
                        s.warp_finish_page(pipeline::b_pid(s, stage), 1);
                    }
                }
            }

            kittens::warp::store(
                g.grad_activations,
                grad_frag,
                kittens::coord<>{layer, 0, m_block * M_TILE, n_block * N_TILE}
            );
        }
    };
};


template <typename C = config>
struct SgdUpdate {
    static constexpr int opcode = OpCode::SGD_UPDATE;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &) { return 0; }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &, state_t &) {}
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
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            // Only a single consumer warp should apply sgd updates.
            // Other consumer warps are effectively no-ops for this instruction.
            // TODO: think about this?
            if (kittens::warpid() != 0) {
                return;
            }

            const auto &inst = s.instruction();
            const int layer = inst[1];
            const int out_block = inst[2];
            const int in_block = inst[3];

            const int out_dim = get_out_dim(g, layer);
            const int in_dim = get_in_dim(g, layer);
            const int row_start = out_block * N_TILE;
            const int col_start = in_block * N_TILE;

            for (int linear = kittens::laneid(); linear < M_TILE * N_TILE; linear += kittens::WARP_THREADS) {
                const int oi = linear / N_TILE;
                const int ii = linear % N_TILE;
                const int row = row_start + oi;
                const int col = col_start + ii;
                if (row < out_dim && col < in_dim) {
                    float w = g.weights[{layer, 0, row, col}];
                    float grad = g.grad_w[{layer, 0, row, col}];
                    float w_new = w - g.lr * grad;
                    g.weights[{layer, 0, row, col}] = w_new;

                    // Single element to check sgd math against host: layer 0, row=1, col=2
                    // Not gating on laneid==0, bc different lanes handle different (row, col) pairs
                    if (g.debug_vis && layer == 0 && row == 1 && col == 2) {
                        printf(
                            "[sgd-debug] layer=%d row=%d col=%d w_before=%f grad=%f lr=%f w_after=%f\n",
                            layer, row, col, w, grad, g.lr, w_new
                        );
                    }
                }
            }

            if (g.debug_vis && kittens::laneid() == 0 && kittens::warpid() == 0) {
                printf("[sgd] layer=%d out_blk=%d in_blk=%d\n", layer, out_block, in_block);
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
    TORCH_CHECK(instructions_tensor.dim() == 3, "instructions must be (sm_count, num_instructions, 32)");
    TORCH_CHECK(instructions_tensor.size(2) == config::INSTRUCTION_WIDTH,
                "expected instruction width ", config::INSTRUCTION_WIDTH);

    TORCH_CHECK(weights.dim() == 3, "weights must be (layers, out_dim, in_dim)");
    const int num_layers = weights.size(0);
    TORCH_CHECK(num_layers > 0, "weights must contain at least one layer");

    std::vector<int> layer_in_dims(num_layers);
    std::vector<int> layer_out_dims(num_layers);
    int max_in_dim = 0;
    int max_out_dim = 0;
    for (int l = 0; l < num_layers; ++l) {
        layer_out_dims[l] = static_cast<int>(weights.size(1));
        layer_in_dims[l] = static_cast<int>(weights.size(2));
        max_in_dim = std::max(max_in_dim, layer_in_dims[l]);
        max_out_dim = std::max(max_out_dim, layer_out_dims[l]);
    }

    TORCH_CHECK(input.dim() == 2 && input.size(1) == layer_in_dims.front(),
                "input must be [batch, in_dim0]");
    TORCH_CHECK(target.dim() == 2 && target.size(1) == layer_out_dims.back(),
                "target must be [batch, out_dim_last]");

    const int batch_size = static_cast<int>(input.size(0));
    const int padded_batch = div_up(batch_size, M_TILE) * M_TILE;
    const int padded_in_dim = div_up(max_in_dim, K_TILE) * K_TILE;
    const int padded_out_dim = div_up(max_out_dim, N_TILE) * N_TILE;
    const int max_dim = std::max(padded_in_dim, padded_out_dim);

    // padded activations/grad buffers
    auto activations = torch::zeros({num_layers + 1, 1, padded_batch, max_dim}, input.options());
    auto grad_activations = torch::zeros_like(activations);

    // copy input into activation[0]
    activations.index_put_({0, 0, torch::indexing::Slice(0, batch_size), torch::indexing::Slice(0, layer_in_dims.front())}, input);

    // padded weights/grad_w
    auto weights_padded = torch::zeros({num_layers, 1, padded_out_dim, padded_in_dim}, weights.options());
    weights_padded.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(0, 1),
         torch::indexing::Slice(0, layer_out_dims.back()),
         torch::indexing::Slice(0, layer_in_dims.front())},
        weights.unsqueeze(1));
    auto grad_w = torch::zeros_like(weights_padded);

    // padded targets
    auto targets = torch::zeros({1, 1, padded_batch, max_dim}, target.options());
    targets.index_put_({0, 0, torch::indexing::Slice(0, batch_size), torch::indexing::Slice(0, layer_out_dims.back())}, target);

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

    typename linear_training_globals::activations_t activ_layout =
        kittens::make_gl<typename linear_training_globals::activations_t>(
            reinterpret_cast<uint64_t>(activations.data_ptr<float>()),
            num_layers + 1,
            1,
            padded_batch,
            max_dim);

    typename linear_training_globals::activations_t grad_activ_layout =
        kittens::make_gl<typename linear_training_globals::activations_t>(
            reinterpret_cast<uint64_t>(grad_activations.data_ptr<float>()),
            num_layers + 1,
            1,
            padded_batch,
            max_dim);

    typename linear_training_globals::weights_t weights_layout =
        kittens::make_gl<typename linear_training_globals::weights_t>(
            reinterpret_cast<uint64_t>(weights_padded.data_ptr<float>()),
            num_layers,
            1,
            padded_out_dim,
            padded_in_dim);

    typename linear_training_globals::grad_w_t grad_w_layout =
        kittens::make_gl<typename linear_training_globals::grad_w_t>(
            reinterpret_cast<uint64_t>(grad_w.data_ptr<float>()),
            num_layers,
            1,
            padded_out_dim,
            padded_in_dim);

    typename linear_training_globals::targets_t targets_layout =
        kittens::make_gl<typename linear_training_globals::targets_t>(
            reinterpret_cast<uint64_t>(targets.data_ptr<float>()),
            1,
            1,
            padded_batch,
            max_dim);

    auto in_dim_tensor = torch::tensor(layer_in_dims, torch::TensorOptions().dtype(torch::kInt32).device(weights.device()));
    auto out_dim_tensor = torch::tensor(layer_out_dims, torch::TensorOptions().dtype(torch::kInt32).device(weights.device()));

    typename linear_training_globals::dims_t in_dims_layout =
        kittens::make_gl<typename linear_training_globals::dims_t>(
            reinterpret_cast<uint64_t>(in_dim_tensor.data_ptr<int>()),
            1,
            1,
            1,
            num_layers);

    typename linear_training_globals::dims_t out_dims_layout =
        kittens::make_gl<typename linear_training_globals::dims_t>(
            reinterpret_cast<uint64_t>(out_dim_tensor.data_ptr<int>()),
            1,
            1,
            1,
            num_layers);

    linear_training_globals g{
        inst_layout,
        timing_layout,
        activ_layout,
        grad_activ_layout,
        weights_layout,
        grad_w_layout,
        targets_layout,
        in_dims_layout,
        out_dims_layout,
        static_cast<float>(lr),
        batch_size,
        padded_batch,
        max_dim,
        num_layers,
        debug_vis ? 1 : 0};

    std::cout << "[linear-training host] launching kernel with batch_size=" << batch_size
              << " lr=" << lr
              << " layers=" << num_layers
              << " debug=" << (debug_vis ? "true" : "false") << std::endl;

    megakernel::mk<config,
                   linear_training_globals,
                   LinearFwd<config>,
                   LossGrad<config>,
                   LinearBwdWeight<config>,
                   LinearBwdInput<config>,
                   SgdUpdate<config>><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    cudaError_t err = cudaDeviceSynchronize();
    std::cout << "[linear-training host] kernel completed err=" << cudaGetErrorString(err) << std::endl;
    TORCH_CHECK(err == cudaSuccess, "linear training kernel launch failed: ", cudaGetErrorString(err));

    // slice outputs to the user-facing shapes
    auto output = activations.index({num_layers, 0, torch::indexing::Slice(0, batch_size),
                                     torch::indexing::Slice(0, layer_out_dims.back())}).contiguous();
    auto grad_out = grad_activations.index({num_layers, 0, torch::indexing::Slice(0, batch_size),
                                            torch::indexing::Slice(0, layer_out_dims.back())}).contiguous();
    auto grad_input = grad_activations.index({0, 0, torch::indexing::Slice(0, batch_size),
                                              torch::indexing::Slice(0, layer_in_dims.front())}).contiguous();
    auto grad_w_view = grad_w.index({torch::indexing::Slice(), 0,
                                     torch::indexing::Slice(0, layer_out_dims.back()),
                                     torch::indexing::Slice(0, layer_in_dims.front())}).contiguous();
    auto weights_view = weights_padded.index({torch::indexing::Slice(), 0,
                                              torch::indexing::Slice(0, layer_out_dims.back()),
                                              torch::indexing::Slice(0, layer_in_dims.front())}).contiguous();

    // write back updated weights to user tensor
    weights.copy_(weights_view);

    return {output, grad_out, grad_input, grad_w_view, weights_view, timings};
}

PYBIND11_MODULE(linear_training_mk_demo, m) {
    m.def(
        "run",
        &run_linear_training,
        "Run MMA-based linear training megakernel",
        pybind11::arg("instructions"),
        pybind11::arg("input"),
        pybind11::arg("target"),
        pybind11::arg("weights"),
        pybind11::arg("lr"),
        pybind11::arg("debug") = false);
}
