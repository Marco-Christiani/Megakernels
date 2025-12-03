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
    using tile_a_t = activ_tile_t;
    using tile_b_t = weight_tile_fwd_t;

    static constexpr int A_LID = 0;
    static constexpr int B_LID = 1;
    static constexpr int SEM_A_READY = 0;
    static constexpr int SEM_B_READY = 1;
    static constexpr int SEM_COUNT = 2;

    __device__ static int a_pid(state_t &s) { return s.pid(A_LID); }
    __device__ static int b_pid(state_t &s) { return s.pid(B_LID); }

    __device__ static tile_a_t &a_tile(state_t &s) {
        return *reinterpret_cast<tile_a_t *>(s.pages[a_pid(s)].ptr());
    }
    __device__ static tile_b_t &b_tile(state_t &s) {
        return *reinterpret_cast<tile_b_t *>(s.pages[b_pid(s)].ptr());
    }

    __device__ static kittens::semaphore &a_ready(state_t &s) {
        return s.semaphores()[SEM_A_READY];
    }
    __device__ static kittens::semaphore &b_ready(state_t &s) {
        return s.semaphores()[SEM_B_READY];
    }

    static __device__ int init_semaphores(state_t &s) {
        if (kittens::laneid() == 0) {
            kittens::init_semaphore(a_ready(s), 1);
            kittens::init_semaphore(b_ready(s), 1);
        }
        return SEM_COUNT;
    }

    static __device__ void wait_pages(state_t &s) {
        s.wait_page_ready(a_pid(s));
        s.wait_page_ready(b_pid(s));
        __syncwarp();
    }

    static __device__ void release_pages(state_t &s) {
        if (kittens::laneid() == 0) {
            s.warp_finish_page(a_pid(s), 1);
            s.warp_finish_page(b_pid(s), 1);
        }
    }
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

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            if (query == 0) return pipeline::A_LID;
            if (query == 1) return pipeline::B_LID;
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const int lane = kittens::laneid();
            const auto &inst = s.instruction();
            const int layer = inst[1];
            const int m_block = inst[2];
            const int n_block = inst[3];

            pipeline::wait_pages(s);

            const int in_dim = get_in_dim(g, layer);
            const int k_tiles = div_up(in_dim, K_TILE);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int parity = kt & 1;
                const int k_start = kt * K_TILE;

                if (lane == 0) {
                    kittens::tma::expect_bytes(pipeline::a_ready(s), sizeof(float) * M_TILE * K_TILE);
                    kittens::tma::expect_bytes(pipeline::b_ready(s), sizeof(float) * N_TILE * K_TILE);

                    kittens::tma::load_async(
                        pipeline::a_tile(s),
                        g.activations,
                        kittens::coord<>{layer, 0, m_block * M_TILE, k_start},
                        pipeline::a_ready(s));

                    kittens::tma::load_async(
                        pipeline::b_tile(s),
                        g.weights,
                        kittens::coord<>{layer, 0, n_block * N_TILE, k_start},
                        pipeline::b_ready(s));

                    if (g.debug_vis) {
                        printf("[fwd-loader] layer=%d m_blk=%d n_blk=%d kt=%d pidA=%d pidB=%d\n",
                               layer, m_block, n_block, kt, pipeline::a_pid(s), pipeline::b_pid(s));
                    }
                }

                // make sure data is visible before consumers advance
                kittens::wait(pipeline::a_ready(s), parity);
                kittens::wait(pipeline::b_ready(s), parity);
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
            const int layer = inst[1];
            const int m_block = inst[2];
            const int n_block = inst[3];

            const int m_start = m_block * M_TILE;
            const int n_start = n_block * N_TILE;
            const int in_dim = get_in_dim(g, layer);
            const int out_dim = get_out_dim(g, layer);
            const int k_tiles = div_up(in_dim, K_TILE);

            output_frag_t c_frag;
            kittens::warp::zero(c_frag);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int parity = kt & 1;
                kittens::wait(pipeline::a_ready(s), parity);
                kittens::wait(pipeline::b_ready(s), parity);
                __syncwarp();

                activ_frag_t a_frag;
                weight_frag_fwd_t b_frag;
                kittens::warp::load(a_frag, pipeline::a_tile(s));
                kittens::warp::load(b_frag, pipeline::b_tile(s));
                kittens::warp::mma_ABt(c_frag, a_frag, b_frag, c_frag);

                __syncwarp();
            }

            // store full tile, padding guarantees bounds
            kittens::warp::store(g.activations, c_frag, kittens::coord<>{layer + 1, 0, m_start, n_start});

            if (g.debug_vis && kittens::laneid() == 0 && kittens::warpid() == 0) {
                printf("[fwd-consumer] layer=%d m_blk=%d n_blk=%d wrote tile\n", layer, m_block, n_block);
            }

            pipeline::release_pages(s);
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

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            if (query == 0) return pipeline::A_LID;
            if (query == 1) return pipeline::B_LID;
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const int lane = kittens::laneid();
            const auto &inst = s.instruction();
            const int layer = inst[1];
            const int out_block = inst[2];
            const int in_block = inst[3];

            pipeline::wait_pages(s);

            const int batch_tiles = div_up(g.batch_size, K_TILE);
            for (int kt = 0; kt < batch_tiles; ++kt) {
                const int parity = kt & 1;
                const int batch_start = kt * K_TILE;

                if (lane == 0) {
                    kittens::tma::expect_bytes(pipeline::a_ready(s), sizeof(float) * K_TILE * N_TILE);
                    kittens::tma::expect_bytes(pipeline::b_ready(s), sizeof(float) * K_TILE * N_TILE);

                    // A tile: grad_out chunk (batch, out)
                    kittens::tma::load_async(
                        reinterpret_cast<weight_tile_bwd_t &>(pipeline::a_tile(s)),
                        g.grad_activations,
                        kittens::coord<>{layer + 1, 0, batch_start, out_block * N_TILE},
                        pipeline::a_ready(s));

                    // B tile: activations chunk (batch, in)
                    kittens::tma::load_async(
                        reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s)),
                        g.activations,
                        kittens::coord<>{layer, 0, batch_start, in_block * N_TILE},
                        pipeline::b_ready(s));

                    if (g.debug_vis) {
                        printf("[bwd-w loader] layer=%d out_blk=%d in_blk=%d kt=%d\n", layer, out_block, in_block, kt);
                    }
                }

                kittens::wait(pipeline::a_ready(s), parity);
                kittens::wait(pipeline::b_ready(s), parity);
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
            const int layer = inst[1];
            const int out_block = inst[2];
            const int in_block = inst[3];

            const int batch_tiles = div_up(g.batch_size, K_TILE);

            output_frag_t grad_tile;
            kittens::warp::zero(grad_tile);

            for (int kt = 0; kt < batch_tiles; ++kt) {
                const int parity = kt & 1;
                kittens::wait(pipeline::a_ready(s), parity);
                kittens::wait(pipeline::b_ready(s), parity);
                __syncwarp();

                weight_frag_col_t grad_out_frag;
                weight_frag_col_t activ_frag;
                kittens::warp::load(grad_out_frag, reinterpret_cast<weight_tile_bwd_t &>(pipeline::a_tile(s)));
                kittens::warp::load(activ_frag, reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s)));

                kittens::warp::mma_AtB(grad_tile, grad_out_frag, activ_frag, grad_tile);
                __syncwarp();
            }

            // store full tile into padded grad_w
            const float scale = 1.f / float(g.batch_size);
            kittens::warp::mul(grad_tile, grad_tile, scale);
            kittens::warp::store(g.grad_w, grad_tile, kittens::coord<>{layer, 0, out_block * N_TILE, in_block * N_TILE});

            pipeline::release_pages(s);
        }
    };
};

template <typename C = config>
struct LinearBwdInput {
    static constexpr int opcode = OpCode::LINEAR_BWD_INPUT;
    using pipeline = two_page_pipeline<C, linear_training_globals>;

    struct controller {
        static __device__ int init_semaphores(const linear_training_globals &, state_t &s) {
            return pipeline::init_semaphores(s);
        }
        static __device__ int release_lid(const linear_training_globals &, typename C::instruction_t &, int &query) {
            if (query == 0) return pipeline::A_LID;
            if (query == 1) return pipeline::B_LID;
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_training_globals &g, state_t &s) {
            const int lane = kittens::laneid();
            const auto &inst = s.instruction();
            const int layer = inst[1];
            const int m_block = inst[2]; // batch tile
            const int n_block = inst[3]; // input tile

            pipeline::wait_pages(s);

            const int out_dim = get_out_dim(g, layer);
            const int k_tiles = div_up(out_dim, K_TILE);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int parity = kt & 1;
                const int k_start = kt * K_TILE;

                if (lane == 0) {
                    kittens::tma::expect_bytes(pipeline::a_ready(s), sizeof(float) * M_TILE * K_TILE);
                    kittens::tma::expect_bytes(pipeline::b_ready(s), sizeof(float) * K_TILE * N_TILE);

                    // A tile: grad_out (batch, out_dim)
                    kittens::tma::load_async(
                        pipeline::a_tile(s),
                        g.grad_activations,
                        kittens::coord<>{layer + 1, 0, m_block * M_TILE, k_start},
                        pipeline::a_ready(s));

                    // B tile: weights (out_dim, in_dim)
                    kittens::tma::load_async(
                        reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s)),
                        g.weights,
                        kittens::coord<>{layer, 0, k_start, n_block * N_TILE},
                        pipeline::b_ready(s));

                    if (g.debug_vis) {
                        printf("[bwd-in loader] layer=%d m_blk=%d n_blk=%d kt=%d\n", layer, m_block, n_block, kt);
                    }
                }

                kittens::wait(pipeline::a_ready(s), parity);
                kittens::wait(pipeline::b_ready(s), parity);
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
            const int layer = inst[1];
            const int m_block = inst[2];
            const int n_block = inst[3];

            const int out_dim = get_out_dim(g, layer);
            const int k_tiles = div_up(out_dim, K_TILE);

            output_frag_t grad_frag;
            kittens::warp::zero(grad_frag);

            for (int kt = 0; kt < k_tiles; ++kt) {
                const int parity = kt & 1;
                kittens::wait(pipeline::a_ready(s), parity);
                kittens::wait(pipeline::b_ready(s), parity);
                __syncwarp();

                activ_frag_t grad_out_frag;
                weight_frag_col_t weight_frag;
                kittens::warp::load(grad_out_frag, pipeline::a_tile(s));
                kittens::warp::load(weight_frag, reinterpret_cast<weight_tile_bwd_t &>(pipeline::b_tile(s)));

                kittens::warp::mma_AB(grad_frag, grad_out_frag, weight_frag, grad_frag);
                __syncwarp();
            }

            kittens::warp::store(
                g.grad_activations,
                grad_frag,
                kittens::coord<>{layer, 0, m_block * M_TILE, n_block * N_TILE});

            pipeline::release_pages(s);
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
                    g.weights[{layer, 0, row, col}] = w - g.lr * grad;
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
