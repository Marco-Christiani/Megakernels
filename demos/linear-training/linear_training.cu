#include <pybind11/pybind11.h>

#include "kittens.cuh"
#include "megakernel.cuh"
#include "conf.cuh"
#include "vm/vm.cuh"
#include "pyutils/torch_helpers.cuh"
#include <cstdio>
#include <iostream>

namespace py = pybind11;

// using namespace megakernel;
using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int IN_DIM = 128;
constexpr int OUT_DIM = 64;
constexpr int BLOCK = 16;

using config = megakernel::linear_config;

struct linear_globals {
    using instruction_layout = vm::instruction_layout<config>;
    using timing_layout = vm::timing_layout<config>;
    // using instruction_layout = megakernel::instruction_layout<config>;
    // using timing_layout = megakernel::timing_layout<config>;

    instruction_layout instructions;
    timing_layout timings;

    // gl<uint, 1, 1, 1, 32> barriers;  // Dummy barriers for now

    float *input;
    float *target;
    float *output;
    float *grad_out;
    float *weights;
    float *grad_w;

    float lr;
    int batch_size;
#if LINEAR_DEMO_DEBUG_ENABLED
    int debug_vis;
#endif

    dim3 grid() const { return dim3(1); }
    dim3 block() const { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() const { return config::DYNAMIC_SHARED_MEMORY; }
};

template <typename C = config> struct LinearFwd {
    static constexpr int opcode = 1;

    struct controller {
        static __device__ int init_semaphores(const linear_globals &, state<C> &) { return 0; }
        static __device__ int release_lid(const linear_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_globals &, state<C> &s) {
            if (laneid() >= 1 && laneid() < config::NUM_PAGES) {
                int pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_globals &, state<C> &s) {
#ifdef KITTENS_BLACKWELL
            if (laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_globals &, state<C> &) {} };

    struct consumer {
        static __device__ void run(const linear_globals &g, state<C> &s) {
            auto &inst = s.instruction();
            int out_block = inst[2];
            int in_block = inst[3];

#if LINEAR_DEMO_DEBUG_ENABLED
            if (g.debug_vis && laneid() == 0 && s.instruction_index == 0 && kittens::warpid() == 0) {
                printf(
                    "[linear-demo] worker_id=%u opcode=%d instruction_depth=%d instruction_rows=%d timings_depth=%d\n",
                    ::kittens::prototype::vm::get_worker_id(),
                    inst[0],
                    (int)g.instructions.depth(),
                    (int)g.instructions.rows(),
                    (int)g.timings.depth());
            }
#endif

            if (kittens::warpid() == 0 && laneid() == 0) {
                for (int b = 0; b < g.batch_size; b++) {
                    float *out_tile = g.output + b * OUT_DIM + out_block * BLOCK;
                    float *in_tile = g.input + b * IN_DIM + in_block * BLOCK;
                    for (int o = 0; o < BLOCK; o++) {
                        float acc = (in_block == 0) ? 0.f : out_tile[o];
                        int out_row = out_block * BLOCK + o;
                        float *w_row = g.weights + out_row * IN_DIM + in_block * BLOCK;
                        for (int k = 0; k < BLOCK; k++) {
                            acc += in_tile[k] * w_row[k];
                        }
                        out_tile[o] = acc;
                    }
                }
            }
        }
    };
};

template <typename C = config> struct LossGrad {
    static constexpr int opcode = 2;

    struct controller {
        static __device__ int init_semaphores(const linear_globals &, state<C> &) { return 0; }
        static __device__ int release_lid(const linear_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_globals &, state<C> &s) {
            if (laneid() >= 1 && laneid() < config::NUM_PAGES) {
                int pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_globals &, state<C> &s) {
#ifdef KITTENS_BLACKWELL
            if (laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_globals &, state<C> &) {} };

    struct consumer {
        static __device__ void run(const linear_globals &g, state<C> &s) {
            auto &inst = s.instruction();
            int out_block = inst[2];
            float scale = 2.0f / static_cast<float>(g.batch_size);

            if (kittens::warpid() == 0 && laneid() == 0) {
                for (int b = 0; b < g.batch_size; b++) {
                    float *out_tile = g.output + b * OUT_DIM + out_block * BLOCK;
                    float *tgt_tile = g.target + b * OUT_DIM + out_block * BLOCK;
                    float *grad_tile = g.grad_out + b * OUT_DIM + out_block * BLOCK;
                    for (int o = 0; o < BLOCK; o++) {
                        grad_tile[o] = (out_tile[o] - tgt_tile[o]) * scale;
                    }
                }
            }
        }
    };
};

template <typename C = config> struct LinearBwdWeight {
    static constexpr int opcode = 3;

    struct controller {
        static __device__ int init_semaphores(const linear_globals &, state<C> &) { return 0; }
        static __device__ int release_lid(const linear_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_globals &, state<C> &s) {
            if (laneid() >= 1 && laneid() < config::NUM_PAGES) {
                int pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_globals &, state<C> &s) {
#ifdef KITTENS_BLACKWELL
            if (laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_globals &, state<C> &) {} };

    struct consumer {
        static __device__ void run(const linear_globals &g, state<C> &s) {
            auto &inst = s.instruction();
            int out_block = inst[2];
            int in_block = inst[3];
            float inv_batch = 1.0f / static_cast<float>(g.batch_size);

            if (kittens::warpid() == 0 && laneid() == 0) {
                for (int o = 0; o < BLOCK; o++) {
                    int out_row = out_block * BLOCK + o;
                    float *grad_w_row = g.grad_w + out_row * IN_DIM + in_block * BLOCK;
                    for (int k = 0; k < BLOCK; k++) {
                        float acc = 0.f;
                        for (int b = 0; b < g.batch_size; b++) {
                            float grad = g.grad_out[b * OUT_DIM + out_row];
                            float in_val = g.input[b * IN_DIM + in_block * BLOCK + k];
                            acc += grad * in_val;
                        }
                        grad_w_row[k] = acc * inv_batch;
                    }
                }
            }
        }
    };
};

template <typename C = config> struct SgdUpdate {
    static constexpr int opcode = 4;

    struct controller {
        static __device__ int init_semaphores(const linear_globals &, state<C> &) { return 0; }
        static __device__ int release_lid(const linear_globals &, typename C::instruction_t &, int &query) {
            return query % C::NUM_PAGES;
        }
    };

    struct loader {
        static __device__ void run(const linear_globals &, state<C> &s) {
            if (laneid() >= 1 && laneid() < config::NUM_PAGES) {
                int pid = s.pid(laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const linear_globals &, state<C> &s) {
#ifdef KITTENS_BLACKWELL
            if (laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };

    struct storer { static __device__ void run(const linear_globals &, state<C> &) {} };

    struct consumer {
        static __device__ void run(const linear_globals &g, state<C> &s) {
            auto &inst = s.instruction();
            int out_block = inst[2];
            int in_block = inst[3];

            if (kittens::warpid() == 0 && laneid() == 0) {
                for (int o = 0; o < BLOCK; o++) {
                    int out_row = out_block * BLOCK + o;
                    float *grad_w_row = g.grad_w + out_row * IN_DIM + in_block * BLOCK;
                    float *w_row = g.weights + out_row * IN_DIM + in_block * BLOCK;
                    for (int k = 0; k < BLOCK; k++) {
                        w_row[k] -= g.lr * grad_w_row[k];
                    }
                }
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
#if !LINEAR_DEMO_DEBUG_ENABLED
    (void)debug_vis;
#endif
    TORCH_CHECK(instructions_tensor.is_cuda(), "instructions must be on CUDA");
    TORCH_CHECK(input.is_cuda() && target.is_cuda() && weights.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(instructions_tensor.scalar_type() == torch::kInt, "instructions must be int32");
    TORCH_CHECK(
        instructions_tensor.dim() == 3,
        "instructions must be [sm_count, num_instructions, 32]");
    TORCH_CHECK(
        instructions_tensor.size(2) == config::INSTRUCTION_WIDTH,
        "expected instruction width ",
        config::INSTRUCTION_WIDTH,
        " got ",
        instructions_tensor.size(2));
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "input must be float32");
    TORCH_CHECK(target.scalar_type() == torch::kFloat, "target must be float32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be float32");
    TORCH_CHECK(input.dim() == 2 && target.dim() == 2, "input/target must be [batch, dim]");
    TORCH_CHECK(input.size(1) == IN_DIM && target.size(1) == OUT_DIM, "unexpected input/target dims");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == OUT_DIM && weights.size(1) == IN_DIM,
                "weights must be [out_dim, in_dim]");

    auto batch_size = static_cast<int>(input.size(0));

    // allocate buffers
    auto output = torch::zeros_like(target);
    auto grad_out = torch::zeros_like(target);
    auto grad_w = torch::zeros_like(weights);
    // auto timings = torch::zeros({instructions_tensor.size(0), config::TIMING_WIDTH},
    //                             torch::TensorOptions().dtype(torch::kInt).device(instructions_tensor.device()));

    // setup globals

    // instructions_tensor [num_instr, 32]
    auto timings = torch::zeros(
        {instructions_tensor.size(0), instructions_tensor.size(1), config::TIMING_WIDTH},
        torch::TensorOptions().dtype(torch::kInt).device(instructions_tensor.device()));

    const size_t inst_depth = static_cast<size_t>(instructions_tensor.size(0));
    const size_t inst_rows = static_cast<size_t>(instructions_tensor.size(1));
    // Layout args: data, batch (compile-time 1 -> nullptr), depth, rows, cols (compile-time -> nullptr)
    typename linear_globals::instruction_layout inst_layout{
        instructions_tensor.data_ptr<int>(), nullptr, inst_depth, inst_rows, nullptr};

    typename linear_globals::timing_layout timing_layout{
        timings.data_ptr<int>(), nullptr, inst_depth, inst_rows, nullptr};

    int device_id = -1;
    auto device_status = cudaGetDevice(&device_id);
    TORCH_CHECK(
        device_status == cudaSuccess,
        "cudaGetDevice failed: ",
        cudaGetErrorString(device_status));
    cudaDeviceProp props{};
    auto props_status = cudaGetDeviceProperties(&props, device_id);
    TORCH_CHECK(
        props_status == cudaSuccess,
        "cudaGetDeviceProperties failed: ",
        cudaGetErrorString(props_status));

#if LINEAR_DEMO_DEBUG_ENABLED
    if (debug_vis) {
        std::cout << "[linear-demo] instruction layout rows=" << inst_layout.rows()
                  << ", depth=" << inst_layout.depth() << std::endl;
        std::cout << "[linear-demo] timing layout rows=" << timing_layout.rows()
                  << ", depth=" << timing_layout.depth() << std::endl;
        std::cout << "[linear-demo] cudaGetDeviceProperties reports " << props.multiProcessorCount
                  << " SMs on device " << device_id << " (" << props.name << ")" << std::endl;
        std::cout << "[linear-demo] NOTE: controller indexes layouts by worker_id (SM id)" << std::endl;
    }
#endif


    // sigh
    // typename linear_globals::instruction_layout inst_layout{
    //     instructions_tensor.data_ptr<int>(),
    //     1,                                              // b 
    //     static_cast<int>(instructions_tensor.size(1)),  // d num instructions
    //     1,                                              // r
    //     config::INSTRUCTION_WIDTH                       // c (32)
    // };

    // typename linear_globals::timing_layout timing_layout{
    //     timings.data_ptr<int>(),
    //     1,                                              // b
    //     static_cast<int>(timings.size(1)),              // d
    //     1,                                              // r
    //     config::TIMING_WIDTH                            // c
    // };


    // auto barriers = torch::zeros({1, 1, 32}, 
    //     torch::TensorOptions().dtype(torch::kInt32).device(instructions_tensor.device()));

    linear_globals g{
        inst_layout,
        timing_layout,
        // {barriers.data_ptr<uint32_t>(), {1, 1, 32}},
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        weights.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        static_cast<float>(lr),
        batch_size
#if LINEAR_DEMO_DEBUG_ENABLED
        , debug_vis ? 1 : 0
#endif
    };

    vm::kvm<config, linear_globals, LinearFwd<config>, LossGrad<config>, LinearBwdWeight<config>, SgdUpdate<config>>
        <<<1, config::NUM_THREADS, g.dynamic_shared_memory()>>>(g);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "linear training kernel launch failed: ", cudaGetErrorString(e));

    return {output, grad_out, grad_w, timings};
}

PYBIND11_MODULE(linear_training, m) {
    m.def(
        "run_linear_training",
        &run_linear_training,
        "Linear training demo",
        py::arg("instructions"),
        py::arg("input"),
        py::arg("target"),
        py::arg("weights"),
        py::arg("lr"),
        py::arg("debug_vis") = false);
}
