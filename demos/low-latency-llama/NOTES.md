```python
# megakernels/scripts/llama_repl.py
@torch.inference_mode()
def main(config: ScriptConfig):
    # set up a model, vanilla setup stuff...
    extra_config = ExtraModelConfig(
        interleave_rope=True,
        max_batch_size=1,
    )
    model = LlamaForCausalLM.from_pretrained( # <-- custom torch model
        config.model, device=config.device, extra_config=extra_config
    )
    # now they start with the custom stuff
    schedule_builder = make_schedule_builder(config.setting) # <-- dispatches to a schedule builder class, with the defaults this is `LatencyScheduleBuilder`
    schedule = schedule_builder.build(model) # <-- makes a Schedule, by layer
    assigned_to_sms = assign_to_sms( # <-- uses a strategy (rr, zz, wave, dag, or pool) to put instructions into sm queues
        config.sched, schedule=schedule, memory_fraction=config.memory_fraction
    )
    tensorize_instructions(schedule.globs, assigned_to_sms) # <-- pad queues with NoOp's

    # they branch on mode, we are interested in the "mk" branch so we will follow that here
    interpreter = make_mk_interpreter(config.setting, config.mk_dir)
    gen = MK_Generator( # <--
        model,
        interpreter,
        schedule,
        barrier_fill_val=0,
        skip_mk=False,
        skip_rest=False,
    )
    # some more boilerplate...
    # the interesting part of the generate function:
    position_ids = torch.arange(prompt_len).to(model.device)

    prefill_inp = BatchState( # <--
        input_ids=input_ids,
        position_ids=position_ids,
    )

    prefill_output: BatchState = model(prefill_inp)
    assert prefill_output.output_ids is not None
    new_input_token = prefill_output.output_ids[:, -1:]

    output_tokens[:, 0] = new_input_token

    start_event.record()
    until_eos, num_generated = gen.generate_with_eos( # <-- calls MK_Generator.generate -> MK_Generator.run -> MK_Generator.intepreter.interpret(self.schedule.globs)
        output_tokens=output_tokens,
        prompt_len=prompt_len,
        ntok=config.max_tokens_per_turn,
        eos_token_ids=eos_token_ids,
        eos_token_check_interval=16,
    )
    end_event.record()
```

Mostly just defines a pure torch LLaMA model with some custom helpers for loading weights. As far as I can tell, its pure torch.

```python
# megakernels/llama.py

# ...
class LlamaForCausalLM(nn.Module): # line 501
    def __init__(
        self,
        config: LlamaConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self.model = LlamaModel(config, extra_config)
        self.lm_head = LlamaLMHead(config, extra_config)
# ...
```


```python
# megakernels/dispatch.py
from megakernels.demos.latency.mk import LatencyMK_Interpreter
from megakernels.demos.latency.python_vm import (
    INSTRUCTION_TO_SOLVER as LATENCY_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.latency.scheduler import LatencyScheduleBuilder
from megakernels.demos.throughput.mk import ThroughputMK_Interpreter
from megakernels.demos.throughput.python_vm import (
    INSTRUCTION_TO_SOLVER as THROUGHPUT_INSTRUCTION_TO_SOLVER,
)
from megakernels.demos.throughput.scheduler import ThroughputScheduleBuilder
from megakernels.mk import MK_Interpreter
from megakernels.python_vm import PyVM_Interpreter
from megakernels.scheduler import ScheduleBuilder

BUILDER_MAP = {
    "latency": LatencyScheduleBuilder,
    "throughput": ThroughputScheduleBuilder,
}

MK_INTERPRETER_MAP = {
    "latency": LatencyMK_Interpreter,
    "throughput": ThroughputMK_Interpreter,
}

INSTRUCTION_TO_SOLVER_MAP = {
    "latency": LATENCY_INSTRUCTION_TO_SOLVER,
    "throughput": THROUGHPUT_INSTRUCTION_TO_SOLVER,
}


def make_schedule_builder(mode: str) -> ScheduleBuilder:
    return BUILDER_MAP[mode]()


def make_mk_interpreter(mode: str, mk_dir: Path) -> MK_Interpreter:
    return MK_INTERPRETER_MAP[mode](mk_dir)


def make_pyvm_interpreter(mode: str) -> PyVM_Interpreter:
    return PyVM_Interpreter(INSTRUCTION_TO_SOLVER_MAP[mode])

```

## Tracing interpreter

```python
# megakernels/demos/latency/mk.py
import torch
from einops import rearrange

from megakernels.demos.latency.instructions import Globals
from megakernels.mk import MK_Interpreter


def interpret_with_mk(
    globs: Globals,
    mk_func,
):
    fourD_k_cache = rearrange(globs.k_cache, "l b t h d -> (l b) t h d")
    fourD_v_cache = rearrange(globs.v_cache, "l b t h d -> (l b) t h d")

    mk_func(
        # vm stuff
        globs.barriers,
        globs.instructions,
        globs.timings,
        # weights
        globs.qkv_proj_weights,
        globs.attn_ln_weights,
        globs.o_proj_weights,
        globs.mlp_ln_weights,
        globs.up_proj_weights,
        globs.gate_proj_weights,
        globs.down_proj_weights,
        globs.lm_head_norm_weights.data,
        globs.lm_head_weights.data,
        fourD_k_cache,
        fourD_v_cache,
        # rope
        globs.rope_cos,
        globs.rope_sin,
        # activations
        globs.hidden_states,
        globs.post_ln_rope_q,
        globs.attn_out,
        globs.attn_lse_intermediates,
        globs.attn_out_intermediates,
        globs.silu_out,
        globs.logits,
        # scalars
        globs.pos_id,
        globs.attn_scale,
        globs.rms_norm_eps,
        globs.skip_attn_reduction,
        stream=torch.cuda.current_stream(),
    )


class LatencyMK_Interpreter(MK_Interpreter): # <- Tracing this to see where self.mk_func comes from
    def interpret(self, globs: Globals):
        interpret_with_mk(globs, self.mk_func)
```


```python
# megakernels/mk.py
def get_mk_func(mk_dir: Path):
    sys.path.append(str(mk_dir.expanduser().absolute()))
    from mk_llama import mk_llama  # compiled extension # <-- tracing back to cuda...

    return mk_llama


class MK_Interpreter:
    def __init__(self, mk_dir: Path):
        self.mk_func = get_mk_func(mk_dir)

    def interpret(self, globs):
        raise NotImplementedError
```

This is where mk_llama is created

```c
// demos/low-latency-llama/llama.cu
#include "llama.cuh"

#include "rms_matvec_rope_append.cu"
#include "attention_partial.cu"
#include "attention_reduction.cu"
#include "matvec_adds.cu"
#include "upgate.cu"
#include "rms_lm_head.cu"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace megakernel;

using rms_qkv_rope_append_op =
    rms_qkv_rope_append<default_config, llama_1b_globals>;
using attention_partial_op =
    attention_partial<default_config, llama_1b_globals>;
using attention_reduction_op =
    attention_reduction<default_config, llama_1b_globals>;
using o_proj_op = o_proj<default_config, llama_1b_globals>;
using rms_upgate_silu_op = rms_upgate_silu<default_config, llama_1b_globals>;
using downproj_op = downproj<default_config, llama_1b_globals>;
using rms_lm_head_op = rms_lm_head<default_config, llama_1b_globals>;

PYBIND11_MODULE(mk_llama, m) {
    m.doc() = "";
    kittens::py::bind_kernel<
        mk<default_config, llama_1b_globals, attention_partial_op,
            attention_reduction_op, rms_qkv_rope_append_op, downproj_op,
            o_proj_op, rms_upgate_silu_op, rms_lm_head_op>>(
        m, "mk_llama", &llama_1b_globals::Bar, &llama_1b_globals::instructions,
        &llama_1b_globals::timings,

        &llama_1b_globals::qkv_weights, &llama_1b_globals::attn_norm_weights,
        &llama_1b_globals::o_weights, &llama_1b_globals::mlp_norm_weights,
        &llama_1b_globals::up_weights, &llama_1b_globals::gate_weights,
        &llama_1b_globals::down_weights,
        &llama_1b_globals::lm_head_norm_weights,
        &llama_1b_globals::lm_head_weights, &llama_1b_globals::k_cache,
        &llama_1b_globals::v_cache,

        &llama_1b_globals::rope_cos, &llama_1b_globals::rope_sin,

        &llama_1b_globals::hidden_states, &llama_1b_globals::q_post_rope,
        &llama_1b_globals::attn_out, &llama_1b_globals::attn_lse_intermediates,
        &llama_1b_globals::attn_out_intermediates, &llama_1b_globals::silu_out,
        &llama_1b_globals::logits,

        &llama_1b_globals::pos_id, &llama_1b_globals::attn_scale,
        &llama_1b_globals::rms_norm_eps,
        &llama_1b_globals::skip_attn_reduction);
}
```


```h
// ThunderKittens/include/pyutils/pyutils.cuh
template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args, pybind11::kwargs kwargs) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        cudaStream_t raw_stream = nullptr;
        if (kwargs.contains("stream")) {
            // Extract stream pointer
            uintptr_t stream_ptr = kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
            raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        }
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, raw_stream>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block(), 0, raw_stream>>>(__g__);
        }
    });
}
```

## Tracing schedule builder

```python
# megakernels/scheduler.py
INTS_PER_INSTRUCTION = 32
TIMING_SLOTS = 128


@dataclass
class DAG_Node:
    def __hash__(self):
        return hash(tuple(self.instruction.serialize()))

    instruction: Instruction
    dependencies: list["DAG_Node"]

    children: set["DAG_Node"] = field(default_factory=set)
    start_time: float = float("inf")
    end_time: float = float("inf")
    remaining_dependencies: set["DAG_Node"] = field(default_factory=set)
    priority: float = 0

    def earliest_ready_time(self, globs: BaseGlobals):
        if len(self.dependencies) == 0:
            return 0

        return max(dep.end_time for dep in self.dependencies)

    def register_with_parents(self):
        for dep in self.dependencies:
            dep.children.add(self)

    def calc_priority(self, globs: BaseGlobals):
        cur_cost = self.priority
        for dep in self.dependencies:
            pri = cur_cost + dep.instruction.cost(globs)
            dep.priority = max(pri, dep.priority)
            dep.calc_priority(globs)


@dataclass
class Schedule:
    globs: BaseGlobals
    dag_nodes: list[DAG_Node]
    end_node: DAG_Node

    def get_linear_instructions(self):
        # NOTE: assumes this is in topological order
        return [node.instruction for node in self.dag_nodes]

    def smart_assign_to_sms(self):
        return assign_dag_to_sms(self)

    def round_robin_assign_to_sms(self):
        instructions = self.get_linear_instructions()
        return round_robin_assign_to_sms(instructions, self.globs.sm_count())


class ScheduleBuilder:
    @classmethod
    def make_globals(cls, model):
        raise NotImplementedError

    @classmethod
    def make_dag(
        cls, globs, stop_after_op: str | None = None, layer_limit: int | None = None
    ):
        raise NotImplementedError

    @classmethod
    def build(
        cls,
        model: LlamaForCausalLM,
        stop_after_op: str | None = None,
        layer_limit: int | None = None,
    ):
        globs = cls.make_globals(model)
        dag_nodes, end_node = cls.make_dag(globs, stop_after_op, layer_limit)
        return Schedule(globs, dag_nodes, end_node)

    @classmethod
    def with_new_globals(cls, schedule: Schedule, model: LlamaForCausalLM):
        return replace(schedule, globs=cls.make_globals(model))
```

```python
# megakernels/demos/latency/scheduler.py
# NOTE: the interface is identical to the ThroughputScheduleBuilder
class LatencyScheduleBuilder(ScheduleBuilder):
    @classmethod
    def make_globals(cls, model):
        return make_globals(model)

    @classmethod
    def make_dag(
        cls, globs, stop_after_op: str | None = None, layer_limit: int | None = None # globs means globals
    ):
        # returns tuple[list[DAG_Node], DAG_Node] where the last node is a NoOp. 
        # From make_dag:
        #   end_node = DAG_Node(NoOp(), last_outputs)
        #   return nodes, end_node
        return make_dag(globs, stop_after_op, layer_limit) 
```

```python
# megakernels/demos/latency/scheduler.py
def assign_to_sms(
    mode: str,
    schedule: Schedule | None = None,
    instructions: list[Instruction] | None = None,
    sm_count: int | None = None,
    memory_fraction: float | None = None,
):
    if schedule is not None:
        instructions = schedule.get_linear_instructions()
        sm_count = schedule.globs.sm_count()

    match mode:
        case "rr":
            return round_robin_assign_to_sms(instructions, sm_count)
        case "zz":
            return zig_zag_assign_to_sms(instructions, sm_count)
        case "wave":
            return wave_assign_to_sms(schedule)
        case "dag":
            return assign_dag_to_sms(schedule)
        case "pool":
            assert memory_fraction is not None
            return pool_assign_to_sms(
                instructions, sm_count, memory_fraction=memory_fraction
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")
```

```python
# megakernels/instructions.py
def serialize_and_pad(instruction: Instruction):
    serialized = instruction.serialize()
    num_padding = INTS_PER_INSTRUCTION - len(serialized)
    assert num_padding >= 0
    return serialized + [0] * num_padding


def tensorize_instructions(
    globs: BaseGlobals,
    instruction_queues: list[list[Instruction]],
):
    num_sms = globs.sm_count()

    max_queue_len = max(len(queue) for queue in instruction_queues)
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))

    flattened = []
    for queue in instruction_queues:
        flattened.extend(serialize_and_pad(instruction) for instruction in queue)

    device = globs.device

    serialized = torch.tensor(flattened, dtype=torch.int32, device=device).view(
        num_sms, -1, INTS_PER_INSTRUCTION
    )

    timings = torch.zeros(
        [num_sms, max_queue_len, TIMING_SLOTS],
        dtype=torch.int32,
        device=device,
    )

    globs.instructions = serialized
    globs.timings = timings
```

Example instructions for the latency demo

```python
# megakernels/demos/latency/instructions.py
@dataclass
class PartialAttention(Instruction):
    layer_idx: int
    kv_head_idx: int
    num_partials: int
    partial_idx: int

    @classmethod
    def opcode(cls) -> int:
        return 2

    @classmethod
    def prev_opcode(cls) -> int:
        return LayerNorm_QKV_MatVecRopeAppend.opcode()

    def cost(self, globs: Globals):
        seq_len = globs.pos_id + 1
        loaded_seq_len = seq_len / self.num_partials

        # num loaded elements from kv cache
        return loaded_seq_len * globs.head_dim * 2


@dataclass
class AttentionReduction(Instruction):
    layer_idx: int
    head_start_idx: int
    # the original number of attention partitions
    num_partials: int
    is_terminal: bool
    # TODO: make sure reduction_list can't go beyond instruction
    reduction_list: list[int]
    # Not required for the last reduction
    output_partial_idx: Optional[int] = None

    @classmethod
    def opcode(cls) -> int:
        return 3

    @classmethod
    def prev_opcode(cls) -> int:
        return PartialAttention.opcode()


@dataclass
class MatVecAdd(Instruction):
    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int
```

## Generator
```python
class Generator:
    def generate(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        ntok_already_generated: int = 1,
    ):
        raise NotImplementedError

    def generate_with_eos(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        eos_token_check_interval: int,
        eos_token_ids: list[int],
    ):
        """
        Return pos id with first eos token, and total num tokens generated
        """
        assert output_tokens.shape[0] == 1, "batch size must be 1"

        for ntok_already_generated in range(
            1,
            ntok,
            eos_token_check_interval,
        ):
            ntok_for_chunk = min(
                eos_token_check_interval, ntok - ntok_already_generated
            )
            self.generate(
                output_tokens,
                prompt_len=prompt_len,
                ntok=ntok_for_chunk,
                ntok_already_generated=ntok_already_generated,
            )

            start_out_idx = ntok_already_generated
            end_out_idx = ntok_already_generated + ntok_for_chunk

            to_cpu = output_tokens[0, start_out_idx:end_out_idx].cpu()
            for j, token in enumerate(to_cpu):
                if token in eos_token_ids:
                    # -1 because we didn't generate the first token
                    return start_out_idx + j, end_out_idx - 1

        return ntok, ntok - 1

class MK_Generator(Generator):
    def __init__(
        self,
        model: LlamaForCausalLM,
        interpreter: MK_Interpreter,
        schedule: Schedule,
        barrier_fill_val: int = 0,
        skip_mk: bool = False,
        skip_rest: bool = False,
    ):
        self.model = model
        self.interpreter = interpreter
        self.schedule = schedule

        self.barrier_fill_val = barrier_fill_val
        self.skip_mk = skip_mk
        self.skip_rest = skip_rest

        self.fill()

    def fill(self):
        self.schedule.globs.barriers.fill_(self.barrier_fill_val)

    def replace_with_noops(self):
        self.schedule.globs.instructions.zero_()

    def run(self, input_ids: Tensor, pos_id: int):
        if not self.skip_rest:
            batch_state = BatchState(
                input_ids=input_ids,
            )

            post_embedding: BatchState = self.model.model.embed_tokens(batch_state)
            hiddens = post_embedding.hidden_states
            assert hiddens is not None
            self.schedule.globs.hidden_states[:] = hiddens.squeeze(1)

        self.fill()
        self.schedule.globs.pos_id = pos_id
        if not self.skip_mk:
            self.interpreter.interpret(self.schedule.globs)

        if self.skip_rest:
            return input_ids

        logits = self.schedule.globs.logits
        output_ids = torch.argmax(logits, dim=-1)

        return output_ids

    def generate(
        self,
        output_tokens: Tensor,
        prompt_len: int,
        ntok: int,
        ntok_already_generated: int = 1,
    ):
        """
        Return num tokens until stop seq, and total num tokens generated
        """
        for i in range(ntok):
            input_token_pos = ntok_already_generated + i - 1
            output_token_pos = input_token_pos + 1

            input_ids = output_tokens[:, input_token_pos : input_token_pos + 1]

            pos_id = prompt_len + ntok_already_generated + i - 1
            output_ids = self.run(input_ids, pos_id=pos_id)
            output_tokens[:, output_token_pos] = output_ids.squeeze(-1)
```

## Batch state

```python
# megakernels/model_types.py
@dataclass
class BatchState:
    input_ids: Tensor
    position_ids: Tensor | None = None
    seq_len: int | None = None
    output_ids: Tensor | None = None
    hidden_states: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None

    kv_indices: Tensor | None = None
    kv_indptr: Tensor | None = None
    kv_last_page_lens: Tensor | None = None
    kv_seqlens: Tensor | None = None
    qo_indptr: Tensor | None = None
    prefill_wrapper: Any | None = None
    decode_wrapper: Any | None = None

    def __post_init__(self):
        if self.seq_len is None:
            self.seq_len = self.input_ids.shape[1]
```