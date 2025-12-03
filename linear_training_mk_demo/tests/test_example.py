import math
import os
from enum import IntEnum

import torch

import linear_training_mk_demo

IN_DIM = 64
OUT_DIM = 64
NUM_LAYERS = 2

M_TILE = 16
N_TILE = 16


class OpCodes(IntEnum):
    LINEAR_FWD = 1
    LOSS_GRAD = 2
    LINEAR_BWD_WEIGHT = 3
    LINEAR_BWD_INPUT = 4
    SGD_UPDATE = 5


def env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.lower() not in ("0", "false", "")


def tile_range(length: int, tile: int) -> range:
    return range(math.ceil(length / tile))


def build_single_queue(batch: int, in_dim: int, out_dim: int, num_layers: int, device: torch.device) -> torch.Tensor:
    insts: list[list[int]] = []

    for layer in range(num_layers):
        for m_block in tile_range(batch, M_TILE):
            for n_block in tile_range(out_dim, N_TILE):
                row = [0] * 32
                row[0] = OpCodes.LINEAR_FWD.value
                row[1] = layer
                row[2] = m_block
                row[3] = n_block
                insts.append(row)

    for m_block in tile_range(batch, M_TILE):
        for n_block in tile_range(out_dim, N_TILE):
            row = [0] * 32
            row[0] = OpCodes.LOSS_GRAD.value
            row[1] = num_layers - 1
            row[2] = m_block
            row[3] = n_block
            insts.append(row)

    for layer in reversed(range(num_layers)):
        for out_block in tile_range(out_dim, N_TILE):
            for in_block in tile_range(in_dim, N_TILE):
                row = [0] * 32
                row[0] = OpCodes.LINEAR_BWD_WEIGHT.value
                row[1] = layer
                row[2] = out_block
                row[3] = in_block
                insts.append(row)

        for m_block in tile_range(batch, M_TILE):
            for n_block in tile_range(in_dim, N_TILE):
                row = [0] * 32
                row[0] = OpCodes.LINEAR_BWD_INPUT.value
                row[1] = layer
                row[2] = m_block
                row[3] = n_block
                insts.append(row)

        for out_block in tile_range(out_dim, N_TILE):
            for in_block in tile_range(in_dim, N_TILE):
                row = [0] * 32
                row[0] = OpCodes.SGD_UPDATE.value
                row[1] = layer
                row[2] = out_block
                row[3] = in_block
                insts.append(row)

    return torch.tensor(insts, device=device, dtype=torch.int32)


def build_instructions(device: torch.device, sm_count: int, batch: int) -> torch.Tensor:
    per_sm_insts = build_single_queue(batch, IN_DIM, OUT_DIM, NUM_LAYERS, device)
    expanded = (
        per_sm_insts.unsqueeze(0)
        .expand(sm_count, per_sm_insts.size(0), per_sm_insts.size(1))
        .contiguous()
    )
    return expanded


def run_demo_step(batch: int = 4, lr: float = 0.05, debug: bool = False):
    torch.manual_seed(0)
    device = torch.device("cuda")

    x = torch.randn(batch, IN_DIM, device=device, dtype=torch.float32)
    target = torch.randn(batch, OUT_DIM, device=device, dtype=torch.float32)
    weights = torch.randn(NUM_LAYERS, OUT_DIM, IN_DIM, device=device, dtype=torch.float32)
    weights_ref = weights.clone()

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    instructions = build_instructions(device, props.multi_processor_count, batch)

    out, grad_out, grad_input, grad_w, weights_out, timings = linear_training_mk_demo.run(
        instructions, x, target, weights, lr, debug
    )
    torch.cuda.synchronize()

    activations: list[torch.Tensor] = [x]
    for l in range(NUM_LAYERS):
        activations.append(activations[-1] @ weights_ref[l].t())

    out_ref = activations[-1]
    grad_acts: list[torch.Tensor | None] = [None] * (NUM_LAYERS + 1)
    grad_acts[-1] = 2.0 * (out_ref - target) / batch

    grad_w_ref = torch.zeros_like(weights_ref)
    for l in reversed(range(NUM_LAYERS)):
        grad_w_ref[l] = grad_acts[l + 1].t() @ activations[l] / batch
        grad_acts[l] = grad_acts[l + 1] @ weights_ref[l]

    weights_ref_updated = weights_ref - lr * grad_w_ref

    return {
        "out": out,
        "grad_out": grad_out,
        "grad_in": grad_input,
        "grad_w": grad_w,
        "weights": weights_out,
        "timings": timings,
        "refs": {
            "out": out_ref,
            "grad_out": grad_acts[-1],
            "grad_in": grad_acts[0],
            "grad_w": grad_w_ref,
            "weights": weights_ref_updated,
        },
    }


def _print_diffs(outputs: dict[str, torch.Tensor], refs: dict[str, torch.Tensor]) -> None:
    def diff(name: str, a: torch.Tensor, b: torch.Tensor):
        max_diff = (a - b).abs().max().item()
        print(f"{name:12s} max diff: {max_diff:.4e}")

    diff("output", outputs["out"], refs["out"])
    diff("grad_out", outputs["grad_out"], refs["grad_out"])
    diff("grad_in", outputs["grad_in"], refs["grad_in"])
    diff("grad_w", outputs["grad_w"], refs["grad_w"])
    diff("weights", outputs["weights"], refs["weights"])


def preview(name: str, tensor: torch.Tensor, num_values: int = 4) -> None:
    flattened = tensor.detach().flatten().cpu()
    slice_ = flattened[:num_values]
    formatted = " ".join(f"{val:.4f}" for val in slice_)
    print(f"[linear-training-mk-demo] {name} preview ({num_values}): {formatted}")


def summarize_timings(
    timings: torch.Tensor, num_instructions: int, timings_enabled: bool, max_instructions: int = 4
) -> None:
    host_timings = timings.detach().cpu()
    sm_dim = host_timings.shape[0]
    if sm_dim == 0:
        return
    print(
        f"[linear-training-mk-demo] timings summary (first {min(max_instructions, num_instructions)} instructions on SM0)"
    )
    for inst_idx in range(min(max_instructions, num_instructions)):
        row = host_timings[0, inst_idx]
        nonzero = row[row != 0]
        if nonzero.numel() == 0:
            if not timings_enabled:
                msg = "  inst {idx:02d}: (timing capture disabled at build time, rerun with ENABLE_TIMINGS=1)"
            else:
                msg = "  inst {idx:02d}: (timing slots all zero)"
            print(msg.format(idx=inst_idx))
            continue
        formatted = " ".join(str(int(val.item())) for val in nonzero[:8])
        print(f"  inst {inst_idx:02d}: {formatted}")


def test_linear_training_matches_torch():
    torch.manual_seed(0)

    results = run_demo_step(batch=4, lr=0.05, debug=False)

    out = results["out"]
    grad_out = results["grad_out"]
    grad_in = results["grad_in"]
    grad_w = results["grad_w"]
    weights = results["weights"]
    refs = results["refs"]

    assert torch.allclose(out, refs["out"], atol=1e-3, rtol=1e-3)
    assert torch.allclose(grad_out, refs["grad_out"], atol=1e-3, rtol=1e-3)
    assert torch.allclose(grad_in, refs["grad_in"], atol=1e-3, rtol=1e-3)
    assert torch.allclose(grad_w, refs["grad_w"], atol=1e-3, rtol=1e-3)
    assert torch.allclose(weights, refs["weights"], atol=1e-3, rtol=1e-3)


def main():
    debug_vis = env_flag("LINEAR_DEMO_DEBUG")
    timings_requested = env_flag("LINEAR_DEMO_TIMINGS") or debug_vis
    batch = int(os.environ.get("LINEAR_DEMO_BATCH", "4"))
    lr = float(os.environ.get("LINEAR_DEMO_LR", "0.05"))

    results = run_demo_step(batch=batch, lr=lr, debug=debug_vis)
    _print_diffs(
        {k: results[k] for k in ("out", "grad_out", "grad_in", "grad_w", "weights")},
        results["refs"],
    )

    if debug_vis:
        preview("output", results["out"])
        preview("grad_out", results["grad_out"])
        preview("grad_in", results["grad_in"])
        preview("grad_w", results["grad_w"])
        preview("weights", results["weights"])

    if timings_requested and results["timings"].numel() > 0:
        sm_count = results["timings"].shape[0]
        instructions_per_sm = results["timings"].shape[1]
        print(
            "[linear-training-mk-demo] instructions tensor shape:",
            (sm_count, instructions_per_sm, results["timings"].shape[2]),
        )
        summarize_timings(results["timings"], instructions_per_sm, timings_enabled=env_flag("ENABLE_TIMINGS"))

    print("Done.")


if __name__ == "__main__":
    main()
