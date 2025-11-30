from enum import IntEnum
import os
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import linear_training


IN_DIM = 128
OUT_DIM = 64
BLOCK = 16
IN_BLOCKS = IN_DIM // BLOCK
OUT_BLOCKS = OUT_DIM // BLOCK


class OpCodes(IntEnum):
    LINEAR_FWD = 1
    LOSS_GRAD = 2
    LINEAR_BWD_WEIGHT = 3
    SGD_UPDATE = 4

def env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.lower() not in ("0", "false", "")


def _build_single_queue(device: torch.device) -> torch.Tensor:
    insts: list[list[int]] = []

    # Forward tiles
    for out_block in range(OUT_BLOCKS):
        for in_block in range(IN_BLOCKS):
            row = [0] * 32
            row[0] = OpCodes.LINEAR_FWD.value  # LinearFwd opcode
            row[2] = out_block
            row[3] = in_block
            insts.append(row)

    # Loss + grad_out per output block
    for out_block in range(OUT_BLOCKS):
        row = [0] * 32
        row[0] = OpCodes.LOSS_GRAD.value  # LossGrad opcode
        row[2] = out_block
        insts.append(row)

    # Grad weights tiles
    for out_block in range(OUT_BLOCKS):
        for in_block in range(IN_BLOCKS):
            row = [0] * 32
            row[0] = OpCodes.LINEAR_BWD_WEIGHT.value  # LinearBwdWeight opcode
            row[2] = out_block
            row[3] = in_block
            insts.append(row)

    # SGD update tiles
    for out_block in range(OUT_BLOCKS):
        for in_block in range(IN_BLOCKS):
            row = [0] * 32
            row[0] = OpCodes.SGD_UPDATE.value  # SgdUpdate opcode
            row[2] = out_block
            row[3] = in_block
            insts.append(row)

    return torch.tensor(insts, device=device, dtype=torch.int32)


def build_instructions(device: torch.device, sm_count: int) -> torch.Tensor:
    per_sm_insts = _build_single_queue(device)
    expanded = (
        per_sm_insts.unsqueeze(0)
        .expand(sm_count, per_sm_insts.size(0), per_sm_insts.size(1))
        .contiguous()
    )
    return expanded


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    debug_vis = env_flag("LINEAR_DEMO_DEBUG")
    timings_requested = env_flag("LINEAR_DEMO_TIMINGS") or debug_vis
    timings_build_enabled = env_flag("ENABLE_TIMINGS")

    batch = 4
    lr = 0.05

    x = torch.randn(batch, IN_DIM, device=device, dtype=torch.float32)
    target = torch.randn(batch, OUT_DIM, device=device, dtype=torch.float32)
    weights = torch.randn(OUT_DIM, IN_DIM, device=device, dtype=torch.float32)
    weights_ref = weights.clone()

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    instructions = build_instructions(device, props.multi_processor_count)
    if debug_vis:
        print(
            "[linear-demo] instructions tensor shape:",
            tuple(instructions.shape),
            "| expected depth >= SM count",
            props.multi_processor_count,
        )
        print(
            "[linear-demo] tensor dims are [sm_count, num_instructions, 32]; repeated queues keep controller indexing in-bounds"
        )

    out, grad_out, grad_w, timings = linear_training.run_linear_training(
        instructions, x, target, weights, lr, debug_vis=debug_vis
    )
    torch.cuda.synchronize()

    # torch reference
    out_ref = x @ weights_ref.t()
    grad_out_ref = 2.0 * (out_ref - target) / batch
    grad_w_ref = grad_out_ref.t() @ x / batch
    weights_ref_updated = weights_ref - lr * grad_w_ref

    def diff(name: str, a: torch.Tensor, b: torch.Tensor):
        max_diff = (a - b).abs().max().item()
        print(f"{name:12s} max diff: {max_diff:.4e}")

    diff("output", out, out_ref)
    diff("grad_out", grad_out, grad_out_ref)
    diff("grad_w", grad_w, grad_w_ref)
    diff("weights", weights, weights_ref_updated)

    if debug_vis:
        preview("output", out)
        preview("grad_out", grad_out)
        preview("grad_w", grad_w)
        preview("weights", weights)

    if timings_requested and timings.numel() > 0:
        summarize_timings(timings, instructions.shape[1], timings_build_enabled)

    print("Done.")


def preview(name: str, tensor: torch.Tensor, num_values: int = 4) -> None:
    flattened = tensor.detach().flatten().cpu()
    slice_ = flattened[:num_values]
    formatted = " ".join(f"{val:.4f}" for val in slice_)
    print(f"[linear-demo] {name} preview ({num_values}): {formatted}")


def summarize_timings(
    timings: torch.Tensor, num_instructions: int, timings_enabled: bool, max_instructions: int = 4
) -> None:
    host_timings = timings.detach().cpu()
    sm_dim = host_timings.shape[0]
    if sm_dim == 0:
        return
    print(
        f"[linear-demo] timings summary (first {min(max_instructions, num_instructions)} instructions on SM0)"
    )
    for inst_idx in range(min(max_instructions, num_instructions)):
        row = host_timings[0, inst_idx]
        nonzero = row[row != 0]
        if nonzero.numel() == 0:
            if not timings_enabled:
                msg = "  inst {idx:02d}: (timing capture disabled at build time, rerun with --timings)"
            else:
                msg = "  inst {idx:02d}: (timing slots all zero)"
            print(msg.format(idx=inst_idx))
            continue
        formatted = " ".join(str(int(val.item())) for val in nonzero[:8])
        print(f"  inst {inst_idx:02d}: {formatted}")


if __name__ == "__main__":
    main()
