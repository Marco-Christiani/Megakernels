import math
import os
from enum import IntEnum

import torch
from torch import inference_mode

import linear_training_mk_demo

IN_DIM = 64
OUT_DIM = 64
NUM_LAYERS = 2

M_TILE = 16
N_TILE = 16

# bf16_manual or bf16_autograd
REF_MODE = os.environ.get("LINEAR_REF_MODE", "bf16_manual")


class OpCodes(IntEnum):
    LINEAR_FWD = 1
    LOSS_GRAD = 2
    LINEAR_BWD_WEIGHT = 3
    LINEAR_BWD_INPUT = 4
    SGD_UPDATE = 5


def env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.lower() not in ("0", "false", "")


def matmul_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MM mimicking tensor core behavior:
    - inputs quantized to bf16
    - fp32 accumulation (no intermediate rounding)
    - result in fp32
    """
    a_bf = a.to(torch.bfloat16).to(torch.float32)  # back to fp32
    b_bf = b.to(torch.bfloat16).to(torch.float32)
    return a_bf @ b_bf  # fp32 matmul on bf16-quantized values


def linear_forward_bf16(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """y = x @ W.T, returning fp32."""
    return matmul_bf16(x, W.transpose(-1, -2))


def linear_backward_bf16(
    x: torch.Tensor, W: torch.Tensor, grad_out: torch.Tensor, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
      grad_w = grad_out.T @ x / batch
      grad_in = grad_out @ W
    where all matmuls are in bf16, results in fp32.
    """
    grad_w = matmul_bf16(grad_out.transpose(-1, -2), x) / batch
    grad_in = matmul_bf16(grad_out, W)
    return grad_in, grad_w


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


@inference_mode()
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

    if REF_MODE == "bf16_manual":
        activations: list[torch.Tensor] = [x]
        for l in range(NUM_LAYERS):
            activations.append(linear_forward_bf16(activations[-1], weights_ref[l]))

        out_ref = activations[-1]

        grad_acts: list[torch.Tensor | None] = [None] * (NUM_LAYERS + 1)
        grad_acts[-1] = 2.0 * (out_ref - target) / batch

        grad_w_ref = torch.zeros_like(weights_ref)
        for l in reversed(range(NUM_LAYERS)):
            grad_in_l, grad_w_l = linear_backward_bf16(
                activations[l], weights_ref[l], grad_acts[l + 1], batch
            )
            grad_acts[l] = grad_in_l
            grad_w_ref[l] = grad_w_l

        grad_out_ref = grad_acts[-1]
        grad_in_ref = grad_acts[0]
        weights_ref_updated = weights_ref - lr * grad_w_ref

    elif REF_MODE == "bf16_autograd":
        x_bf = x.to(torch.bfloat16).detach().requires_grad_(True)
        weights_bf = weights_ref.to(torch.bfloat16).detach().requires_grad_(True)

        act = x_bf
        for l in range(NUM_LAYERS):
            act = act @ weights_bf[l].t()
        out_bf = act
        out_bf.retain_grad()

        loss = ((out_bf.to(torch.float32) - target) ** 2).sum() / batch
        loss.backward()

        out_ref = out_bf.to(torch.float32)
        grad_out_ref = out_bf.grad.to(torch.float32)
        grad_in_ref = x_bf.grad.to(torch.float32)
        grad_w_ref = weights_bf.grad.to(torch.float32)
        weights_ref_updated = weights_ref - lr * grad_w_ref

    else:
        raise ValueError(f"Unknown REF_MODE: {REF_MODE}")

    return {
        "out": out,
        "grad_out": grad_out,
        "grad_in": grad_input,
        "grad_w": grad_w,
        "weights": weights_out,
        "timings": timings,
        "weights_init": weights_ref,
        "refs": {
            "out": out_ref,
            "grad_out": grad_out_ref,
            "grad_in": grad_in_ref,
            "grad_w": grad_w_ref,
            "weights": weights_ref_updated,
        },
    }


@inference_mode()
def _print_diffs(outputs: dict[str, torch.Tensor], refs: dict[str, torch.Tensor]) -> None:
    def diff(name: str, a: torch.Tensor, b: torch.Tensor):
        diff_tensor = a - b
        abs_diff = diff_tensor.abs()
        max_diff_val = abs_diff.max().item()
        mae = abs_diff.mean()
        mse = diff_tensor.square().mean()
        std = diff_tensor.std()

        # find worst element
        flat_idx = abs_diff.view(-1).argmax()
        unraveled = torch.unravel_index(flat_idx, abs_diff.shape)
        idx = tuple(int(i) for i in unraveled)
        a_val = a[idx].item()
        b_val = b[idx].item()

        print(
            f"{name:12s} "
            # f"{a.div(b).max():.4} "
            f"max abs diff: {max_diff_val:.4} "
            f"mean abs diff: {mae:.4} "
            f"mse: {mse:.4f} "
            f"diff std: {std:.4} "
            f"@idx={idx} (mk={a_val:.4e}, ref={b_val:.4e})"
        )

        # For 3D tensors like grad_w / weights (L, Out, In) also compare
        # against a transposed reference to detect pure layout bugs
        if a.ndim == 3 and b.shape == a.shape:
            b_T = b.transpose(1, 2).contiguous()
            trans_diff = (a - b_T).abs()
            trans_max = trans_diff.max().item()
            print(f"{name:12s} transposed-ref max abs diff: {trans_max:.4}")

    diff("output", outputs["out"], refs["out"])
    diff("grad_out", outputs["grad_out"], refs["grad_out"])
    diff("grad_in", outputs["grad_in"], refs["grad_in"])
    diff("grad_w", outputs["grad_w"], refs["grad_w"])
    diff("weights", outputs["weights"], refs["weights"])


@inference_mode()
def preview(name: str, tensor: torch.Tensor, num_values: int = 4) -> None:
    flattened = tensor.detach().flatten().cpu()
    slice_ = flattened[:num_values]
    formatted = " ".join(f"{val:.4f}" for val in slice_)
    print(f"[linear-training-mk-demo] {name} preview ({num_values}): {formatted}")


@inference_mode()
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

    results = run_demo_step(batch=4, lr=0.01, debug=False)

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


def test_linear_training_bf16_trains_down_loss():
    """
    End-to-end training sanity check in a bf16-style regime.

    We:
      - construct a small, exactly learnable 2-layer linear problem,
      - generate targets using explicit bf16 matmuls,
      - run the megakernel SGD for several steps,
      - and assert that the loss decreases.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    batch = 4
    lr = 1e-2

    # Inputs
    x = torch.randn(batch, IN_DIM, device=device, dtype=torch.float32)

    # True weights in bf16, small scale for stability.
    true_weights_bf = (
        torch.randn(NUM_LAYERS, OUT_DIM, IN_DIM, device=device, dtype=torch.float32) * 0.25
    ).to(torch.bfloat16)

    # Generate targets via explicit bf16 matmuls.
    act = x
    for l in range(NUM_LAYERS):
        # W: [out, in], need W.T for [in, out] to do x @ W.T
        act = matmul_bf16(act, true_weights_bf[l].to(torch.float32).transpose(-1, -2))
    target = act  # fp32, but came from bf16 math

    # Initialize trainable weights near zero.
    weights = (
        torch.randn(NUM_LAYERS, OUT_DIM, IN_DIM, device=device, dtype=torch.float32) * 0.05
    )

    # Build instructions for this batch once.
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    instructions = build_instructions(device, props.multi_processor_count, batch)

    losses = []
    num_steps = 16
    for _ in range(num_steps):
        out, _, _, _, _, _ = linear_training_mk_demo.run(
            instructions, x, target, weights, lr, False
        )
        torch.cuda.synchronize()
        loss = ((out - target) ** 2).mean()
        losses.append(loss.item())

    # Require that training actually reduces loss over time.
    assert losses[-1] < losses[0], f"bf16-style training did not reduce loss {losses}"

def main():
    debug_vis = env_flag("LINEAR_DEMO_DEBUG")
    timings_requested = env_flag("LINEAR_DEMO_TIMINGS") or debug_vis
    batch = int(os.environ.get("LINEAR_DEMO_BATCH", "4"))
    lr = float(os.environ.get("LINEAR_DEMO_LR", "0.05"))

    results = run_demo_step(batch=batch, lr=lr, debug=debug_vis)

    # Check whether SGD update matches applying lr * grad_w ONCE to the
    # initial weights. Yeah, I have a double apply bug..
    with torch.no_grad():
        weights_init = results["weights_init"]
        grad_w = results["grad_w"]
        weights_out = results["weights"]
        weights_pred = weights_init - lr * grad_w
        delta = (weights_out - weights_pred).abs()
        print(
            "weights_vs_pred max abs diff:",
            f"{delta.max().item():.4e}",
        )

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

    test_linear_training_bf16_trains_down_loss()

    print("Done.")


if __name__ == "__main__":
    main()
