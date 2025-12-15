import argparse
from enum import Enum
import math
import os
import statistics
from typing import List, NamedTuple

import torch
from torch.amp.autocast_mode import autocast

import linear_training_mk_demo

M_TILE = 32
N_TILE = 32
K_TILE = 32


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


def tile_range(length: int, tile: int) -> range:
    return range(div_up(length, tile))


def build_single_queue(
    batch: int, in_dim: int, out_dim: int, num_layers: int, device: torch.device
) -> torch.Tensor:
    insts: list[list[int]] = []

    for layer in range(num_layers):
        for m_block in tile_range(batch, M_TILE):
            for n_block in tile_range(out_dim, N_TILE):
                row = [0] * 32
                row[0] = 1  # LINEAR_FWD
                row[1] = layer
                row[2] = m_block
                row[3] = n_block
                insts.append(row)

    for m_block in tile_range(batch, M_TILE):
        for n_block in tile_range(out_dim, N_TILE):
            row = [0] * 32
            row[0] = 2  # LOSS_GRAD
            row[1] = num_layers - 1
            row[2] = m_block
            row[3] = n_block
            insts.append(row)

    for layer in reversed(range(num_layers)):
        for out_block in tile_range(out_dim, N_TILE):
            for in_block in tile_range(in_dim, N_TILE):
                row = [0] * 32
                row[0] = 3  # LINEAR_BWD_WEIGHT
                row[1] = layer
                row[2] = out_block
                row[3] = in_block
                insts.append(row)

        for m_block in tile_range(batch, M_TILE):
            for n_block in tile_range(in_dim, N_TILE):
                row = [0] * 32
                row[0] = 4  # LINEAR_BWD_INPUT
                row[1] = layer
                row[2] = m_block
                row[3] = n_block
                insts.append(row)

        for out_block in tile_range(out_dim, N_TILE):
            for in_block in tile_range(in_dim, N_TILE):
                row = [0] * 32
                row[0] = 5  # SGD_UPDATE
                row[1] = layer
                row[2] = out_block
                row[3] = in_block
                insts.append(row)

    return torch.tensor(insts, device=device, dtype=torch.int32)


def shard_instructions(
    per_sm_insts: torch.Tensor, sm_count: int, device: torch.device
) -> torch.Tensor:
    return (
        per_sm_insts.unsqueeze(0)
        .expand(sm_count, per_sm_insts.size(0), per_sm_insts.size(1))
        .contiguous()
    )


def gflops_forward_backward(
    batch: int, in_dim: int, out_dim: int, num_layers: int
) -> float:
    macs = 2.0 * batch * in_dim * out_dim
    per_layer = 3 * macs + 2 * out_dim * in_dim  # fwd + bwd_weight + bwd_input + sgd
    loss = 2.0 * batch * out_dim
    total_ops = num_layers * per_layer + loss
    return total_ops / 1e9


class RunOnceResult(NamedTuple):
    timings_mk: list[float]
    timings_pt: list[float]


class Scenario(Enum):
    MEGAKERNEL = 1
    PYTORCH = 2
    ALL = 3

def run_once(
    batch: int,
    in_dim: int,
    out_dim: int,
    num_layers: int,
    steps: int,
    warmup: int,
    lr: float,
    scenario: Scenario = Scenario.ALL,
) -> RunOnceResult:
    torch.manual_seed(0)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())

    x = torch.randn(batch, in_dim, device=device, dtype=torch.float32)
    target = torch.randn(batch, out_dim, device=device, dtype=torch.float32)
    # weights is the canonical parameter tensor (float32)
    weights = torch.randn(
        num_layers, out_dim, in_dim, device=device, dtype=torch.float32
    )
    weights_pt = weights.clone()
    timings_mk: List[float] = []

    # ---------- megakernel run ----------
    if scenario in (Scenario.MEGAKERNEL, Scenario.ALL):
        instructions = shard_instructions(
            build_single_queue(batch, in_dim, out_dim, num_layers, device),
            props.multi_processor_count,
            device,
        )

        for i in range(warmup + steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            if i >= warmup:
                torch.cuda.nvtx.range_push(f"megakernel-run-{i}")
                start.record()

            linear_training_mk_demo.run(instructions, x, target, weights, lr, False)

            if i >= warmup:
                end.record()
                torch.cuda.nvtx.range_pop()
                elapsed = start.elapsed_time(end)
                timings_mk.append(elapsed)

    # ---------- PyTorch training run ----------
    timings_pt: List[float] = []
    if scenario in (Scenario.PYTORCH, Scenario.ALL):
        weights_pt.requires_grad_(True)
        x.requires_grad_(True)
        opt = torch.optim.SGD(params=[weights_pt], lr=lr)

        for i in range(warmup + steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            if i >= warmup:
                torch.cuda.nvtx.range_push(f"torch-run-{i}")
                start.record()

            opt.zero_grad()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                out = x
                for l in range(num_layers):
                    out = torch.nn.functional.linear(input=out, weight=weights_pt[l])
                    # out = out @ weights_pt[l].t()
                    # act = act @ weights[l].t()
                loss = torch.nn.functional.mse_loss(input=out, target=target)

                # typically not done in autocast but done to be a bit closer to the mk path
                loss.backward()
                opt.step()
                # loss = ((out.to(torch.float32) - target) ** 2).sum() / batch

            if i >= warmup:
                end.record()
                torch.cuda.nvtx.range_pop()
                elapsed = start.elapsed_time(end)
                timings_pt.append(elapsed)

    return RunOnceResult(timings_mk=timings_mk, timings_pt=timings_pt)


def run_grid():
    def show(timings_ms: List[float], name: str, batch: int, in_dim: int, out_dim: int, num_layers: int) -> None:
        if not timings_ms:
            print(f"[{name}] no timings collected")
            return

        n = len(timings_ms)
        avg_ms = statistics.mean(timings_ms)
        p50 = statistics.median(timings_ms)
        sorted_ts = sorted(timings_ms)
        idx95 = min(n - 1, math.floor(0.95 * n))
        p95 = sorted_ts[idx95]

        gflops = gflops_forward_backward(
            batch, in_dim, out_dim, num_layers
        )
        # giga-ops per run (total_ops / 1e9) to tflops = (gigaops / seconds) / 1000
        tflops = (gflops / (avg_ms / 1000.0)) / 1000.0

        summary = (
            f"[{name}] batch={batch} in_dim={in_dim} out_dim={out_dim} "
            f"layers={num_layers} avg_ms={avg_ms:.3f} p50_ms={p50:.3f} "
            f"p95_ms={p95:.3f} est_tflops={tflops:.3f}"
        )
        print(summary)
    batches = [b for b in [4, 16, 64] if b % M_TILE == 0]
    dims = [d for d in [64, 256, 1024] if d % K_TILE == 0]
    layers = [2, 4, 8]

    for batch in batches:
        for dim in dims:
            in_dim = dim
            out_dim = dim
            for num_layers in layers:
                print()
                result = run_once(
                    batch,
                    in_dim,
                    out_dim,
                    num_layers,
                    steps=10,
                    warmup=2,
                    lr=0.01,
                    scenario=Scenario.ALL,
                )
                show(result.timings_mk, name="mk", batch=batch, in_dim=in_dim, out_dim=out_dim, num_layers=num_layers)
                show(result.timings_pt, name="pt", batch=batch, in_dim=in_dim, out_dim=out_dim, num_layers=num_layers)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the linear training megakernel."
    )
    sp = parser.add_subparsers(dest="mode", required=True)
    all_sp = sp.add_parser("all")
    one_sp = sp.add_parser("one")

    one_sp.add_argument(
        "--batch",
        type=int,
        default=env_int("LINEAR_BENCH_BATCH", 4),
        help="Batch size (env: LINEAR_BENCH_BATCH)",
    )
    one_sp.add_argument(
        "--in-dim",
        type=int,
        default=env_int("LINEAR_BENCH_IN_DIM", 64),
        help="Input dimension (env: LINEAR_BENCH_IN_DIM)",
    )
    one_sp.add_argument(
        "--out-dim",
        type=int,
        default=env_int("LINEAR_BENCH_OUT_DIM", 64),
        help="Output dimension (env: LINEAR_BENCH_OUT_DIM)",
    )
    one_sp.add_argument(
        "--num-layers",
        type=int,
        default=env_int("LINEAR_BENCH_NUM_LAYERS", 2),
        help="Number of layers (env: LINEAR_BENCH_NUM_LAYERS)",
    )
    one_sp.add_argument(
        "--steps",
        type=int,
        default=env_int("LINEAR_BENCH_STEPS", 10),
        help="Benchmark steps (env: LINEAR_BENCH_STEPS)",
    )
    one_sp.add_argument(
        "--warmup",
        type=int,
        default=env_int("LINEAR_BENCH_WARMUP", 2),
        help="Warmup iterations (env: LINEAR_BENCH_WARMUP)",
    )
    one_sp.add_argument(
        "--lr",
        type=float,
        default=env_float("LINEAR_BENCH_LR", 0.01),
        help="Learning rate (env: LINEAR_BENCH_LR)",
    )
    one_sp.add_argument(
        "--pt",
        action="store_true",
        required=False,
        help="Only run pytorch",
    )
    one_sp.add_argument(
        "--mk",
        action="store_true",
        required=False,
        help="Only run megakernel",
    )
    one_sp.add_argument(
        "-d",
        action="store_true",
        required=False,
        help="Dry run",
    )
    args = parser.parse_args()
    if args.mode == "all":
        run_grid()
        return
    assert not (args.pt and args.mk), "--pt and --mk are mutually exclusive"
    if args.d:
        print("Dry run, nothing to do.")
        return

    def show(timings_ms: List[float], name: str) -> None:
        if not timings_ms:
            print(f"[{name}] no timings collected")
            return

        n = len(timings_ms)
        avg_ms = statistics.mean(timings_ms)
        p50 = statistics.median(timings_ms)
        sorted_ts = sorted(timings_ms)
        idx95 = min(n - 1, math.floor(0.95 * n))
        p95 = sorted_ts[idx95]

        gflops = gflops_forward_backward(
            args.batch, args.in_dim, args.out_dim, args.num_layers
        )
        # giga-ops per run (total_ops / 1e9) to tflops = (gigaops / seconds) / 1000
        tflops = (gflops / (avg_ms / 1000.0)) / 1000.0

        summary = (
            f"[{name}] batch={args.batch} in_dim={args.in_dim} out_dim={args.out_dim} "
            f"layers={args.num_layers} steps={args.steps} warmup={args.warmup} "
            f"avg_ms={avg_ms:.3f} p50_ms={p50:.3f} p95_ms={p95:.3f} est_tflops={tflops:.3f}"
        )
        print(summary)

    scenario = Scenario.MEGAKERNEL if args.mk else Scenario.PYTORCH if args.pt else Scenario.ALL
    result = run_once(
        args.batch,
        args.in_dim,
        args.out_dim,
        args.num_layers,
        args.steps,
        args.warmup,
        args.lr,
        scenario=scenario,
    )
    if scenario in (Scenario.MEGAKERNEL, Scenario.ALL):
        show(result.timings_mk, name="mk")
    if scenario in (Scenario.PYTORCH, Scenario.ALL):
        show(result.timings_pt, name="pt")


if __name__ == "__main__":
    main()
