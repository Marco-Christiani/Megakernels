# Performance & Profiling

## Nsight Systems (trace)

```sh
# run only mk
nsys profile -t cuda,nvtx --sample=none -o profiles/bm_mk -- uv run benchmarks/benchmark.py --mk
# run only torch
nsys profile -t cuda,nvtx --sample=none -o profiles/bm_pt -- uv run benchmarks/benchmark.py --pt

# convert to plaintext report
nsys stats profiles/bm_mk.nsys-rep > profiles/basic_mk.stats.txt
nsys stats profiles/bm_pt.nsys-rep > profiles/basic_pt.stats.txt
```

## Nsight Compute (metrics)


```sh
sudo ncu --section SpeedOfLight --csv --target-processes all -- $HOME/.local/bin/uv run benchmarks/benchmark.py --mk > profiles/basic_mk.ncu_sol.csv
sudo ncu --section SpeedOfLight --csv --target-processes all -- $HOME/.local/bin/uv run benchmarks/benchmark.py --pt > profiles/basic_pt.ncu_sol.csv

# Targeting metrics and kernel name
sudo ncu \
  --kernel-name-base demangled \
  -k 'regex:megakernel' \
  --metrics 'sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__pipe_tensor_op_hmfu_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__sass_average_branch_targets_threads_uniform.pct,smsp__warp_issue_stalled_barrier_per_warp_active,lts__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tma_cycles_active.avg' \
  -o profiles/bm.ncu_custom -f \
  -- $HOME/.local/bin/uv run benchmarks/benchmark.py --mk

# For pytorch use --pt and -k 'regex:at::'

# NCU SOL
sudo ncu --section SpeedOfLight --kernel-name-base demangled \
    -k "regex:megakernel" \
    --target-processes all \
    --csv \
    --force-overwrite \
    -o profiles/mk.ncu_sol \
    $HOME/.local/bin/uv run benchmarks/benchmark.py --mk

```

Notes and resources:

  - SM utilization/occupancy: sm__throughput
  - Tensor Core activity: smsp__pipe_tensor_op_hmfu_active / sm__pipe_tensor_op_hmma_cycles_active
  - memory throughput: dram__/lts__
  - stall reasons: warp_issue_stalled_barrier
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#warp-stall-reasons
    - not issued: https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#warp-stall-reasons-not-issued
  - TMA activity: sm__pipe_tma_cycles_active
  - Sets, sections, rules
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#sets-and-sections
  - Occupancy
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#occupancy-metrics
  - Metric groups
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#metric-groups
  - Timeline / PM Sampling
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#sampling
  - Roofline chart analysis
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#roofline-charts
  - Memory Chart
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#memory-chart
  - Memory Tables
    - https://docs.nvidia.com/nsight-compute/2025.4/ProfilingGuide/index.html#memory-tables


---

## See also

- MOC - [[README]] 
- Current invariants - [[CONTRACTS]] 
- Current design and plan - [[TRAINING_IR]]
- Historical design context - [[MK_REPORT]]
- pipeline reference - [[NOTES]]