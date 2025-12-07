```plaintext
GMEM (padded)
  activations[layer, 1, batch, dim]
  grad_activations[layer, 1, batch, dim]
  weights[layer, 1, out_dim, in_dim]
  grad_w[layer, 1, out_dim, in_dim]
       |
       |  (TMA two-tile loads per K block)
       v
SMEM tiles (A = MxK, B = NxK or KxN)
       |
       |  (warp::load)
       v
register fragments (rt_*)
       |
       |  (warp::mma_AB / mma_ABt / mma_AtB)
       v
register accumulators (MxN)
       |
       |  (lane-strided store back to GMEM)
       v
GMEM (next activations / grad / grad_w)
```

### Instructions
- Forward layer l: tiles over `[batch, out_dim]` using `LINEAR_FWD`.
- Loss: `LOSS_GRAD` tiles over `[batch, out_dim_last]`.
- Backward per layer from top to bottom: `LINEAR_BWD_WEIGHT` over `[out_dim, in_dim]`, `LINEAR_BWD_INPUT` over `[batch, in_dim]`, then `SGD_UPDATE` on the same grid.

### Loader/consumer contract
- Two logical pages per op (A/B) + two semaphores. Loader waits on both pages, then for each K tile: `tma::expect_bytes` + `tma::load_async` for A/B, waits on parity, `__syncthreads()` to keep consumers in lockstep.
- Consumers mirror the K loop: wait on parity, `warp::load` -> `warp::mma_*` accumulate, then store masked by live dims. Every consumer warp issues `warp_finish_page` for both logical pages once the loop completes.


---

## Notes

Parity is derived from instruction_index bits. If ANY instruction under- or over-arrives on its expected page_finished count, later waiters can block forever on a parity value that will never be reached.


```c++
// include/util.cuh
__device__ inline void wait_page_ready(int pid) {
#pragma unroll
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
        auto bit = (instruction_index >> i) & 1;
        kittens::wait(page_finished[pid][i], bit);
    }
}
```

Advances parity only after the full expected arrival count for this instruction is met. Mismatched arrivals desynchronize parity from instruction_index and strand later waiters.

```c++
// include/util.cuh
__device__ inline void finish_page(int pid, int count) {
#pragma unroll
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
        arrive(page_finished[pid][i], count);
    }
}
```