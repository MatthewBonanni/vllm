# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: dynamic token_to_req lookup vs static tl.static_range loop
for _scatter_topk_kernel in sparse MLA attention.

Measures:
  1. Warm kernel latency (both approaches, averaged over many iters)
  2. Cold JIT cost (original: recompiles per unique NUM_REQS; new: compiles once)

DeepSeek-V3 dimensions:
  max_seq_len = 4096 (context window slice used in topk mask)
  num_topk    = 64   (top-k sparse indices per query token)

Usage:
  python benchmarks/kernels/bench_scatter_topk.py
  python benchmarks/kernels/bench_scatter_topk.py --warmup 20 --iters 200

Results (warmup=10, iters=100, H100, CUDA 13.0, torch 2.10.0):
  Correctness check passed.

  [ Warm kernel latency ]
      B |  q/req |  total_q |  orig (ms) |   new (ms) |  speedup |  saved (ms)
  ------------------------------------------------------------------------
      1 |     64 |       64 |     0.0204 |     0.0201 |    1.01x |     +0.0002
      2 |     64 |      128 |     0.0205 |     0.0199 |    1.03x |     +0.0005
      4 |     64 |      256 |     0.0202 |     0.0201 |    1.00x |     +0.0001
      8 |     64 |      512 |     0.0215 |     0.0201 |    1.07x |     +0.0014
     16 |     64 |     1024 |     0.0205 |     0.0200 |    1.02x |     +0.0005
     32 |     64 |     2048 |     0.0201 |     0.0200 |    1.00x |     +0.0000
     64 |     64 |     4096 |     0.0201 |     0.0203 |    0.99x |     -0.0002

      1 |    256 |      256 |     0.0197 |     0.0203 |    0.97x |     -0.0005
      2 |    256 |      512 |     0.0211 |     0.0200 |    1.05x |     +0.0011
      4 |    256 |     1024 |     0.0206 |     0.0199 |    1.03x |     +0.0006
      8 |    256 |     2048 |     0.0199 |     0.0200 |    1.00x |     -0.0001
     16 |    256 |     4096 |     0.0201 |     0.0201 |    1.00x |     -0.0001
     32 |    256 |     8192 |     0.0200 |     0.0203 |    0.99x |     -0.0003
     64 |    256 |   16384  |     0.0497 |     0.0202 |    2.47x |     +0.0295

      1 |   1024 |     1024 |     0.0200 |     0.0209 |    0.96x |     -0.0009
      2 |   1024 |     2048 |     0.0209 |     0.0197 |    1.06x |     +0.0012
      4 |   1024 |     4096 |     0.0197 |     0.0197 |    1.00x |     -0.0000
      8 |   1024 |     8192 |     0.0202 |     0.0200 |    1.01x |     +0.0003
     16 |   1024 |    16384 |     0.0243 |     0.0208 |    1.17x |     +0.0035
     32 |   1024 |    32768 |     0.0553 |     0.0330 |    1.67x |     +0.0223
     64 |   1024 |    65536 |     0.1942 |     0.0780 |    2.49x |     +0.1162

  Warm speedup summary:
    Min:  0.96x  Max:  2.49x  Mean: 1.19x

  [ Cold JIT cost — first call on a previously-unseen batch size ]
      B |  q/req |  orig cold (ms) |  new cold (ms) |  JIT saved (ms)
  -----------------------------------------------------------------
      3 |    256 |           45.73 |           0.08 |          +45.65
      5 |    256 |           49.49 |           0.08 |          +49.41
      7 |    256 |           53.25 |           0.09 |          +53.16
      9 |    256 |           53.26 |           0.09 |          +53.17
     11 |    256 |           56.45 |           0.09 |          +56.36
     13 |    256 |           58.88 |           0.09 |          +58.79
     17 |    256 |           73.61 |           0.10 |          +73.51
     19 |    256 |           67.58 |           0.10 |          +67.49

  Key takeaways:
    - Warm latency: neutral at small batch/token counts; up to 2.49x faster at
      large batch x long sequences (B=64, q/req=1024, 65K tokens total).
    - Cold JIT: original stalls 45–74 ms per unseen batch size due to Triton
      recompilation of the static_range loop. New kernel compiles once and
      hits cache for every subsequent batch size (~600x cold-call speedup).
"""

import argparse
import time

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# DeepSeek-V3 representative constants
# ---------------------------------------------------------------------------
NUM_TOPK = 64
MAX_SEQ_LEN = 4096  # max KV length used when building the topk mask


# ---------------------------------------------------------------------------
# Original kernel — NUM_REQS is tl.constexpr, uses tl.static_range
# Recompiles for every unique batch size.
# ---------------------------------------------------------------------------


@triton.jit
def _scatter_topk_kernel_orig(
    mask_ptr,
    topk_ptr,
    cu_q_lens_ptr,
    num_words: tl.constexpr,
    num_topk: tl.constexpr,
    topk_stride: tl.constexpr,
    max_q_len: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    NUM_REQS: tl.constexpr,
):
    row_idx = tl.program_id(0)

    b: tl.int32 = 0
    for i in tl.static_range(NUM_REQS):
        next_start = tl.load(cu_q_lens_ptr + i + 1)
        b += tl.where(next_start <= row_idx, 1, 0)

    q_start = tl.load(cu_q_lens_ptr + b)
    q_local = row_idx - q_start

    topk_row_ptr = topk_ptr + row_idx * topk_stride
    offsets = tl.arange(0, BLOCK_TOPK)
    in_range = offsets < num_topk
    indices = tl.load(topk_row_ptr + offsets, mask=in_range, other=-1)

    valid = in_range & (indices >= 0)
    word_indices = indices >> 5
    bit_indices = indices & 31
    bits = (1 << bit_indices).to(tl.int32)

    mask_row_ptr = mask_ptr + (b * max_q_len + q_local) * num_words
    tl.atomic_or(mask_row_ptr + word_indices, bits, mask=valid)


# ---------------------------------------------------------------------------
# New kernel — token_to_req_ptr replaces the static loop.
# Compiles once; batch size is a runtime value, not constexpr.
# ---------------------------------------------------------------------------


@triton.jit
def _scatter_topk_kernel_new(
    mask_ptr,
    topk_ptr,
    cu_q_lens_ptr,
    token_to_req_ptr,
    num_words: tl.constexpr,
    num_topk: tl.constexpr,
    topk_stride: tl.constexpr,
    max_q_len: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row_idx = tl.program_id(0)

    b = tl.load(token_to_req_ptr + row_idx).to(tl.int32)

    q_start = tl.load(cu_q_lens_ptr + b)
    q_local = row_idx - q_start

    topk_row_ptr = topk_ptr + row_idx * topk_stride
    offsets = tl.arange(0, BLOCK_TOPK)
    in_range = offsets < num_topk
    indices = tl.load(topk_row_ptr + offsets, mask=in_range, other=-1)

    valid = in_range & (indices >= 0)
    word_indices = indices >> 5
    bit_indices = indices & 31
    bits = (1 << bit_indices).to(tl.int32)

    mask_row_ptr = mask_ptr + (b * max_q_len + q_local) * num_words
    tl.atomic_or(mask_row_ptr + word_indices, bits, mask=valid)


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------


def _make_inputs(
    B: int,
    q_per_req: int,
    num_topk: int,
    max_seq_len: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,  # mask      (B, max_q_len, num_words)
    torch.Tensor,  # topk_packed (total_q, num_topk)
    torch.Tensor,  # cu_q_lens   (B+1,)
    torch.Tensor,  # token_to_req (total_q,)
]:
    total_q = B * q_per_req
    num_words = (max_seq_len + 31) // 32

    mask = torch.zeros(B, q_per_req, num_words, dtype=torch.int32, device=device)
    topk_packed = torch.randint(
        0, max_seq_len, (total_q, num_topk), dtype=torch.int32, device=device
    )
    q_lens_t = torch.full((B,), q_per_req, dtype=torch.int32, device=device)
    cu_q_lens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(q_lens_t, dim=0, out=cu_q_lens[1:])
    token_to_req = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32, device=device),
        q_lens_t,
    )
    return mask, topk_packed, cu_q_lens, token_to_req


def _call_orig(mask, topk_packed, cu_q_lens, B: int, max_q_len: int) -> None:
    total_q = topk_packed.shape[0]
    num_topk = topk_packed.shape[1]
    num_words = mask.shape[2]
    BLOCK_TOPK = triton.next_power_of_2(num_topk)
    mask.zero_()
    _scatter_topk_kernel_orig[(total_q,)](
        mask,
        topk_packed,
        cu_q_lens,
        num_words=num_words,
        num_topk=num_topk,
        topk_stride=topk_packed.stride(0),
        max_q_len=max_q_len,
        BLOCK_TOPK=BLOCK_TOPK,
        NUM_REQS=B,
    )


def _call_new(mask, topk_packed, cu_q_lens, token_to_req, max_q_len: int) -> None:
    total_q = topk_packed.shape[0]
    num_topk = topk_packed.shape[1]
    num_words = mask.shape[2]
    BLOCK_TOPK = triton.next_power_of_2(num_topk)
    mask.zero_()
    _scatter_topk_kernel_new[(total_q,)](
        mask,
        topk_packed,
        cu_q_lens,
        token_to_req,
        num_words=num_words,
        num_topk=num_topk,
        topk_stride=topk_packed.stride(0),
        max_q_len=max_q_len,
        BLOCK_TOPK=BLOCK_TOPK,
    )


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def _bench(fn, num_warmup: int, num_iters: int) -> float:
    """Warm up, then return mean latency in ms."""
    for _ in range(num_warmup):
        fn()
    torch.accelerator.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / num_iters * 1000


def _cold_call_ms(fn) -> float:
    """Time a single kernel call including JIT compilation if not cached."""
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def _check_correctness(device: torch.device) -> None:
    B, q_per_req = 4, 32
    mask_o, topk_packed, cu_q_lens, token_to_req = _make_inputs(
        B, q_per_req, NUM_TOPK, MAX_SEQ_LEN, device
    )
    mask_n = mask_o.clone()

    _call_orig(mask_o, topk_packed, cu_q_lens, B, q_per_req)
    _call_new(mask_n, topk_packed, cu_q_lens, token_to_req, q_per_req)

    assert torch.equal(mask_o, mask_n), (
        "Correctness check FAILED: original and new kernels produced different masks"
    )
    print("Correctness check passed.")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_benchmark(num_warmup: int, num_iters: int) -> None:
    device = torch.device("cuda")

    _check_correctness(device)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    q_per_req_vals = [64, 256, 1024]

    print()
    print("=" * 100)
    print(
        "Benchmark: _scatter_topk_kernel  —  "
        "static tl.static_range(NUM_REQS) vs dynamic token_to_req lookup"
    )
    print("=" * 100)
    print(
        f"num_topk={NUM_TOPK}, max_seq_len={MAX_SEQ_LEN}, "
        f"warmup={num_warmup}, iters={num_iters}"
    )
    print()

    # ── Warm latency table ──────────────────────────────────────────────────
    print("[ Warm kernel latency ]")
    print(
        f"{'B':>5} | {'q/req':>6} | {'total_q':>8} | "
        f"{'orig (ms)':>10} | {'new (ms)':>10} | {'speedup':>8} | {'saved (ms)':>11}"
    )
    print("-" * 72)

    all_speedups: list[float] = []

    for q_per_req in q_per_req_vals:
        for B in batch_sizes:
            mask, topk_packed, cu_q_lens, token_to_req = _make_inputs(
                B, q_per_req, NUM_TOPK, MAX_SEQ_LEN, device
            )
            total_q = B * q_per_req

            _orig_args = (mask, topk_packed, cu_q_lens, B, q_per_req)
            orig_ms = _bench(
                lambda a=_orig_args: _call_orig(*a),
                num_warmup,
                num_iters,
            )
            _new_args = (mask, topk_packed, cu_q_lens, token_to_req, q_per_req)
            new_ms = _bench(
                lambda a=_new_args: _call_new(*a),
                num_warmup,
                num_iters,
            )

            speedup = orig_ms / new_ms if new_ms > 0 else float("inf")
            saved = orig_ms - new_ms
            all_speedups.append(speedup)

            print(
                f"{B:>5} | {q_per_req:>6} | {total_q:>8} | "
                f"{orig_ms:>10.4f} | {new_ms:>10.4f} | "
                f"{speedup:>7.2f}x | {saved:>+11.4f}"
            )
        print()

    print("=" * 100)
    if all_speedups:
        print("Warm speedup summary:")
        print(f"  Min:  {min(all_speedups):.2f}x")
        print(f"  Max:  {max(all_speedups):.2f}x")
        print(f"  Mean: {sum(all_speedups) / len(all_speedups):.2f}x")
    print()

    # ── Cold JIT cost table ─────────────────────────────────────────────────
    # Use batch sizes not seen during the warm benchmark to force fresh JIT.
    # The new kernel only compiles once regardless of B; the original compiles
    # per unique NUM_REQS value.
    print("[ Cold JIT cost — first call on a previously-unseen batch size ]")
    print(
        f"{'B':>5} | {'q/req':>6} | "
        f"{'orig cold (ms)':>15} | {'new cold (ms)':>14} | {'JIT saved (ms)':>15}"
    )
    print("-" * 65)

    cold_batch_sizes = [3, 5, 7, 9, 11, 13, 17, 19]  # not used in warm phase

    # Pre-warm the new kernel with a dummy call so its first-compile cost is
    # already paid; subsequent cold calls are truly cache-hit for the new kernel.
    dummy_mask, dummy_topk, dummy_cu, dummy_ttr = _make_inputs(
        1, 64, NUM_TOPK, MAX_SEQ_LEN, device
    )
    _call_new(dummy_mask, dummy_topk, dummy_cu, dummy_ttr, 64)
    torch.accelerator.synchronize()

    for B in cold_batch_sizes:
        q_per_req = 256
        mask, topk_packed, cu_q_lens, token_to_req = _make_inputs(
            B, q_per_req, NUM_TOPK, MAX_SEQ_LEN, device
        )

        _orig_args = (mask, topk_packed, cu_q_lens, B, q_per_req)
        orig_cold = _cold_call_ms(lambda a=_orig_args: _call_orig(*a))
        _new_args = (mask, topk_packed, cu_q_lens, token_to_req, q_per_req)
        new_cold = _cold_call_ms(lambda a=_new_args: _call_new(*a))

        print(
            f"{B:>5} | {q_per_req:>6} | "
            f"{orig_cold:>15.2f} | {new_cold:>14.2f} | "
            f"{orig_cold - new_cold:>+15.2f}"
        )

    print("=" * 100)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark _scatter_topk_kernel: "
            "static tl.static_range(NUM_REQS) vs dynamic token_to_req lookup"
        )
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Benchmark iterations (default: 100)"
    )
    args = parser.parse_args()
    run_benchmark(args.warmup, args.iters)


if __name__ == "__main__":
    main()
