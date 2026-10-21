# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parameter sweep of hisparse_resolve_residency: upstream main vs PR 61263.

Each variant's csrc/hisparse_kernels.cu is compiled into its own small
extension (src/<variant>/), so both kernels load into one process and every
point is timed A/B/A/B on the same GPU. The resolver mutates the LRU and
device_global_indices, so each CUDA-graph replay restores the pristine state
and flushes L2 before the resolve; the resolver kernel's own device time is
read from CUPTI (torch.profiler) per replay.
"""

import argparse
import csv
import itertools
import math
import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
VARIANTS = ["main", "pr", "slot", "smemfix"]
BLOCK = 64
ROW_INT32 = 4  # 16-byte KV rows: the resolver never reads row contents.
LOAD_INVERSE = 4

SCHEMA = (
    "resolve(Tensor host_cache, Tensor! hot_cache, Tensor hot_block_table,"
    " Tensor global_indices, Tensor! hot_indices,"
    " Tensor! device_global_indices, Tensor! lru_slots,"
    " Tensor? request_state_indices, int region_stride, int max_union_rows,"
    " Tensor(a!)? miss_mask=None, Tensor(b!)? stats=None,"
    " Tensor(c!)? attention_indices=None, int attention_block_stride=0,"
    " Tensor? request_ids=None, Tensor? source_block_table=None,"
    " int source_block_size=0, Tensor(d!)? resolved_global_indices=None,"
    " Tensor(e!)? valid_counts=None, Tensor(f!)? swap_host_physical_rows=None,"
    " Tensor(g!)? swap_device_physical_rows=None,"
    " Tensor(h!)? swap_counts=None, Tensor? resident_block_table=None,"
    " int resident_block_size=0, int resident_null_block=0) -> ()"
)

BINDING = r"""
#include "hisparse_kernels.cu"

#include <torch/csrc/stable/library.h>

// out (int32 CUDA): blocks/SM, regs/thread, smem opt-in/block, smem/SM, SMs.
void kernel_info(torch::stable::Tensor& out, int64_t smem_bytes) {
  const int dev = out.get_device_index();
  int info[5] = {0, 0, 0, 0, 0};
  auto kernel = hisparse_resolve_residency_kernel;
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem_bytes));
  }
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info[0], kernel, 1024,
                                                smem_bytes);
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, kernel);
  info[1] = attr.numRegs;
  cudaDeviceGetAttribute(&info[2], cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         dev);
  cudaDeviceGetAttribute(&info[3],
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
  cudaDeviceGetAttribute(&info[4], cudaDevAttrMultiProcessorCount, dev);
  cudaMemcpy(out.mutable_data_ptr<int32_t>(), info, sizeof(info),
             cudaMemcpyHostToDevice);
}

STABLE_TORCH_LIBRARY(NAMESPACE, ops) {
  ops.def("SCHEMA");
  ops.def("kernel_info(Tensor! out, int smem_bytes) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(NAMESPACE, CUDA, ops) {
  ops.impl("resolve", TORCH_BOX(&hisparse_resolve_residency));
  ops.impl("kernel_info", TORCH_BOX(&kernel_info));
}
"""

RENAMED = (
    "hisparse_resolve_residency",
    "hisparse_invalidate_written_slots",
    "hisparse_gather_plan",
    "hisparse_gather_compact",
)


def build(variant: str):
    src = HERE / "src" / variant
    build_dir = HERE / "build" / variant
    build_dir.mkdir(parents=True, exist_ok=True)
    binding = build_dir / f"binding_{variant}.cu"
    binding.write_text(
        BINDING.replace("NAMESPACE", f"hsb_{variant}").replace("SCHEMA", SCHEMA)
    )
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    flags = [
        "-O3",
        "-std=c++17",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        "-DUSE_CUDA",
        *(["-lineinfo"] if os.environ.get("BENCH_LINEINFO") else []),
        *([f"-Xptxas=-O{variant[-1]}"] if variant[-2:-1] == "O" else []),
        *(f"-D{name}={name}_{variant}" for name in RENAMED),
    ]
    load(
        name=f"hsb_{variant}_sm{arch}",
        sources=[str(binding)],
        extra_include_paths=[str(src)],
        extra_cuda_cflags=flags,
        extra_cflags=[
            "-O3",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        ],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=False,
    )
    return getattr(torch.ops, f"hsb_{variant}")


def smem_layout(dbs: int, mu: int, optin: int) -> dict:
    nbc = math.ceil(dbs / 32)
    fixed = 4 * (3 * nbc + 7) + 2 * dbs
    min_smem = fixed + 8 * (mu + 1)
    pr_hash = min(LOAD_INVERSE * mu + 1, (optin - fixed) // 8)
    return {
        "fits": min_smem <= optin,
        "min_smem": min_smem,
        "main": (mu + 1, fixed + 8 * (mu + 1)),
        **{v: (pr_hash, fixed + 8 * pr_hash) for v in VARIANTS[1:]},
    }


class Point:
    """Exact LRU / device_global_indices state for one sweep point."""

    def __init__(self, dbs, top_k, q, reqs, nonres, hit, seed=0):
        self.dbs, self.top_k, self.q, self.reqs = dbs, top_k, q, reqs
        self.mu = q * top_k
        self.n_nonres = round(nonres * top_k)
        self.n_hit = round(hit * self.n_nonres)
        self.union = q * self.n_nonres
        self.misses = q * (self.n_nonres - self.n_hit)
        dev = torch.device("cuda")
        gen = torch.Generator().manual_seed(seed)

        host_tokens = q * self.n_nonres + dbs
        res_tokens = q * (top_k - self.n_nonres)
        nh = math.ceil(host_tokens / BLOCK) + 1
        nr = math.ceil(res_tokens / BLOCK) + 1
        nb = nh + nr
        self.host_rows = (reqs * nb + 1) * BLOCK

        source = torch.arange(reqs * nb, dtype=torch.int32).view(reqs, nb) + 1
        resident = torch.zeros(reqs, nb, dtype=torch.int32)
        resident[:, nh:] = torch.arange(1, nr + 1, dtype=torch.int32)

        rows = torch.empty(reqs * q, top_k, dtype=torch.int32)
        dgi = torch.empty(reqs, dbs, dtype=torch.int32)
        lru = torch.empty(reqs, dbs, dtype=torch.int16)
        for r in range(reqs):
            host_pos = torch.randperm(nh * BLOCK, generator=gen)[:host_tokens]
            res_pos = (
                nh * BLOCK + torch.randperm(nr * BLOCK, generator=gen)[:res_tokens]
            )
            sel_host = host_pos[: q * self.n_nonres].view(q, self.n_nonres)
            others = host_pos[q * self.n_nonres :]
            sel_res = res_pos.view(q, top_k - self.n_nonres)
            for j in range(q):
                row = torch.cat([sel_host[j], sel_res[j]])
                rows[r * q + j] = row[torch.randperm(top_k, generator=gen)]
            hits = sel_host[:, : self.n_hit].reshape(-1)
            cached = torch.cat([hits, others[: dbs - hits.numel()]])
            cached_g = source[r, cached // BLOCK] * BLOCK + cached % BLOCK
            dgi[r, torch.randperm(dbs, generator=gen)] = cached_g.int()
            lru[r] = torch.randperm(dbs, generator=gen).to(torch.int16)

        self.global_indices = rows.to(dev)
        self.dgi0 = dgi.to(dev)
        self.lru0 = lru.to(dev)
        self.dgi = self.dgi0.clone()
        self.lru = self.lru0.clone()
        self.source = source.to(dev)
        self.resident = resident.to(dev)
        self.request_ids = (
            torch.arange(reqs, dtype=torch.int32).repeat_interleave(q).to(dev)
        )
        self.state = torch.arange(reqs, dtype=torch.int32, device=dev)
        hot_blocks_per_req = dbs // BLOCK
        self.hot_cache = torch.empty(
            reqs * hot_blocks_per_req,
            BLOCK,
            ROW_INT32,
            dtype=torch.int32,
            device=dev,
        )
        self.hot_block_table = (
            torch.arange(reqs * hot_blocks_per_req, dtype=torch.int32)
            .view(reqs, hot_blocks_per_req)
            .to(dev)
        )
        n = reqs * q
        self.outputs = {
            name: torch.empty(n, top_k, dtype=torch.int32, device=dev)
            for name in ("hot", "attn", "swap_host", "swap_dev")
        }
        self.outputs["valid"] = torch.empty(n, dtype=torch.int32, device=dev)
        self.outputs["swap_counts"] = torch.empty(n, dtype=torch.int32, device=dev)
        self.stats = torch.zeros(2, dtype=torch.uint64, device=dev)

    def call(self, ops, host_cache):
        o = self.outputs
        ops.resolve(
            host_cache,
            self.hot_cache,
            self.hot_block_table,
            self.global_indices,
            o["hot"],
            self.dgi,
            self.lru,
            self.state,
            self.dbs,
            self.mu,
            None,
            self.stats,
            o["attn"],
            BLOCK,
            self.request_ids,
            self.source,
            BLOCK,
            None,
            o["valid"],
            o["swap_host"],
            o["swap_dev"],
            o["swap_counts"],
            self.resident,
            BLOCK,
            0,
        )

    def restore(self):
        self.dgi.copy_(self.dgi0)
        self.lru.copy_(self.lru0)
        self.stats.zero_()

    def snapshot(self):
        out = {k: v.clone() for k, v in self.outputs.items()}
        out["dgi"] = self.dgi.clone()
        out["lru"] = self.lru.clone()
        return out


def kernel_times(prof, name="hisparse_resolve_residency_kernel"):
    events = [
        e
        for e in prof.events()
        if name in e.name and e.device_type == torch.autograd.DeviceType.CUDA
    ]
    events.sort(key=lambda e: e.time_range.start)
    return [e.time_range.elapsed_us() for e in events]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def run_point(point, ops, host_cache, flush, reps, blocks):
    for v in VARIANTS:
        point.restore()
        point.call(ops[v], host_cache)
    torch.cuda.synchronize()

    snaps = {}
    for v in VARIANTS:
        point.restore()
        point.call(ops[v], host_cache)
        snaps[v] = point.snapshot()
    torch.cuda.synchronize()
    for v in VARIANTS[1:]:
        for k in snaps["main"]:
            if not torch.equal(snaps["main"][k], snaps[v][k]):
                raise AssertionError(f"main/{v} mismatch in {k}")
    swaps = int(snaps["main"]["swap_counts"].sum())
    expected = point.reqs * point.misses
    if swaps != expected:
        raise AssertionError(f"swap count {swaps} != expected {expected}")

    host_us = {}
    for v in VARIANTS:
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            point.call(ops[v], host_cache)
            ts.append((time.perf_counter() - t0) * 1e6)
        torch.cuda.synchronize()
        host_us[v] = pct(ts, 50)

    graphs = {}
    for v in VARIANTS:
        g = torch.cuda.CUDAGraph()
        point.restore()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            point.restore()
            flush.zero_()
            point.call(ops[v], host_cache)
        graphs[v] = g
    for v in VARIANTS:
        for _ in range(5):
            graphs[v].replay()
    torch.cuda.synchronize()

    order = [v for _ in range(blocks) for v in VARIANTS]
    for _attempt in range(5):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for v in order:
                for _ in range(reps):
                    graphs[v].replay()
            torch.cuda.synchronize()
        times = kernel_times(prof)
        if len(times) == len(order) * reps:
            break
        print(f"CUPTI returned {len(times)} resolver kernels, retrying", flush=True)
    else:
        times = None
    per = {v: [] for v in VARIANTS}
    if times is not None:
        for i, v in enumerate(order):
            per[v].extend(times[i * reps : (i + 1) * reps])
    else:
        # CUPTI can drop a record on very long sessions; profile per block.
        for v in order:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(reps):
                    graphs[v].replay()
                torch.cuda.synchronize()
            block = kernel_times(prof)
            if len(block) < reps - 1:
                raise RuntimeError("CUPTI kept dropping resolver kernel records")
            per[v].extend(block)
    del graphs
    return per, host_us


def key(row):
    return tuple(
        str(row[k])
        for k in (
            "device_buffer_size",
            "top_k",
            "q_rows",
            "requests",
            "nonresident_pct",
            "hit_pct",
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", default="main,pr,slot,smemfix")
    ap.add_argument("--dbs", default="2048,8192,32768")
    ap.add_argument("--top-k", default="512,2048")
    ap.add_argument("--q", default="1,2,4")
    ap.add_argument("--reqs", default="1,8,32,128")
    ap.add_argument("--nonres", default="0,0.1,0.5,1.0")
    ap.add_argument("--hit", default="0,0.5,0.9")
    args = ap.parse_args()
    VARIANTS[:] = args.variants.split(",")
    assert VARIANTS[0] == "main"

    torch.cuda.init()
    gpu = torch.cuda.get_device_name()
    ops = {v: build(v) for v in VARIANTS}
    info = torch.empty(5, dtype=torch.int32, device="cuda")
    ops["main"].kernel_info(info, 0)
    _, regs, optin, smem_per_sm, sms = info.tolist()
    print(
        f"{gpu}: optin={optin} smem/SM={smem_per_sm} SMs={sms} regs={regs}", flush=True
    )

    ints = lambda x: tuple(int(v) for v in x.split(","))
    floats = lambda x: tuple(float(v) for v in x.split(","))
    dbs_s, topk_s, q_s, reqs_s = (
        ints(args.dbs),
        ints(args.top_k),
        ints(args.q),
        ints(args.reqs),
    )
    nonres_s, hit_s = floats(args.nonres), floats(args.hit)
    if args.quick:
        dbs_s, topk_s, q_s, reqs_s = (8192,), (2048,), (1, 4), (8, 128)
        nonres_s, hit_s = (0.0, 1.0), (0.0, 0.9)

    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    flush = torch.empty(max(2 * l2, 64 << 20), dtype=torch.uint8, device="cuda")
    max_host_rows = (
        128 * (math.ceil((4 * 2048 + 32768) / BLOCK) + math.ceil(4 * 2048 / BLOCK) + 2)
        + 1
    ) * BLOCK
    host_cache = torch.empty(
        max_host_rows, ROW_INT32, dtype=torch.int32, pin_memory=True
    )

    fields = [
        "gpu",
        "device_buffer_size",
        "top_k",
        "q_rows",
        "requests",
        "nonresident_pct",
        "hit_pct",
        "max_union_rows",
        "union_per_req",
        "misses_per_req",
        "status",
        "regs",
        "sms",
        *(
            f"{v}_{c}"
            for v in VARIANTS
            for c in (
                "hash_size",
                "table_size",
                "fill",
                "smem",
                "blocks_per_sm",
                "med_us",
                "p90_us",
                "host_us",
            )
        ),
        *(f"{v}_speedup_{s}" for v in VARIANTS[1:] for s in ("med", "p90")),
    ]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {key(r) for r in csv.DictReader(f)}
    out = open(args.out, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    if not done:
        writer.writeheader()

    for dbs, top_k, q, reqs, nonres in itertools.product(
        dbs_s, topk_s, q_s, reqs_s, nonres_s
    ):
        for hit in hit_s if nonres > 0 else hit_s[:1]:
            mu = q * top_k
            row = {
                "gpu": gpu,
                "device_buffer_size": dbs,
                "top_k": top_k,
                "q_rows": q,
                "requests": reqs,
                "nonresident_pct": int(nonres * 100),
                "hit_pct": int(hit * 100) if nonres > 0 else "",
                "max_union_rows": mu,
                "regs": regs,
                "sms": sms,
            }
            if key(row) in done:
                continue
            layout = smem_layout(dbs, mu, optin)
            if mu > dbs:
                row["status"] = "pruned: device_buffer_size < q*top_k"
            elif not layout["fits"]:
                row["status"] = f"pruned: needs {layout['min_smem']} B smem > {optin}"
            if "status" in row:
                writer.writerow(row)
                continue
            point = Point(dbs, top_k, q, reqs, nonres, hit)
            row["union_per_req"] = point.union
            row["misses_per_req"] = point.misses
            for v in VARIANTS:
                hash_size, smem = layout[v]
                table = min(hash_size, LOAD_INVERSE * q * top_k)
                ops[v].kernel_info(info, smem)
                row[f"{v}_hash_size"] = hash_size
                row[f"{v}_table_size"] = table
                row[f"{v}_fill"] = round(point.union / table, 4)
                row[f"{v}_smem"] = smem
                row[f"{v}_blocks_per_sm"] = info[0].item()
            per, host_us = run_point(
                point, ops, host_cache, flush, args.reps, args.blocks
            )
            for v in VARIANTS:
                row[f"{v}_med_us"] = round(pct(per[v], 50), 2)
                row[f"{v}_p90_us"] = round(pct(per[v], 90), 2)
                row[f"{v}_host_us"] = round(host_us[v], 2)
            for v in VARIANTS[1:]:
                for s in ("med", "p90"):
                    row[f"{v}_speedup_{s}"] = round(
                        row[f"main_{s}_us"] / row[f"{v}_{s}_us"], 3
                    )
            row["status"] = "ok"
            writer.writerow(row)
            out.flush()
            print(
                f"dbs={dbs} k={top_k} q={q} R={reqs} nonres={nonres} "
                f"hit={hit}: "
                + " ".join(f"{v} {row[f'{v}_med_us']}" for v in VARIANTS)
                + " | "
                + " ".join(f"{v} x{row[f'{v}_speedup_med']}" for v in VARIANTS[1:]),
                flush=True,
            )
            del point
            torch.cuda.empty_cache()
    out.close()


if __name__ == "__main__":
    main()
