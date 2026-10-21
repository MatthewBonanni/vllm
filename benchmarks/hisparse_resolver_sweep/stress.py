# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Replay one sweep point many times per variant and count output mismatches vs main."""

import argparse

import bench
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="fix,smemfix,both")
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--point", default="8192,512,4,128,1.0,0.5")
ap.add_argument("--diag", action="store_true")
ap.add_argument("--only", type=int, default=-1)
args = ap.parse_args()
variants = ["main"] + args.variants.split(",")
bench.VARIANTS[:] = variants
dbs, top_k, q, reqs, nonres, hit = args.point.split(",")
ops = {v: bench.build(v) for v in variants}
point = bench.Point(int(dbs), int(top_k), int(q), int(reqs), float(nonres), float(hit))
if args.only >= 0:
    keep = point.state[args.only].item()
    point.state.fill_(-1)
    point.state[args.only] = keep
host_cache = torch.empty(
    point.host_rows, bench.ROW_INT32, dtype=torch.int32, pin_memory=True
)
for t in point.outputs.values():
    t.fill_(-7)
point.restore()
point.call(ops["main"], host_cache)
ref = point.snapshot()
for v in variants[1:]:
    bad = {}
    for _ in range(args.iters):
        for t in point.outputs.values():
            t.fill_(-7)
        point.restore()
        point.call(ops[v], host_cache)
        snap = point.snapshot()
        for k in ref:
            if not torch.equal(ref[k], snap[k]):
                bad[k] = bad.get(k, 0) + 1
        if args.diag and not torch.equal(ref["swap_counts"], snap["swap_counts"]):
            rc = ref["swap_counts"].view(point.reqs, point.q).sum(1)
            sc = snap["swap_counts"].view(point.reqs, point.q).sum(1)
            d = (sc - rc).nonzero().flatten().tolist()
            print(
                f"{v}: {len(d)} requests differ, e.g. req {d[:8]} "
                f"misses ref {rc[d[:8]].tolist()} got {sc[d[:8]].tolist()}",
                flush=True,
            )
            args.diag = False
    print(v, "mismatching iters per tensor:", bad or "none", flush=True)
