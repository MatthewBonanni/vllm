"""Launch one variant's resolver a few times at one point, for ncu."""

import argparse

import torch

import bench

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--point", default="8192,512,4,1,0.0,0.0")
ap.add_argument("--calls", type=int, default=3)
args = ap.parse_args()
bench.VARIANTS[:] = ["main"] if args.variant == "main" else ["main", args.variant]
ops = bench.build(args.variant)
dbs, top_k, q, reqs, nonres, hit = args.point.split(",")
point = bench.Point(int(dbs), int(top_k), int(q), int(reqs), float(nonres), float(hit))
host_cache = torch.empty(point.host_rows, bench.ROW_INT32, dtype=torch.int32, pin_memory=True)
for _ in range(args.calls):
    point.restore()
    point.call(ops, host_cache)
torch.cuda.synchronize()
