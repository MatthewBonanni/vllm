# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Three-way table for the lazy-table fix: main vs PR vs fix."""

import csv
import math
import statistics
import sys

GPUS = ("A100", "H200", "B200")


def short(gpu):
    return next((g for g in GPUS if g in gpu), gpu)


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


rows = []
for p in sys.argv[1:]:
    rows += [r for r in csv.DictReader(open(p)) if r["status"] == "ok"]
for r in rows:
    r["gpu"] = short(r["gpu"])
gpus = [g for g in GPUS if any(r["gpu"] == g for r in rows)]

print(
    "| non-res | GPU | PR vs main | fix vs main | fix vs PR | PR slower than main | fix slower than main |"
)
print("|---|---|---|---|---|---|---|")
for n in ("0", "10", "50", "100"):
    for g in gpus:
        s = [r for r in rows if r["gpu"] == g and r["nonresident_pct"] == n]
        pr = [float(r["speedup_med"]) for r in s]
        fx = [float(r["fix_speedup_med"]) for r in s]
        fp = [float(r["fix_vs_pr_med"]) for r in s]
        f = lambda xs: f"{gm(xs):.3f}× ({min(xs):.3f}–{max(xs):.3f})"
        print(
            f"| {n}% | {g} | {f(pr)} | {f(fx)} | {f(fp)} | "
            f"{sum(x < 1 for x in pr)}/{len(s)} | {sum(x < 1 for x in fx)}/{len(s)} |"
        )
