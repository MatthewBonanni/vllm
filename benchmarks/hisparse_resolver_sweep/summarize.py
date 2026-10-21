# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Merge per-GPU sweep CSVs into results.csv and write SUMMARY tables."""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_ORDER = ("H200", "B200", "A100")


def short(gpu):
    return next((g for g in GPU_ORDER if g in gpu), gpu)


def geomean(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        r["gpu"] = short(r["gpu"])
    rows.sort(key=lambda r: GPU_ORDER.index(r["gpu"]))
    return rows


def f(x, nd=2):
    return f"{float(x):.{nd}f}"


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(out)


def main():
    paths = sys.argv[1:] or sorted((HERE / "results").glob("*.csv"))
    rows = load(paths)
    fields = list(rows[0].keys())
    with open(HERE / "results.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    pruned = [r for r in rows if r["status"] != "ok"]
    gpus = [g for g in GPU_ORDER if any(r["gpu"] == g for r in ok)]
    md = []

    md.append("### Overall (geomean of main/PR median kernel time)\n")
    body = []
    for g in gpus:
        cells = [g]
        for nr in ("0", "10", "50", "100"):
            sp = [
                float(r["speedup_med"])
                for r in ok
                if r["gpu"] == g and r["nonresident_pct"] == nr
            ]
            cells.append(f"{geomean(sp):.3f} ({min(sp):.2f}–{max(sp):.2f})")
        body.append(cells)
    md.append(table(["GPU", "0% non-res", "10%", "50%", "100%"], body))

    md.append("\n### Shared memory: hash size and occupancy\n")
    seen = {}
    for r in ok:
        k = (r["gpu"], r["device_buffer_size"], r["top_k"], r["q_rows"])
        seen.setdefault(k, r)
    body = []
    for (g, dbs, k, q), r in seen.items():
        capped = int(r["pr_hash_size"]) < 4 * int(r["max_union_rows"]) + 1
        body.append(
            [
                g,
                dbs,
                k,
                q,
                r["main_hash_size"],
                r["pr_hash_size"] + (" (smem-capped)" if capped else ""),
                f"{int(r['main_smem']) // 1024}K → {int(r['pr_smem']) // 1024}K",
                f"{r['main_blocks_per_sm']} → {r['pr_blocks_per_sm']}",
            ]
        )
    md.append(
        table(
            [
                "GPU",
                "dbs",
                "top_k",
                "q",
                "main hash",
                "PR hash",
                "smem/block",
                "blocks/SM",
            ],
            body,
        )
    )

    md.append("\n### All-resident rows (0% non-resident)\n")
    body = []
    for g in gpus:
        sub = [r for r in ok if r["gpu"] == g and r["nonresident_pct"] == "0"]
        for dbs in ("2048", "8192", "32768"):
            s = [r for r in sub if r["device_buffer_size"] == dbs]
            if not s:
                continue
            sp = [float(r["speedup_med"]) for r in s]
            worst = min(s, key=lambda r: float(r["speedup_med"]))
            dt = [float(r["pr_med_us"]) - float(r["main_med_us"]) for r in s]
            body.append(
                [
                    g,
                    dbs,
                    len(s),
                    f"{geomean(sp):.3f}",
                    f"{min(sp):.3f}",
                    f"k={worst['top_k']} q={worst['q_rows']} R={worst['requests']}: "
                    f"{f(worst['main_med_us'])} → {f(worst['pr_med_us'])} µs",
                    f"{max(dt):+.2f}",
                ]
            )
    md.append(
        table(
            ["GPU", "dbs", "points", "geomean", "worst", "worst point", "max Δ µs"],
            body,
        )
    )

    md.append(
        "\n### device_buffer_size=8192, R=128 requests (median µs main → PR, speedup)\n"
    )
    for g in gpus:
        md.append(f"\n**{g}**\n")
        body = []
        for k in ("512", "2048"):
            for q in ("1", "2", "4"):
                cells = [k, q]
                for nr, hit in (
                    ("0", ""),
                    ("10", "0"),
                    ("10", "90"),
                    ("50", "0"),
                    ("50", "50"),
                    ("50", "90"),
                    ("100", "0"),
                    ("100", "50"),
                    ("100", "90"),
                ):
                    m = [
                        r
                        for r in ok
                        if r["gpu"] == g
                        and r["device_buffer_size"] == "8192"
                        and r["top_k"] == k
                        and r["q_rows"] == q
                        and r["requests"] == "128"
                        and r["nonresident_pct"] == nr
                        and r["hit_pct"] == hit
                    ]
                    cells.append(
                        f"{f(m[0]['main_med_us'], 1)}→{f(m[0]['pr_med_us'], 1)} "
                        f"**{f(m[0]['speedup_med'])}×**"
                        if m
                        else "–"
                    )
                body.append(cells)
        md.append(
            table(
                [
                    "top_k",
                    "q",
                    "0%",
                    "10%/h0",
                    "10%/h90",
                    "50%/h0",
                    "50%/h50",
                    "50%/h90",
                    "100%/h0",
                    "100%/h50",
                    "100%/h90",
                ],
                body,
            )
        )

    md.append(
        "\n### Host-side eager call cost (median µs, includes "
        "cudaFuncSetAttribute when smem > 48 KB)\n"
    )
    body = []
    for g in gpus:
        sub = [r for r in ok if r["gpu"] == g]
        d = [float(r["pr_host_us"]) - float(r["main_host_us"]) for r in sub]
        both_big = [
            float(r["pr_host_us"]) - float(r["main_host_us"])
            for r in sub
            if int(r["main_smem"]) <= 48 * 1024 < int(r["pr_smem"])
        ]
        body.append(
            [
                g,
                f"{statistics.median(d):+.2f}",
                f"{statistics.median(both_big):+.2f}" if both_big else "–",
            ]
        )
    md.append(table(["GPU", "median Δ (all)", "median Δ (main ≤48K, PR >48K)"], body))

    md.append("\n### Pruned\n")
    reasons = defaultdict(set)
    for r in pruned:
        reasons[(r["gpu"], r["status"])].add(
            f"dbs={r['device_buffer_size']} k={r['top_k']} q={r['q_rows']}"
        )
    md.append(
        "\n".join(
            f"- {g}: {s} — {', '.join(sorted(v))}" for (g, s), v in reasons.items()
        )
    )

    (HERE / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
