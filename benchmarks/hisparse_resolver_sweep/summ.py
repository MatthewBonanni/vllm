import csv, math, statistics as st, sys
V = sys.argv[1]
F = dict(a.split("=") for a in sys.argv[2:])
gm = lambda xs: math.exp(st.fmean(math.log(x) for x in xs))
rows = {g: [r for r in csv.DictReader(open(f)) if r["status"] == "ok"] for g, f in F.items()}
print("| Selected pages not on GPU | GPU | PR vs main | this branch vs main | slower than main (PR / this) |")
print("|---|---|---|---|---|")
for n in ("0", "10", "50", "100"):
    for g in F:
        s = [r for r in rows[g] if r["nonresident_pct"] == n]
        p = [float(r["pr_speedup_med"]) for r in s]
        v = [float(r[f"{V}_speedup_med"]) for r in s]
        fmt = lambda xs: f"{gm(xs):.3f}× ({min(xs):.2f}–{max(xs):.2f})" if gm(xs) < 3 else f"{gm(xs):.1f}× ({min(xs):.1f}–{max(xs):.1f})"
        print(f"| {n}% | {g} | {fmt(p)} | {fmt(v)} | {sum(x < 1 for x in p)} / {sum(x < 1 for x in v)} of {len(s)} |")
