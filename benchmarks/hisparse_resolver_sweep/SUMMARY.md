# hisparse_resolve_residency sweep: upstream main vs PR 61263

- **main**: upstream/main `3a1bd8347a`. Its `csrc/hisparse_kernels.cu` is byte-identical to the PR's merge base `8055293353`.
- **PR**: `b605dfa0` (vllm-project/vllm#61263).
- **GPUs**: H200 (ibm-h200-1), B200 (pirate2 dell-b200-01), A100-SXM4-40GB (rdu4-a100-0). No L4 was free. One GPU each, torch 2.13.
- **Points**: 640 timed per GPU, 1920 total, plus 80 pruned per GPU. Full data is in `results.csv`; raw per-GPU CSVs are in `results/`.

## Findings

1. **Heavy host paging (≥50% of selected pages not on the GPU): the PR wins by a lot.** At 100% it is 10–125× faster (geomean about 33×). At 50% it is 1.07–1.9× faster (geomean about 1.25×). On main the union fills the table to `union/(max_union_rows+1)`. At 100% that fill is about 1.0, so every missed lookup in phase 2 walks almost the whole table.
2. **All-resident (0%) steps: the PR is 1–4% slower almost everywhere** (geomean 0.97–0.99, under 2 µs absolute). The worst case is B200 dbs=32768 k=512 q=4 R=128: 13.6 → 15.3 µs (0.89×), where occupancy drops from 2 to 1 blocks/SM. There is one B200 outlier the other way: dbs=8192 k=2048 q=2 R=128 is 1.18× faster.
3. **New regression at low paging with a large buffer: dbs=32768 at 10% non-resident is 15–25% slower on all three GPUs.**
   - H200 k=2048 q=2: 82 → 105 µs.
   - B200 k=512 q=2 R=128: 100 → 132 µs.
   - A100 k=2048 q=1: 0.83×.
   - Smaller versions appear at dbs=8192 k=2048 q=4 (0.94–0.97×).
   - The table is nearly empty here (PR fill 0.025), so shorter probe chains save little. Meanwhile smem per block grows to 140–226 KB. *Untested hypothesis:* the extra smem squeezes out the L1 carve-out that the random `row_dgi[slot]` gather over 32K slots relies on, and the up to 4× larger table init adds a bit more. An `ncu` L1-hit-rate comparison would confirm or rule this out.
4. **Hash size depends on the device's smem budget.** The table hits the smem cap only at the largest unions:
   - H200 and B200: (8192, 2048, q=4) → 26620 entries; (32768, 2048, q=4) → 19324 entries.
   - A100: (8192, 2048, q=4) → 18428 entries; (32768, 2048, q≥2) → 11132 entries.
   Even capped, the table stays well above main's `mu+1`. On SM89/SM120-class parts (~99 KB) the cap would bind much sooner. That case wasn't measured.
5. **Blocks per SM** are register-limited (1024 threads). That gives 1 per SM on H200 and A100 for both variants. On B200 the PR drops from 2 to 1 for the larger configs (see `tables.md`).
6. **Host-side cost of the per-launch `cudaFuncSetAttribute`**: when the PR crosses 48 KB and main doesn't, an eager call costs +0.5–0.9 µs. Under CUDA graphs this cost disappears.

## device_buffer_size=8192 (heavy paging focus)

R=128 tables per GPU are below. With large top_k (2048) and heavy paging (50–100% non-resident), the PR is 1.07–1.7× faster at 50% and 45–125× faster at 100%. At 10% it is break-even, from 0.94× (q=4) to 1.43× (B200 q=2). At 0% it is 0.95–0.98×, except B200 q≥2 at 1.16–1.18×.

## Method and caveats

- **Build**: each variant's `hisparse_kernels.cu` is compiled into a small standalone stable-ABI extension (`bench.py: build`; the sources are vendored in `src/<variant>`, refs in `src/REFS`). The rest of vLLM is not needed, and both kernels load into one process. Each point checks that main and PR produce byte-identical outputs and identical LRU/`device_global_indices` state, and that the swap count equals the expected misses.
- **State**: built directly per point.
    - Each request's q rows select **disjoint** pages: `nonres%·top_k` per row from host-backed blocks, the rest from `resident_block_table` blocks.
    - Of the host-backed pages, `hit%` sit in `device_global_indices`. The remaining hot slots hold other, unselected host pages, so the buffer is full and phase 2 performs `dbs − hits` failed lookups. The LRU order is a random permutation.
    - Because rows are disjoint, the union equals `nonres% × max_union_rows`. **The 100% rows are therefore the worst-case union**, where main's table is full. Real MTP rows overlap, so real unions on main fill the table less.
- **Timing**: one CUDA graph per variant, each replay doing restore state → L2 flush → resolve. The resolver kernel's device time per replay is read from CUPTI (torch.profiler). Replays run in A/B/A/B order, 200 per variant. The tables report the median; `results.csv` also has p90 and the eager host-call median.
- **Pruned**: `dbs=2048, top_k=2048, q∈{2,4}`, because device_buffer_size < q·top_k is rejected by config. Nothing was pruned for shared memory on these GPUs.

## Reproduce

```
python bench.py --out <gpu>.csv   # resumable; --quick for a smoke run
python summarize.py results/*.csv # writes results.csv and tables.md
```

## Tables

### Overall (geomean of main/PR median kernel time)

| GPU | 0% non-res | 10% | 50% | 100% |
|---|---|---|---|---|
| H200 | 0.970 (0.95–0.99) | 0.991 (0.78–1.06) | 1.295 (1.00–1.84) | 36.134 (10.84–118.05) |
| B200 | 0.985 (0.89–1.22) | 1.006 (0.75–1.46) | 1.258 (0.97–1.91) | 33.376 (9.68–124.84) |
| A100 | 0.978 (0.95–1.00) | 0.982 (0.83–1.05) | 1.241 (1.07–1.49) | 33.024 (11.25–86.06) |

### Shared memory: hash size and occupancy

| GPU | dbs | top_k | q | main hash | PR hash | smem/block | blocks/SM |
|---|---|---|---|---|---|---|---|
| H200 | 2048 | 512 | 1 | 513 | 2049 | 8K → 20K | 1 → 1 |
| H200 | 2048 | 512 | 2 | 1025 | 4097 | 12K → 36K | 1 → 1 |
| H200 | 2048 | 512 | 4 | 2049 | 8193 | 20K → 68K | 1 → 1 |
| H200 | 2048 | 2048 | 1 | 2049 | 8193 | 20K → 68K | 1 → 1 |
| H200 | 8192 | 512 | 1 | 513 | 2049 | 23K → 35K | 1 → 1 |
| H200 | 8192 | 512 | 2 | 1025 | 4097 | 27K → 51K | 1 → 1 |
| H200 | 8192 | 512 | 4 | 2049 | 8193 | 35K → 83K | 1 → 1 |
| H200 | 8192 | 2048 | 1 | 2049 | 8193 | 35K → 83K | 1 → 1 |
| H200 | 8192 | 2048 | 2 | 4097 | 16385 | 51K → 147K | 1 → 1 |
| H200 | 8192 | 2048 | 4 | 8193 | 26620 (smem-capped) | 83K → 226K | 1 → 1 |
| H200 | 32768 | 512 | 1 | 513 | 2049 | 80K → 92K | 1 → 1 |
| H200 | 32768 | 512 | 2 | 1025 | 4097 | 84K → 108K | 1 → 1 |
| H200 | 32768 | 512 | 4 | 2049 | 8193 | 92K → 140K | 1 → 1 |
| H200 | 32768 | 2048 | 1 | 2049 | 8193 | 92K → 140K | 1 → 1 |
| H200 | 32768 | 2048 | 2 | 4097 | 16385 | 108K → 204K | 1 → 1 |
| H200 | 32768 | 2048 | 4 | 8193 | 19324 (smem-capped) | 140K → 226K | 1 → 1 |
| B200 | 2048 | 512 | 1 | 513 | 2049 | 8K → 20K | 2 → 2 |
| B200 | 2048 | 512 | 2 | 1025 | 4097 | 12K → 36K | 2 → 2 |
| B200 | 2048 | 512 | 4 | 2049 | 8193 | 20K → 68K | 2 → 2 |
| B200 | 2048 | 2048 | 1 | 2049 | 8193 | 20K → 68K | 2 → 2 |
| B200 | 8192 | 512 | 1 | 513 | 2049 | 23K → 35K | 2 → 2 |
| B200 | 8192 | 512 | 2 | 1025 | 4097 | 27K → 51K | 2 → 2 |
| B200 | 8192 | 512 | 4 | 2049 | 8193 | 35K → 83K | 2 → 2 |
| B200 | 8192 | 2048 | 1 | 2049 | 8193 | 35K → 83K | 2 → 2 |
| B200 | 8192 | 2048 | 2 | 4097 | 16385 | 51K → 147K | 2 → 1 |
| B200 | 8192 | 2048 | 4 | 8193 | 26620 (smem-capped) | 83K → 226K | 2 → 1 |
| B200 | 32768 | 512 | 1 | 513 | 2049 | 80K → 92K | 2 → 2 |
| B200 | 32768 | 512 | 2 | 1025 | 4097 | 84K → 108K | 2 → 2 |
| B200 | 32768 | 512 | 4 | 2049 | 8193 | 92K → 140K | 2 → 1 |
| B200 | 32768 | 2048 | 1 | 2049 | 8193 | 92K → 140K | 2 → 1 |
| B200 | 32768 | 2048 | 2 | 4097 | 16385 | 108K → 204K | 2 → 1 |
| B200 | 32768 | 2048 | 4 | 8193 | 19324 (smem-capped) | 140K → 226K | 1 → 1 |
| A100 | 2048 | 512 | 1 | 513 | 2049 | 8K → 20K | 1 → 1 |
| A100 | 2048 | 512 | 2 | 1025 | 4097 | 12K → 36K | 1 → 1 |
| A100 | 2048 | 512 | 4 | 2049 | 8193 | 20K → 68K | 1 → 1 |
| A100 | 2048 | 2048 | 1 | 2049 | 8193 | 20K → 68K | 1 → 1 |
| A100 | 8192 | 512 | 1 | 513 | 2049 | 23K → 35K | 1 → 1 |
| A100 | 8192 | 512 | 2 | 1025 | 4097 | 27K → 51K | 1 → 1 |
| A100 | 8192 | 512 | 4 | 2049 | 8193 | 35K → 83K | 1 → 1 |
| A100 | 8192 | 2048 | 1 | 2049 | 8193 | 35K → 83K | 1 → 1 |
| A100 | 8192 | 2048 | 2 | 4097 | 16385 | 51K → 147K | 1 → 1 |
| A100 | 8192 | 2048 | 4 | 8193 | 18428 (smem-capped) | 83K → 162K | 1 → 1 |
| A100 | 32768 | 512 | 1 | 513 | 2049 | 80K → 92K | 1 → 1 |
| A100 | 32768 | 512 | 2 | 1025 | 4097 | 84K → 108K | 1 → 1 |
| A100 | 32768 | 512 | 4 | 2049 | 8193 | 92K → 140K | 1 → 1 |
| A100 | 32768 | 2048 | 1 | 2049 | 8193 | 92K → 140K | 1 → 1 |
| A100 | 32768 | 2048 | 2 | 4097 | 11132 (smem-capped) | 108K → 162K | 1 → 1 |
| A100 | 32768 | 2048 | 4 | 8193 | 11132 (smem-capped) | 140K → 162K | 1 → 1 |

### All-resident rows (0% non-resident)

| GPU | dbs | points | geomean | worst | worst point | max Δ µs |
|---|---|---|---|---|---|---|
| H200 | 2048 | 16 | 0.972 | 0.954 | k=512 q=2 R=1: 5.95 → 6.24 µs | +0.31 |
| H200 | 8192 | 24 | 0.967 | 0.960 | k=512 q=4 R=1: 9.22 → 9.60 µs | +1.02 |
| H200 | 32768 | 24 | 0.973 | 0.957 | k=2048 q=2 R=1: 10.62 → 11.10 µs | +0.93 |
| B200 | 2048 | 16 | 0.986 | 0.956 | k=2048 q=1 R=32: 7.65 → 8.00 µs | +0.48 |
| B200 | 8192 | 24 | 0.991 | 0.952 | k=2048 q=1 R=32: 7.74 → 8.13 µs | +0.84 |
| B200 | 32768 | 24 | 0.979 | 0.889 | k=512 q=4 R=128: 13.63 → 15.33 µs | +1.79 |
| A100 | 2048 | 16 | 0.980 | 0.962 | k=2048 q=1 R=1: 8.13 → 8.45 µs | +0.55 |
| A100 | 8192 | 24 | 0.974 | 0.954 | k=2048 q=2 R=8: 13.89 → 14.56 µs | +1.37 |
| A100 | 32768 | 24 | 0.981 | 0.966 | k=2048 q=1 R=128: 16.64 → 17.22 µs | +0.73 |

### device_buffer_size=8192, R=128 requests (median µs main → PR, speedup)

**H200**

| top_k | q | 0% | 10%/h0 | 10%/h90 | 50%/h0 | 50%/h50 | 50%/h90 | 100%/h0 | 100%/h50 | 100%/h90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1 | 5.2→5.3 **0.98×** | 22.1→21.4 **1.03×** | 22.3→21.3 **1.04×** | 33.0→23.6 **1.40×** | 33.1→23.5 **1.41×** | 33.1→23.4 **1.41×** | 314.5→25.1 **12.55×** | 314.4→24.9 **12.61×** | 314.1→25.0 **12.57×** |
| 512 | 2 | 8.7→8.9 **0.98×** | 27.8→26.2 **1.06×** | 26.9→25.5 **1.06×** | 41.6→27.5 **1.51×** | 41.7→27.6 **1.51×** | 41.6→27.6 **1.51×** | 616.8→27.6 **22.39×** | 614.5→27.2 **22.59×** | 613.7→27.1 **22.67×** |
| 512 | 4 | 15.7→16.0 **0.98×** | 38.4→37.4 **1.02×** | 38.1→37.1 **1.03×** | 55.6→39.0 **1.42×** | 55.4→39.3 **1.41×** | 55.5→39.1 **1.42×** | 1224.6→37.9 **32.32×** | 1217.9→37.1 **32.81×** | 1213.0→37.0 **32.82×** |
| 2048 | 1 | 7.6→7.9 **0.96×** | 28.6→27.4 **1.05×** | 28.3→27.0 **1.05×** | 42.2→29.6 **1.43×** | 42.2→29.3 **1.44×** | 41.9→29.1 **1.44×** | 1472.1→30.1 **48.94×** | 1466.2→29.4 **49.85×** | 1463.4→29.1 **50.25×** |
| 2048 | 2 | 13.4→13.8 **0.97×** | 44.5→43.0 **1.03×** | 44.5→42.9 **1.03×** | 59.8→44.2 **1.35×** | 60.2→44.3 **1.36×** | 59.2→43.9 **1.35×** | 2979.7→41.4 **71.90×** | 2941.2→39.8 **73.94×** | 2933.5→39.4 **74.47×** |
| 2048 | 4 | 25.1→26.1 **0.96×** | 74.8→78.0 **0.96×** | 73.2→75.8 **0.96×** | 91.2→85.5 **1.07×** | 89.8→84.5 **1.06×** | 89.7→82.8 **1.08×** | 5950.6→71.4 **83.39×** | 5839.2→68.3 **85.43×** | 5556.5→66.0 **84.22×** |

**B200**

| top_k | q | 0% | 10%/h0 | 10%/h90 | 50%/h0 | 50%/h50 | 50%/h90 | 100%/h0 | 100%/h50 | 100%/h90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1 | 6.1→6.2 **0.99×** | 23.7→22.9 **1.03×** | 23.8→23.1 **1.03×** | 34.6→24.5 **1.41×** | 34.9→24.8 **1.41×** | 35.0→24.6 **1.42×** | 302.8→26.1 **11.61×** | 302.9→26.1 **11.59×** | 302.5→26.4 **11.46×** |
| 512 | 2 | 8.5→8.9 **0.96×** | 34.5→33.5 **1.03×** | 34.1→33.1 **1.03×** | 47.0→35.0 **1.34×** | 47.8→35.5 **1.35×** | 47.8→35.4 **1.35×** | 893.7→33.3 **26.83×** | 883.1→33.7 **26.20×** | 880.4→33.6 **26.23×** |
| 512 | 4 | 13.4→13.9 **0.96×** | 46.9→48.0 **0.98×** | 45.8→47.1 **0.97×** | 61.8→49.2 **1.26×** | 62.6→49.9 **1.25×** | 60.4→49.4 **1.22×** | 1792.6→44.5 **40.33×** | 1760.1→45.0 **39.12×** | 1739.5→44.8 **38.80×** |
| 2048 | 1 | 8.3→8.7 **0.96×** | 29.8→30.5 **0.98×** | 29.3→30.2 **0.97×** | 40.9→31.7 **1.29×** | 41.7→31.8 **1.31×** | 41.4→31.7 **1.31×** | 1412.7→31.2 **45.32×** | 1401.0→31.8 **44.08×** | 1396.7→31.4 **44.54×** |
| 2048 | 2 | 16.4→13.9 **1.18×** | 65.3→45.5 **1.43×** | 63.9→44.9 **1.42×** | 78.2→46.6 **1.68×** | 79.0→46.7 **1.69×** | 78.0→46.3 **1.69×** | 4267.5→41.0 **104.04×** | 4142.1→41.1 **100.66×** | 4114.8→41.0 **100.46×** |
| 2048 | 4 | 29.0→25.1 **1.16×** | 109.9→82.1 **1.34×** | 106.8→79.4 **1.34×** | 123.5→84.6 **1.46×** | 123.8→84.6 **1.46×** | 122.1→83.2 **1.47×** | 8895.9→71.3 **124.84×** | 8578.7→71.2 **120.54×** | 7766.8→69.8 **111.34×** |

**A100**

| top_k | q | 0% | 10%/h0 | 10%/h90 | 50%/h0 | 50%/h50 | 50%/h90 | 100%/h0 | 100%/h50 | 100%/h90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 1 | 10.2→10.4 **0.98×** | 55.8→53.8 **1.04×** | 55.7→53.8 **1.04×** | 71.6→56.7 **1.26×** | 72.2→57.0 **1.27×** | 87.7→68.7 **1.28×** | 810.9→60.1 **13.50×** | 807.8→60.0 **13.47×** | 807.1→59.9 **13.47×** |
| 512 | 2 | 14.8→15.0 **0.98×** | 67.4→64.1 **1.05×** | 66.8→63.6 **1.05×** | 86.3→66.5 **1.30×** | 86.8→67.3 **1.29×** | 98.0→76.7 **1.28×** | 1588.6→67.1 **23.66×** | 1580.2→66.7 **23.70×** | 1577.9→66.7 **23.65×** |
| 512 | 4 | 23.7→24.2 **0.98×** | 87.0→84.6 **1.03×** | 86.4→84.0 **1.03×** | 121.3→87.7 **1.38×** | 121.8→88.3 **1.38×** | 121.9→88.1 **1.38×** | 3164.3→84.7 **37.37×** | 3131.5→83.1 **37.70×** | 3115.3→82.8 **37.63×** |
| 2048 | 1 | 16.6→17.2 **0.96×** | 73.2→70.8 **1.03×** | 72.4→70.0 **1.03×** | 96.8→74.3 **1.30×** | 111.5→85.2 **1.31×** | 120.3→91.8 **1.31×** | 3508.3→76.3 **45.97×** | 3472.3→74.6 **46.55×** | 3457.0→74.3 **46.55×** |
| 2048 | 2 | 27.3→28.6 **0.95×** | 113.4→118.9 **0.95×** | 111.7→117.0 **0.95×** | 145.4→121.3 **1.20×** | 146.3→120.5 **1.21×** | 144.5→119.1 **1.21×** | 7079.3→113.2 **62.53×** | 6968.6→109.8 **63.49×** | 6911.4→107.7 **64.20×** |
| 2048 | 4 | 48.8→50.1 **0.97×** | 183.5→187.6 **0.98×** | 179.5→183.1 **0.98×** | 223.3→206.8 **1.08×** | 221.2→204.2 **1.08×** | 220.0→201.0 **1.09×** | 14584.8→180.8 **80.66×** | 14196.4→173.8 **81.67×** | 13504.5→168.0 **80.38×** |

### Host-side eager call cost (median µs, includes cudaFuncSetAttribute when smem > 48 KB)

| GPU | median Δ (all) | median Δ (main ≤48K, PR >48K) |
|---|---|---|
| H200 | -0.05 | +0.54 |
| B200 | -0.09 | +0.86 |
| A100 | -0.09 | +0.54 |

### Pruned

- H200: pruned: device_buffer_size < q*top_k — dbs=2048 k=2048 q=2, dbs=2048 k=2048 q=4
- B200: pruned: device_buffer_size < q*top_k — dbs=2048 k=2048 q=2, dbs=2048 k=2048 q=4
- A100: pruned: device_buffer_size < q*top_k — dbs=2048 k=2048 q=2, dbs=2048 k=2048 q=4
