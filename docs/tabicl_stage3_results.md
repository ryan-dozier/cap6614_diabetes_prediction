# TabICL stage 3: local correctness and feasibility

## Scope and conclusion

**Ready for stage 4:** all four sequential attempts passed their declared
execution, numerical, and persistence gates; both frozen n2000 contexts also
passed the prescribed local resource screen. There were no failed attempts or
implementation corrections. M07 remains unchecked pending stage-4 final review;
M08 and predictive research runs are unchanged. Subsequent stage-4 acceptance
is recorded in [the integration plan](tabicl_integration_plan.md#12-final-acceptance-and-stage-4-handoff)
and [checklist](../PROJECT_CHECKLIST.md); the unchecked M07 references below
describe the stage-3 boundary.

This stage evaluates the explicit experimental profile
`tabicl_v2_none_fp32_uncached`, not the authors' default benchmarking profile or
a universal TabICL requirement. Only training and validation partitions are
loaded. No accuracy, ROC-AUC, other predictive metrics, or test-partition
inference are used.

The predeclared protocol is in [the integration plan](tabicl_integration_plan.md).
All attempts use new output directories; there is no automatic retry with a
different model configuration or relaxed acceptance criterion.

## Locally verified environment and checkpoint

- Environment: `synthetic-tabular-ai`, executable
  `/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python`.
- Python 3.11.17; TabICL 2.2.0; PyTorch 2.14.1+cu126; NumPy 2.4.6;
  pandas 3.0.6; scikit-learn 1.9.1; SciPy 1.17.1; huggingface-hub 2.2.0;
  einops 0.8.2; psutil 7.2.2.
- Linux/WSL2, x86_64; CUDA runtime 12.6; driver 617.42.
- NVIDIA GeForce RTX 4060 Laptop GPU, capability 8.9,
  8,585,216,000 bytes total device memory (8188 MiB).
- Host RAM: 16,374,460,416 bytes. Per-operation available RAM and process RSS
  are saved in each report, rather than assumed constant.
- Before the first run, `nvidia-smi` reported 0 MiB used, 0% utilization, and
  no compute processes. This is an initial observation, not exclusive GPU
  reservation throughout the runs.
- CUDA matmul TF32 false; cuDNN TF32 true. AMP and FA3 are disabled.

The single official checkpoint was acquired with installed `huggingface_hub`,
`token=False`, the exact filename and fixed revision below. No package was
installed or upgraded. Before any model construction, local size and SHA-256
were verified:

| Field | Verified value |
|---|---|
| Repository | `jingang/TabICL` |
| Revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| Filename | `tabicl-classifier-v2-20260212.ckpt` |
| Size | 110368038 bytes |
| SHA-256 | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| Project path | `checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt` |

Network access was used only for checkpoint acquisition. All smoke commands
and fresh-process reloads use `HF_HUB_OFFLINE=1`, explicit local paths, and
`allow_auto_download=False`. Checkpoints and run artifacts are ignored by Git.

Acquisition used this Python API call under
`env HF_HUB_OFFLINE=0 conda run --no-capture-output -n synthetic-tabular-ai python`.
The surrounding script skipped download if the file existed and asserted the
returned path, size, and SHA-256 before proceeding:

```python
hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
    revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
    local_dir="checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
    token=False,
)
```

## Fixed configuration and acceptance criteria

- `strict_common_16`; seed 42; sampling seed 42; threshold 0.5.
- Training and validation come from the same domain for each command.
  Identity disjointness is checked before fitting.
- `norm_methods=["none"]`, `use_amp=false`, `use_fa3=false`, `kv_cache=false`,
  eight estimators, native **ensemble** batch size 1, requested column-output
  offload `"cpu"`, four native threads, `device="cuda:0"`.
- External **query** batch size 128. No early stopping, calibration, or
  validation-context fitting. Native preprocessing remains train-fitted.
- Fixed probability comparison: absolute tolerance
  `7.62939453125e-6` (64 FP32 epsilons), relative tolerance zero.
  This is an engineering acceptance criterion, not a mathematically guaranteed
  error bound. Any threshold disagreement blocks numerical eligibility.
- Native binary probabilities must be finite, in range, shape `(n, 2)`, with
  rows summing to one within `3.814697265625e-6`; the adapter selects class 1
  using `classes_`. Saved adapter probability arrays have shape `(n,)`.
- Fixed allocator headroom: at least 512 MiB below total device memory for
  every recorded CUDA allocator peak reservation, including reload operations.
  This is not a guarantee about total VRAM consumed by every CUDA allocation.
- Native requested CPU column-output offload is not an unplanned CPU model
  fallback. No detected OOM recovery or swap-tier fallback is permitted for
  resource eligibility.

## Commands and attempt status

All commands were run from the repository root in the order below, inspecting
the completed report, diagnostics, raw probabilities, and reload result before
advancing. `run01` directories were absent before execution.

```bash
HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python \
  -m src.experiments.smoke_tabicl \
  --config configs/smoke/tabicl_v2_source_none_fp32.json \
  --output-dir results/smoke/tabicl_v2_source_n512_none_fp32_run01 \
  --suite equivalence --sample-size 512 --query-size 65 --batch-size 128

HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python \
  -m src.experiments.smoke_tabicl \
  --config configs/smoke/tabicl_v2_kaggle_none_fp32.json \
  --output-dir results/smoke/tabicl_v2_kaggle_n512_none_fp32_run01 \
  --suite equivalence --sample-size 512 --query-size 65 --batch-size 128

HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python \
  -m src.experiments.smoke_tabicl \
  --config configs/smoke/tabicl_v2_source_none_fp32.json \
  --output-dir results/smoke/tabicl_v2_source_n2000_none_fp32_run01 \
  --suite feasibility --sample-size 2000 --query-size 257 --batch-size 128

HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python \
  -m src.experiments.smoke_tabicl \
  --config configs/smoke/tabicl_v2_kaggle_none_fp32.json \
  --output-dir results/smoke/tabicl_v2_kaggle_n2000_none_fp32_run01 \
  --suite feasibility --sample-size 2000 --query-size 257 --batch-size 128
```

| Dataset (internal ID) | Context | Execution | Numerical | Persistence | Resource |
|---|---:|---|---|---|---|
| Original Synthetic (`source`) | 512 | Passed | Passed; max difference 7.7486038208e-7; 0 flips | Passed; reload bitwise equal | Not certified by equivalence-only suite |
| Playground S5E12 (`kaggle`) | 512 | Passed | Passed; max difference 1.0132789612e-6; 0 flips | Passed; reload bitwise equal | Not certified by equivalence-only suite |
| Original Synthetic (`source`) | 2000 | Passed | Passed; max difference 1.4305114746e-6; 0 flips | Passed; reload bitwise equal | Passed |
| Playground S5E12 (`kaggle`) | 2000 | Passed | Passed; max difference 1.1324882507e-6; 0 flips | Passed; reload bitwise equal | Passed |

The `resource_eligible=false` field in an equivalence-only report is expected,
not a failed feasibility result.

### Training identities

The full exact selected sequences and order are retained in each run's
`training_sample.csv.gz`, with explicit `sample_order`, `row_index`, domain,
and split. Both n2000 contexts are checked against the corresponding existing
CatBoost `training_sample.csv.gz`, not regenerated approximately or reshuffled.

| Dataset | Context | Class 0 / class 1 |
|---|---:|---:|
| Original Synthetic | 512 | 205 / 307 |
| Playground S5E12 | 512 | 193 / 319 |
| Original Synthetic | 2000 | 800 / 1200 |
| Playground S5E12 | 2000 | 753 / 1247 |

Verified Original Synthetic n2000 identities:

- Ordered: `7d81147cce93cd5c56b2e66484f9f8902ae4ef1784bab270c32016e43f44d170`.
- Unordered: `239741018e0fa7b477a8985fdf7d214c05cd43b708b28ab061946e709b4eb9a0`.
- Exact saved reference:
  `results/predictive/catboost_source_strict_common_16_n2000_seed42/training_sample.csv.gz`.

Verified Playground S5E12 n2000 identities:

- Ordered: `612fcfb9dae65a36b591f44065baef3cf629b028435737db4a5480b09469fe3b`.
- Unordered: `d4eca71ee3d52a04818645f1eb01c33ea26eeea0f2bde38bfab8cbb8ebabd63e`.
- Exact saved reference:
  `results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/training_sample.csv.gz`.

The Original Synthetic timing screen selected 2,049 identity-sorted validation
rows without label selection, ordered hash
`f364e769d2ea078cbdc8232e6bd9528cec430df8d79322cace91761975288902`.
Validation query identities are saved separately for the equivalence and timing
screens.

The Playground S5E12 timing screen likewise selected 2,049 identity-sorted
validation rows, ordered hash
`3fa75bf54ccf64cec46e7004eb9eb2e733de10bbb8ddec0bdc251a9552ea3c28`.

## Correctness and persistence evidence

Both n512 reports pass three repeated identical predictions, unchunked
prediction, query chunks 1/17/32/128, restored query permutation, sentinels
alone, unseen categories, feature-order/extra/missing-feature rejection,
missing/nonfinite-value rejection, and fit/predict input nonmutation.
All 13 probability arrays per n512 run were inspected for finite values and
expected row counts. Repeated inference and restored permutations were
bitwise equal; smaller chunk/sentinel changes stayed below the fixed gate.

Each run saves native fitted state with embedded weights, training context,
and preprocessors, plus an adapter sidecar. Its private checkpoint copy is
removed before launching a fresh Python process for reload. The project
checkpoint remains untouched. Both n512 reloads completed offline with
`classes_=[0,1]`, the isolated checkpoint absent, and zero probability
difference. This certifies only the tested same-device, same-dependency
configuration; no cross-device/cross-version portability is claimed. Native
fitted-state files are trusted pickle-based artifacts, not safe untrusted input.

Both n2000 reports passed the same mandatory checks with 257 equivalence
queries, as well as all three 2,049-query timing comparisons. All 17 probability
arrays per n2000 run were checked for finite in-range values. Both native
reloads completed in a fresh offline process after the isolated checkpoint
was removed, with zero probability difference. Effective metadata confirms
eight estimators, `classes_=[0,1]`, no validation fitting, and the unchanged
explicit none/FP32/uncached profile.

## Native diagnostics and interpretation

Per-run evidence includes `smoke.json`, `reload.json`,
`native_diagnostics.log`, `native_reload.log`, `reload_process.log`,
`model.adapter.json`, training/query identities, and raw `.npy` probabilities.
The two n512 native logs each contain 7,944 lines and 83 captured operations;
the fresh-process reload logs were also retained.

Both n512 reports say **no recovery detected**, with no swap fallback detected.
Captured Python stdout/stderr and warnings are not a complete trace of native
execution. These findings do not prove that internal scheduling or memory-based
splits were unchanged. Effective initialization and resolved inference settings
are preserved in model metadata. No configuration changes were used to obtain
these passes.

Both n2000 reports likewise record no recovery detected and no swap fallback.
Each main native log contains 36,364 lines; each native reload log has 296
lines and each reload process log 288 lines. Spot checks show requested CPU
column-output storage with GPU row/ICL operations. Fresh-process reload
diagnostics are retained separately.

## Timing, resources, and remaining gates

Original Synthetic n512 complete suite: 129.013 seconds.
Playground S5E12 n512 complete suite: 121.648 seconds.
These include numerous uncached equivalence operations and persistence work;
they are not full-workload performance measurements.

| n512 dataset | Preparation (s) | Cold prediction (s), 65 rows | Save (s) | Reload (s) | Reload prediction (s) | Max allocated / reserved (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Original Synthetic | 2.618 | 3.711 | 1.177 | 2.561 | 2.754 | 182.747 / 206 |
| Playground S5E12 | 1.601 | 2.519 | 1.257 | 2.696 | 2.446 | 182.747 / 206 |

All recorded n512 allocator-headroom checks passed, but resource certification
still requires the prescribed n2000 context and larger validation screen.
Native log spot checks show requested CPU column-output storage and GPU row/ICL
operations, not a CPU model fallback.

The n2000 runs additionally verified exact frozen training identities/order,
completed all correctness and reload checks, and performed a separate 2,049-row
validation-query screen: warm-up followed by three measured repetitions.
Individual preparation/prediction/save/reload durations, allocator peaks,
before/after device-free memory, RSS, and host-available RAM are retained
per operation in the reports. Tables below summarize key operations; the
reports retain every repetition, chunking check, permutation, and sentinel.

### Original Synthetic n2000 measured screen

All measurements below are synchronized wall-clock seconds, except where units
are stated otherwise. Inference is uncached; repeated external chunks repeat
context computation. The complete suite's counter duration was 688.828 seconds.

| Operation | Duration (s) | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|
| Preparation | 1.929 | 108.120 | 120 |
| Cold prediction, 257 queries / chunk 128 | 6.961 | 348.577 | 478 |
| Unchunked, 257 queries | 2.013 | 361.323 | 500 |
| Single-query chunks, 257 queries | 448.240 | 338.952 | 382 |
| Timing warm-up, 2049 queries / chunk 128 | 32.525 | 348.577 | 478 |
| Timing repetition 1 | 31.565 | 348.577 | 478 |
| Timing repetition 2 | 31.776 | 348.577 | 478 |
| Timing repetition 3 | 31.322 | 348.577 | 478 |
| Native save | 0.891 | 116.245 | 180 |
| Fresh-process native reload | 2.618 | 108.120 | 120 |
| Reload prediction, 257 queries / chunk 128 | 7.202 | 348.577 | 478 |

Timing median: **31.564860 seconds, 64.913958 rows/second**. All three timing
repetitions were bitwise equal to warm-up probabilities. The largest numerical
difference, 1.4305114746e-6, occurred with query chunks of 32. Repeats with
identical queries/chunks and reload were bitwise equal. Permutation restoration
had maximum difference 1.7881393433e-7, within the fixed criterion.

All per-operation allocator-headroom checks passed: the largest reserved peak
was 500 MiB against an 8188 MiB device (512 MiB required headroom). Observed
before/after free device memory ranged 6886-7096 MiB; these are boundary
snapshots, not continuous minima. Process RSS snapshots ranged
757.977-1669.641 MiB, and host-available RAM 7909.246-9631.867 MiB.
These host snapshots are not host peak-allocation measurements.

### Playground S5E12 n2000 measured screen

Complete suite counter duration: 669.124 seconds.

| Operation | Duration (s) | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|
| Preparation | 1.377 | 108.120 | 120 |
| Cold prediction, 257 queries / chunk 128 | 6.521 | 348.577 | 478 |
| Unchunked, 257 queries | 2.054 | 361.323 | 500 |
| Single-query chunks, 257 queries | 424.284 | 338.952 | 382 |
| Timing warm-up, 2049 queries / chunk 128 | 32.165 | 348.577 | 478 |
| Timing repetition 1 | 31.263 | 348.577 | 478 |
| Timing repetition 2 | 32.933 | 348.577 | 478 |
| Timing repetition 3 | 32.908 | 348.577 | 478 |
| Native save | 0.985 | 116.245 | 180 |
| Fresh-process native reload | 2.350 | 108.120 | 120 |
| Reload prediction, 257 queries / chunk 128 | 6.921 | 348.577 | 478 |

Timing median: **32.908128 seconds, 62.264253 rows/second**. All three timing
repetitions were bitwise equal to warm-up probabilities. The largest numerical
difference, 1.1324882507e-6, occurred with chunks of 1 and 32. Identical repeated
inference and reload were bitwise equal; permutation restoration had maximum
difference 1.7881393433e-7.

All allocator-headroom checks passed; maximum reservation was 500 MiB and
maximum allocation 361.323 MiB. Boundary free device memory ranged
6886-7096 MiB. RSS snapshots ranged 758.059-1971.723 MiB and host-available RAM
7576.484-9309.219 MiB. Again, these are boundary snapshots, not continuous
host-memory peaks or total-VRAM peaks.

These screens establish feasibility only for these two contexts and bounded
validation workloads on this machine/runtime. They do not establish throughput
for full test partitions, arbitrary query sizes, other profiles, or concurrent
models. Uncached single-row chunks were intentionally slow and were not skipped.
Numerical checks passed their engineering criteria; this is not a mathematical
proof of equivalence on arbitrary inputs.

## Changes and validation

No implementation correction was necessary. The user reported
114 passing tests before stage 3; these document/result-only steps do not
repeat that full suite. No configuration, dependency, checklist, or canonical
data files were changed. No commit, push, PR, or research experiment was made.

All four smoke commands exited zero, and the actual reports, native diagnostic
and reload logs, saved probability arrays, and identity files were inspected.
The checkpoint was verified before construction and reused unchanged.
Final checks include `git diff --check`, report/artifact consistency checks,
Git-ignore checks for all generated run files and the checkpoint, and review
of branch/worktree status. Existing uncommitted stage-1/2 work is preserved.

## Stage-4 handoff

- No unresolved blocker remains for final configuration/documentation/PR
  preparation under this exact profile. Keep its explicit experimental name
  and checkpoint pin; do not relabel it the native/default TabICL benchmark.
- Review this evidence alongside the implementation and final research
  configurations. Stage 4 replaced the adapter metadata's stale source-only
  wording with a reference to these bounded checks and an explicit warning
  that this historical evidence does not certify a future run or arbitrary
  inputs, devices, versions or profiles. No inference behavior changed.
- Retain external query chunks of 128 for the intended comparison. Do not
  extrapolate the timing screen into a measured full-test runtime.
- Preserve same-device/same-dependency trusted-artifact persistence limits,
  native scheduling caveats, and the distinction between allocator headroom
  and total VRAM/host-memory peaks.
- Keep M07 unchecked until final acceptance review. Do not alter M08 or launch
  test-partition research runs as part of this stage.
