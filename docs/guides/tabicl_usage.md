# TabICL setup, usage, diagnostics, and persistence

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

The TabICL adapter implements the explicit **`tabicl_v2_none_fp32_uncached`**
experimental profile described in
[the integration plan](../plans/tabicl_integration_plan.md). `norm_methods=["none"]`
is a project experiment choice, **not a universal requirement of TabICL**.
It differs from TabICL 2.2.0's default `["none", "power"]` normalization views.
Native train-fitted ordinal encoding, mean-imputer setup, constant-feature
filtering, standard scaling and outlier clipping still apply. Canonical
data are unchanged; categorical declarations are expressed by string dtypes on
copied DataFrames. Validation is never used as model context or for fitting
preprocessing. The adapter returns class-aware P(y=1), with the shared `>=0.5`
threshold policy.

The adapter, configuration/factory/environment hooks, native persistence,
tests and train/validation-only smoke command are implemented. Both frozen
n2000 contexts passed local correctness, persistence and resource gates;
see [stage-3 results](../results/tabicl_stage3_results.md) for the four-run evidence
and scope. **Both initial TabICL predictive runs are complete (M08)**; see the
[predictive results review](../results/tabicl_predictive_results.md). Their completion
is established by saved predictive artifacts, separately from local smoke checks.

The runtime requires exactly `tabicl==2.2.0`, pinned in both dependency files.
No PyTorch/CUDA replacement or other package upgrade was needed. The default
checkpoint filename alone is insufficient: the adapter
requires the fixed revision, 110368038-byte size and SHA-256 below,
checks the local file before native construction/fit, and always supplies
`allow_auto_download=False`. It never downloads weights, selects an automatic
device, retries with a different profile, or falls back to CPU model execution.

## Pinned checkpoint setup

The public, ungated `jingang/TabICL` checkpoint is
`tabicl-classifier-v2-20260212.ckpt` at revision
`4dcd344ece2c00be9e831fdd35bed57b5ad83e19`, SHA-256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
Code and checkpoint use BSD-3-Clause; see the
[versioned license/access evidence and citations](../plans/tabicl_integration_plan.md).
Only this setup step requires network access. On a new machine, download only
the pinned file if missing, then verify it before model construction. An
existing mismatching file is an error, never silently overwritten:

```bash
HF_HUB_OFFLINE=0 conda run --no-capture-output -n synthetic-tabular-ai python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
from src.models.tabicl import (
    CHECKPOINT_FILENAME, CHECKPOINT_REVISION, CHECKPOINT_SHA256,
    CHECKPOINT_SIZE, _sha256,
)

path = Path("checkpoints/tabicl") / CHECKPOINT_REVISION / CHECKPOINT_FILENAME
if not path.exists():
    hf_hub_download(
        repo_id="jingang/TabICL", filename=CHECKPOINT_FILENAME,
        revision=CHECKPOINT_REVISION, local_dir=path.parent, token=False,
    )
if path.stat().st_size != CHECKPOINT_SIZE or _sha256(path) != CHECKPOINT_SHA256:
    raise RuntimeError(f"Checkpoint identity mismatch; preserved for inspection: {path}")
print(f"Verified: {path}")
PY
```

The resulting local path is
`checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt`.
Checkpoints and all generated result/fitted-state/probability files stay out
of Git. All inference and reload commands use `HF_HUB_OFFLINE=1`.

The two **smoke-only** configurations are
[Original Synthetic](../../configs/smoke/tabicl_v2_source_none_fp32.json) and
[Playground S5E12](../../configs/smoke/tabicl_v2_kaggle_none_fp32.json). Their names,
parameters and metadata identify the none/FP32 experiment explicitly.
They use 8 TabICL ensemble members, native ensemble `batch_size=1`, external
query batch 128, AMP/FA3/cache disabled, CUDA device 0, and requested CPU
column-output storage. Native resource managers still choose internal splits
and may recover from OOM or change output placement. Captured Python
stdout/stderr/warnings are saved in `native_diagnostics.log` (and a separate
reload log). Routine native output is captured without console echo; Python
warnings and detected recovery/swap warnings remain visible. The research runner
prints flushed run/context/domain progress messages and final metrics, suitable
for cluster job logs. Detailed diagnostics remain enabled internally.
**`no_recovery_detected` is not proof that no scheduling changes
occurred.** Complete native scheduling is not exposed by a stable public API.
The adapter restores the previous torch thread count even if prediction fails;
do not run these adapters concurrently in one process.

Tests and CLI inspection (no real-model inference):

```bash
HF_HUB_OFFLINE=1 conda run -n synthetic-tabular-ai python -m unittest \
  discover -s tests -p 'test_tabicl*.py' -v
conda run -n synthetic-tabular-ai python -m src.experiments.smoke_tabicl --help
```

The completed smoke runs are recorded in the stage-3 evidence document.
If a future behavior change requires new evidence, use a **new** output directory:

```bash
# Small-context equivalence checks; repeat with the kaggle configuration.
HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python -m \
  src.experiments.smoke_tabicl \
  --config configs/smoke/tabicl_v2_source_none_fp32.json \
  --output-dir results/smoke/tabicl_v2_source_n512_none_fp32_run02 \
  --suite equivalence --sample-size 512 --query-size 65 --batch-size 128

# Frozen n2000 context, equivalence and resource screening; repeat for kaggle.
HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python -m \
  src.experiments.smoke_tabicl \
  --config configs/smoke/tabicl_v2_source_none_fp32.json \
  --output-dir results/smoke/tabicl_v2_source_n2000_none_fp32_run02 \
  --suite feasibility --sample-size 2000 --query-size 257 --batch-size 128
```

Do not use the shared research runner with these smoke-only configurations:
it evaluates test partitions. The smoke command loads only same-domain train
and validation, stably selects identity-sorted validation queries without label
selection, and computes no predictive metrics. Small checks use n512/65 queries;
feasibility requires n2000/at least 257 queries, frozen training identity hashes
and sequences, 128-row chunks, plus 2049 validation queries with warm-up and
three timed repetitions. Equivalence checks include repetitions, query
permutation, singleton/non-dividing chunks, sentinel composition, unseen
categories, input nonmutation/schema rejection and fresh-process native reload.
Existing output directories are refused; failures retain diagnostics and
exit nonzero, without changing settings or relaxing acceptance criteria.

The fixed FP32 comparison criterion is absolute difference
`<= 7.62939453125e-6`, relative tolerance zero; native probability row sums
must deviate from 1 by no more than `3.814697265625e-6`. These epsilon multipliers
are **engineering acceptance criteria, not mathematically guaranteed error
bounds**. No CLI tolerance overrides exist. All threshold flips are recorded
and block numerical eligibility pending review, including flips within the
boundary band. Systematic query-composition changes must not be dismissed as
floating-point drift. The adapter allowlist also permits explicitly requested
AMP under a separately named `tabicl_v2_none_amp_uncached` profile, but this
smoke command refuses it: FP32 evidence/tolerances do not certify that profile.

Resource screening records separate synchronized preparation, cold/warm
prediction, save and fresh-process reload timings and allocator peaks, host
memory/RSS and device free/total memory. Eligibility requires the fixed
allocator headroom gate and no **detected** recovery/swap fallback; allocator
headroom is not total-VRAM headroom. Both domains passed the bounded screen;
this does not prove arbitrary full-test workloads feasible.

Persistence uses official TabICL `save(..., save_model_weights=True,
save_training_data=True, save_kv_cache=False)` / `load` with
`model.tabicl_fit` and `model.adapter.json`. It embeds the weights, context
and fitted preprocessing; the sidecar records versions/profile/schema,
checkpoint provenance and artifact hash. Only load trusted native artifacts:
the official format is pickle-based. The initial loader enforces the same
dependency versions and **same saved device** because native unpickling uses
the saved device before applying a load override. It does not promise CPU-only
portability of a GPU-saved artifact. The smoke command uses a private temporary
checkpoint copy and removes only that copy before fresh-process offline reload,
leaving the original checkpoint untouched. All four real-model reloads passed
offline with exactly matching probabilities on the same saved CUDA device and
dependency versions.

## Verified local evidence and completed M08 runs

All four sequential checks (Original Synthetic and Playground S5E12, n512 and
n2000) passed numerical and reload gates. Both n2000 selections matched the
frozen saved CatBoost identities **and order**. Maximum probability difference
was `1.430511474609375e-6`, within the fixed engineering criterion, with zero
threshold disagreements. Both n2000 resource gates passed; maximum PyTorch
allocated/reserved peaks were 361.323/500 MiB. No recovery/swap was detected,
subject to the diagnostic limitations above.

| n2000 train / validation domain | Preparation (s) | 2049-query median (s) | Rows/s |
|---|---:|---:|---:|
| Original Synthetic | 1.929 | 31.564860 | 64.914 |
| Playground S5E12 | 1.377 | 32.908128 | 62.264 |

Preparation loads weights and prepares context/preprocessing, not gradient
training. Uncached prediction repeats substantial context computation per
external chunk; native CPU column-output storage does not mean CPU model
execution. The timing screen used validation only, warm-up plus three
repetitions, and is **not measured full-test performance**. Arbitrary inputs,
devices, versions and profiles are not certified by these bounded checks.

The research configurations are
[Original Synthetic](../../configs/tabicl_v2_source_n2000_none_fp32.json) and
[Playground S5E12](../../configs/tabicl_v2_kaggle_n2000_none_fp32.json). They preserve
the validated profile, both seeds 42, the exact 2000-row sampling protocol,
`strict_common_16`, threshold 0.5, ten calibration bins and same-domain
validation (unused by the adapter). Output names include generation, domain,
budget, none/FP32/uncached profile and seed, distinct from smoke/existing runs.
Equal ensemble counts across model families do not imply equal compute.

**M08: both full predictive runs below completed after M07.**
See [the result comparison and runtime review](../results/tabicl_predictive_results.md).
Existing output directories are protected; repeating these runs requires new
experiment/output names. The legacy reporting manifest selects only
CatBoost/TabPFN. E04 adds the separate
schema-version-2 report with both TabICLv2 runs, paired CatBoost comparisons,
and validated three-model figures.
Each command evaluates all three full test domains: Original Synthetic,
Playground S5E12 and NHANES 2015–2016. Do not substitute these commands for
validation-only smoke checks:

```bash
HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python \
  -m src.experiments.runner --config configs/tabicl_v2_source_n2000_none_fp32.json
HF_HUB_OFFLINE=1 conda run --no-capture-output -n synthetic-tabular-ai python \
  -m src.experiments.runner --config configs/tabicl_v2_kaggle_n2000_none_fp32.json
```

E04b additionally provides an explicit eight-run manifest with FT-Transformer;
see [reporting](reporting.md).
