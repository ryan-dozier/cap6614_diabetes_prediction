# M07 TabICL integration: stage-1 compatibility and adapter design

Inspection date: 2026-10-25 (local date). Scope: stage 1 of 4.

**Current status after stages 2-4:** the bounded adapter is implemented; all
four local checks passed, including both frozen n2000 feasibility gates.
See [stage-3 evidence](../results/tabicl_stage3_results.md) and the
[current README](../guides/tabicl_usage.md).
Research configurations and the TabICL 2.2.0 dependency pin are finalized.
M07 acceptance is complete; M08 full predictive runs remain pending.
Sections 1-10 preserve the original inspection and its then-unresolved facts,
not the current execution status.

**Recommendation: ready for stage-2 implementation of the bounded protocol
below, not ready for M08 experiments.** The installed API, offline loading
path, and official persistence API are sufficiently established. A local
checkpoint has not been found, checkpoint contents have not been loaded, and
real-model feasibility on 8 GB is unknown. Neither M07 nor M08 is completed by
this document.

Evidence labels used throughout:

- **L**: locally observed metadata, signatures, hardware, files, or source.
- **V**: official versioned source/model metadata, not a local inference result.
- **P**: proposed project policy, requiring implementation and tests.
- **U**: unverified behavior requiring a later check.

## 1. Repository and environment inspection

### Repository safety and research boundary

The initial branch was already `feat/tabicl-adapter`, with a clean worktree,
at commit `41e282f` (`adding a checklist of future items`). No branch switch was
needed. No applicable `AGENTS.md` was found in the repository or its ancestor
directories; no repository `.github` instruction directory was present.
Only this design document is added. No fetch, pull, install, upgrade, checkpoint
download, model construction, fit, predict, commit, push, or PR operation was
performed.

Inspected [M07/M08](../../PROJECT_CHECKLIST.md), [README](../../Readme.md),
dependency declarations, the adapters and experiment modules listed below,
smoke/benchmark source and relevant tests, and
[the shared display mapping](../../src/domain_labels.py).
No test partition or saved predictive score was read. The only experiment
data artifacts read were four **training-sample identity files**, not their
features, labels, predictions, or metrics.

### Exact inspected environment (L)

All Python commands used the explicit executable below, not the editor-selected
environment:

```text
/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python
Python 3.11.17 (GCC 14.3.0)
tabicl                 2.2.0
torch                  2.14.1+cu126
numpy                  2.4.6
pandas                 3.0.6
scikit-learn           1.9.1
scipy                  1.17.1
huggingface-hub        2.2.0
einops                 0.8.2
safetensors            0.8.0
```

Installed package root:
`/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/lib/python3.11/site-packages/tabicl`.
Distribution metadata has no `direct_url.json`; install origin is not established
from that metadata alone. Import and public signature inspection succeeded with
`HF_HUB_OFFLINE=1`, after inspecting metadata and source. No estimator was created.

CUDA is available; PyTorch CUDA runtime is 12.6. Device 0 is NVIDIA GeForce
RTX 4060 Laptop GPU, compute capability 8.9, reported total memory
8,585,216,000 bytes. One PyTorch observation reported 7,440,695,296 free bytes.
Separately, `nvidia-smi` reported driver 617.42, 8188 MiB total and 7919 MiB free.
These are different-time observations, not a reserved memory budget.
Host RAM was 16,374,460,416 bytes total, 11,384,135,680 available at inspection.
No inference memory or speed measurements exist for this integration.

CUDA's observed default autocast dtype is float16.
`torch.backends.cuda.matmul.allow_tf32` was false; cuDNN's TF32 flag was true.
Record these again during smoke testing rather than assuming another process
or machine has identical arithmetic settings.

### Dependencies and checkpoint inventory (L/V)

Installed metadata and the [release pyproject][pyproject] require Python >=3.10,
torch >=2.2, scikit-learn >=1.3.0, einops >=0.7, plus numpy, scipy, psutil,
tqdm >=4.64.0 and huggingface-hub. `python -m pip check` returned
`No broken requirements found.` The inspected versions satisfy declared bounds;
that is not proof of full pandas/sklearn/runtime compatibility under inference.
The DataFrame path needs pandas, already installed.

Neither [environment.yml](../../environment.yml) nor
[requirements-local.txt](../../requirements-local.txt) currently declares TabICL.
Both already pin the relevant existing dependencies above. Later add only
`tabicl==2.2.0`; do not regenerate the whole environment or upgrade torch,
numpy, pandas, sklearn, CUDA, or Hub packages without a demonstrated need.
No forecast, SHAP, pretraining, fine-tuning, or `all` extras are needed.

No TabICL entries were found in the inspected default Hugging Face cache,
the bounded `~/.cache` search, repository checkpoint-name search, or installed
TabICL package weight-file search. `~/.cache/tabicl` does not exist.
`HF_HOME`, `HF_HUB_CACHE`, `XDG_CACHE_HOME`, and `TABICL_CACHE_DIR` were unset.
**No local checkpoint was found in these locations; arbitrary external paths
were not exhaustively searched.** Do not infer that weights are bundled with
the wheel or that an import proves checkpoint availability.

## 2. Versioned source and checkpoint identity

Official tag [`v2.2.0`][tag] resolves to commit
`0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`.
The installed bytes matched that commit for all eleven inspected files:
`__about__.py`, `_sklearn/{classifier,base,preprocessing}.py`, and
`_model/{tabicl,embedding,encoders,layers,learning,inference,inference_config}.py`.
Thus the source links below describe the installed release, not an assumed
latest API. Other package files were read where needed but were not all
byte-compared against the tag.

Proposed checkpoint (V/P):

| Identity | Fixed value |
|---|---|
| Model family | TabICLv2 classifier, not TabICLv1 or the regressor |
| Repository | `jingang/TabICL` |
| Repository revision | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| Filename | `tabicl-classifier-v2-20260212.ckpt` |
| Published LFS SHA-256 | `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` |
| Published file size | 110,368,038 bytes |
| Proposed local location | `checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt` |

The [fixed-revision Hub metadata][checkpoint-api] supplies revision, size and
LFS digest. These are **published identities**, not a locally computed weight
hash. Its [model card][card] advertises BSD-3-Clause. API metadata says public,
not gated, not disabled; no token or acceptance flow was required to read it.
Actual checkpoint retrieval has not been attempted.

The [code license][license] is BSD-3-Clause, with a separate Apache-2.0 notice
for derived forecasting code (not used here). Preserve applicable notices on
redistribution, including if redistributing embedded weights. Cite both the
TabICL ICML 2025 paper ([2502.05564](https://arxiv.org/abs/2502.05564)) and
TabICLv2 ([2602.11139](https://arxiv.org/abs/2602.11139)), as requested by the
model card. Do not import TabPFN's checkpoint terms into this protocol.

### Evidence table

Line ranges refer to release source at the fixed commit above.

| Question | Finding | Source | Level / confidence |
|---|---|---|---|
| Which release is local? | 2.2.0 metadata, successful offline import and signature checks; inspected source matches tag. | Local `importlib.metadata`, `inspect.signature`; [tag][tag], [pyproject][pyproject] | L+V, high; no inference |
| Which checkpoint is compatible? | Classifier explicitly recognizes the v2 filename above, plus v1/v1.1 alternatives. | [classifier.py lines 331-421][classifier] | L+V, high; checkpoint payload U |
| Can defaults change/download weights? | Default filename is fixed in this release, but Hub calls omit revision; a branch's file contents can change. `model_path=None` tries local Hub cache then downloads if allowed. | [classifier.py lines 331-421][classifier] | L+V, high |
| Offline local-only path? | Existing explicit path goes directly to `torch.load(..., map_location="cpu", weights_only=True)`; missing path with auto-download false raises. | [classifier.py lines 371-421][classifier] | L+V, high |
| Categorical API? | No categorical-index constructor parameter. DataFrame dtypes select ordinal encoding; numeric codes alone are numeric. | [classifier.py lines 282-329][classifier]; [preprocessing.py lines 58-181][preprocessing] | L+V, high |
| Strings and unseen categories? | String/object/category/nullable-boolean selection; ordinal encoder uses unknown and encoded-missing value -1. Numeric imputer is fit on context. | [preprocessing.py lines 101-165][preprocessing] | L+V, high; pandas 3 runtime U |
| Fit behavior? | Loads pretrained model onto device, fits label/input encoders and ensemble preprocessing, stores context; optionally computes caches. No gradient training. | [classifier.py lines 423-563][classifier] | L+V, high |
| Output classes/shape? | LabelEncoder supplies sorted `classes_`; probabilities have query rows by classes, undo class shuffles and normalize rows. | [classifier.py lines 475-478, 667-779][classifier] | L+V, high; runtime U |
| Does batch_size split queries? | No: classifier batching splits transformed ensemble datasets on axis 0. Query rows need the shared adapter's external chunks. | [classifier.py lines 565-665][classifier] | L+V, high |
| Query-set statistics? | Standard preprocessing parameters fit only on train. Column inducing attention and ICL keys/values use train only; row interaction is per row. | [preprocessing.py lines 693-780][preprocessing]; [embedding.py lines 340-420][embedding]; [layers.py lines 430-480, 625-650][layers]; [interaction.py lines 215-252][interaction] | L+V, high for inspected path |
| Always query-independent? | No unconditional guarantee: normalizer ValueError triggers clipping of the entire supplied query array before retry. All-NaN query-column masking is also outside our contract. | [preprocessing.py lines 751-780][preprocessing]; [classifier.py lines 667-687][classifier] | L+V, high; triggered example U |
| Length-based attention scaling? | SSMax scaling receives key/source length, not number of query rows. ICL Encoder is built without row RoPE. | [attention.py lines 97-108][attention]; [learning.py lines 94-110][learning]; [encoders.py lines 76-118][encoders] | L+V, high; checkpoint settings U |
| Automatic precision? | `"auto"` switches AMP/FA3 based on training sizes/device. Explicit false values avoid those heuristics. | [base.py lines 68-176][base] | L+V, high |
| Automatic memory recovery? | Memory estimates determine internal batches; multi-batch CUDA OOM halves batch and retries, printing only when verbose. Even requested GPU/CPU output placement can cascade to another storage tier. | [inference.py lines 1035-1126, 1279-1463][inference] | L+V, high |
| Official fitted-state API? | `save`/`load` exist; official format uses pickle protocol 5 with fitted state, optional weights/context/cache. Not a generic project pickle invention. | [base.py lines 212-430][base] | L+V, high; round trip U |
| Device portability? | Weights/cache are serialized on CPU, but deserialization resolves saved device before load's device override. CPU-only relocation is not guaranteed. | [base.py lines 295-333, 399-430][base] | L+V, high |
| Fits 8 GB at n2000? | Unknown; hardware is available, no model work done. Published large-data/H100 examples do not prove laptop feasibility. | Local CUDA checks; [release README][readme] | L/V for inputs; U for feasibility |

## 3. Existing contracts and minimal extension points

There is no registry to redesign. [create_adapter](../../src/experiments/runner.py#L122)
is a small explicit factory with lazy model imports.
[ExperimentConfig](../../src/experiments/config.py#L88) accepts only CatBoost and
TabPFN, dispatches a parameter allowlist, and currently gives TabPFN-specific
wording for the non-CatBoost early-stopping rejection.

[ModelAdapter](../../src/models/base.py) already provides exact schema/order
validation, copy-on-input, categorical `astype(str)`, missing/nonfinite rejection,
external query chunks, positive-class lookup, fixed-threshold prediction,
and resource hooks. Its required `fit(train, labels, val, labels)` signature does
not require using validation. Follow the existing pretrained adapter's refusal
to inspect validation, including tests using unusable validation sentinels.

Proposed contract (P):

1. `TabICLAdapter(features, categorical_features, target, *, seed, model_params)`
   subclasses the shared base. Runtime package import remains lazy; enforce
   exactly TabICL 2.2.0 and verify required constructor/save/load signatures.
2. Validate seed and config, explicit device availability, resolved local
   checkpoint filename, size and SHA-256 **before estimator construction**.
   Resolve relative paths against the project root, not the caller's CWD.
   Missing/mismatched checkpoint, unavailable CUDA, invalid inputs or wrong
   package version fail explicitly; no network, CPU fallback or retry profile.
3. `fit` rejects refit, validates copied training predictors and binary labels
   with both classes and equal length, calls official `fit(train, y.copy())`
   once, and never passes or reads validation. Load/preprocessing/device
   transfer belong inside `fit`, so synchronized preparation timing includes
   them. No eager weight load in adapter initialization.
4. `predict_proba` validates/copied inputs, invokes the estimator, validates
   `(n, 2)` and classes `{0,1}` through `_positive_probability`, returning
   float64 `(n,)` P(y=1) in original row order. Do not assume column 1 without
   checking classes, and do not use native `predict` for threshold tie behavior.
   The shared `>= 0.5` policy is authoritative.
5. Use inherited external batching, GPU synchronization/allocator hooks, and
   JSON-compatible metadata. Inspect actual ensemble configurations rather
   than copying TabPFN's `n_estimators_` attribute (not TabICL's API).
   Require the effective count to equal the requested count; otherwise fail.

### Minimum file changes for later stages

| File/surface | Required work |
|---|---|
| New `src/models/tabicl.py` | Bounded adapter, checkpoint identity guard, native persistence wrapper and sidecar, metadata/resource hooks; no base-class redesign. |
| [config.py](../../src/experiments/config.py) | Add model ID, TabICL allowlist and validation; generalize pretrained early-stopping error without weakening CatBoost behavior. |
| [runner.py](../../src/experiments/runner.py) | Lazy factory branch, train-only selection-policy text, TabICL/torch environment versions, TabICL OOM diagnostic context. Retain per-domain full-test loop for M08. |
| [results.py](../../src/experiments/results.py) | Existing atomic JSON, protected run directory, sample/prediction writers suffice; no format change required. |
| [sampling.py](../../src/experiments/sampling.py) | No algorithm/order change. Reuse exactly; assert frozen identity hashes below. |
| New focused tests | Mock official estimator/runtime for adapter, config, factory, optional imports, save/load and failure contracts; no downloaded weights or real inference in unit suite. |
| New train/val-only smoke command | Separate TabICL command; reuse existing sampler, partition loader, result writers and measurement-hook pattern, not TabPFN numerical constants. |
| Later `configs/tabicl_{source,kaggle}_n2000.json` | Stage 4 only after stage-3 evidence; distinct model/settings/output names. |
| [README](../../Readme.md), [environment.yml](../../environment.yml), [requirements-local.txt](../../requirements-local.txt) | Later document chosen profile, native persistence limitations and authorized checkpoint acquisition; add exact TabICL pin only. |

The [runner CLI](../../src/experiments/runner.py#L284) reads a JSON config; no model
CLI choices exist to update. Its help remains generic. New smoke CLI help must
say train/validation only, local checkpoint required, never downloads, and
explain ensemble versus query batch sizes. `--help` and imports must work
without importing torch/tabicl/catboost. Existing
[optional-import tests](../../tests/test_tabpfn_sampling.py) provide the pattern.
[src/models/__init__.py](../../src/models/__init__.py) need not eagerly export models.

Existing commands inspected:
`python -m src.experiments.smoke_tabpfn`,
`python -m src.experiments.benchmark_tabpfn`, and
`python -m unittest discover -s tests -v`.
The first two intentionally reject other models and should remain unchanged.
Their measurement and deterministic query-selection patterns are useful prior
art, but their batch reference, precision and tolerances are not transferable.
Do not route stage-3 checks through `runner.run`: it eventually loads tests.

The existing [reporting validation](../../src/reporting/validation.py),
[tables](../../src/reporting/tables.py) and
[figure mapping](../../src/reporting/figures.py) deliberately target specific
CatBoost/TabPFN comparisons. Adding TabICL research reports is E04 work after
M08, not an incidental stage-2 redesign. Keep raw result schemas compatible.
No change to [domain labels](../../src/domain_labels.py) is needed.

## 4. Proposed allowlist and first local profile

This is a deliberately non-default **TabICLv2, none-normalization,
uncached, FP32 profile**, not a claim to reproduce the authors' default
benchmark pipeline. All settings must be materialized in metadata.

| User model parameter | Validation / initial choice | Rationale |
|---|---|---|
| `model_path` | Required nonempty local file path; exact identity above | No implicit model/cache selection. |
| `device` | Explicit `cpu` or `cuda:<nonnegative index>`; initial `cuda:0` | CPU is an explicitly requested diagnostic option only. |
| `n_estimators` | Strict positive int, not bool; initial **8** | Official TabICL default, not TabPFN's ensemble count. Different ensemble families/compute. |
| `batch_size` | Strict positive int; initial **1** | At most one ensemble member per classifier batch; independent of total ensemble count. |
| `norm_methods` | Initially accept exactly `["none"]` | Avoid query-array-wide nonlinear fallback; see section 5. |
| `use_amp` | Strict bool; initial **false** | Establish a full-precision numerical reference before reduced-precision comparisons. |
| `use_fa3` | Initially accept only **false** | RTX 4060 is not the documented Hopper FA3 target; avoid optional kernels and implicit casts. |
| `kv_cache` | Initially accept only **false** | Avoid retained context KV memory; first establish ordinary inference and persistence. |
| `offload_mode` | Initially accept only `"cpu"` | Request CPU column-output storage to constrain VRAM; vendor cascading caveat below still applies. |
| `n_jobs` | Strict positive int; initial **4** | Bound native prediction CPU threads; record actual host/torch threading. |

Framework-controlled settings, not free-form overrides:
`random_state=seed` (42), `allow_auto_download=False`, fixed
`checkpoint_version`, `feat_shuffle_method="latin"`,
`class_shuffle_method="shift"`, `outlier_threshold=4.0`,
`softmax_temperature=0.9`, `average_logits=True`,
`support_many_classes=False`, `disk_offload_dir=None`,
`verbose=True`, and `inference_config=None`.
Reject aliases, arbitrary nested inference config, device overrides inside
stage configs, fit controls, and unknown keys.
Disabling many-class support does not affect the binary protocol.
Leave advanced `MgrConfig` defaults alone but record their resolved values.

Outer experiment choices: model `"tabicl"`, `strict_common_16`, model seed 42,
sampling seed 42, `train_sample_size=2000`, same-domain validation domain,
`early_stopping_rounds=null`, threshold 0.5, initial external
`prediction_batch_size=128`, eventual `test_domains=["source","kaggle","nhanes"]`.
Keep the existing calibration-bin convention (10 in the matched configurations).
External 128 is a proposed conservative starting point, not a measured optimum
or a value inherited from TabPFN.

The no-AMP choice is methodological, not an assertion that FP32 fits this GPU.
The inspected forward explicitly creates float32 tensors, and disabling both
AMP and FA3 avoids size-driven precision changes and FA3's possible cast.
If this profile is infeasible, a separately named **explicit AMP=true**
profile is a candidate for a later reviewed run, not an automatic fallback.
Its float16 default, numerical checks and scientific interpretation must be
recorded separately; passing FP32 does not certify AMP.

## 5. Preprocessing and query-batching policies

### Unchanged canonical inputs; train-only internal transforms

Use the base adapter's copied DataFrame with schema categorical columns cast
to strings. TabICL has no categorical-feature-index parameter: dtype conversion
is how we declare the project schema's categorical features. This also prevents
integer-coded categories from silently becoming continuous.

The native encoder groups categorical columns before numeric columns internally;
record this fitted mapping as well as the canonical input order. Never reorder
the canonical files to match its internal transformed order. Test numeric-coded
and string-valued categories and unseen strings under installed pandas 3.
Unseen strings encode as -1; there is no target-domain refit.

Retain the project's stricter rejection of empty, missing and nonfinite input
even though TabICL supports some NaN paths. No native missing-feature/SHAP
masking behavior is part of M07. Dense DataFrames are the adapter input contract;
native numeric arrays are supported by TabICL but not a new shared API surface.

Even `"none"` is **not raw/no preprocessing**: native mean imputation setup,
constant-feature filtering, custom standard scaling (including clipping),
and two-stage train-distribution outlier clipping remain. The normalizer alone
is absent. Training-only fitted statistics and ordinal vocabulary are reused
for every query; validation and test features/labels never fit transforms.
Canonical data remain unscaled, unencoded and unchanged on disk.

### Why not the native default normalizers initially?

Default normalization views are `["none","power"]`. The power path transforms
the entire query array in one call. On a `ValueError`, native code clips **all**
that array's standardized entries to training min/max and retries.
Consequently a problematic row can change another row's processing by sharing
its chunk. This is a control-flow dependency, not query-statistic fitting and
not floating-point noise. It has been source-identified, not locally reproduced.
An unrestricted default profile cannot be certified chunk-invariant here.

The bounded `["none"]` choice avoids that branch without patching vendor code,
adding a custom preprocessor, or using query statistics. Other normalizers are
outside the first allowlist and require a separately reviewed protocol.

### Expected invariances and fixed evaluation policy

For finite binary inputs with this profile, source inspection supports
query-order equivariance, independence from unrelated query rows, and external
chunk equivalence in exact arithmetic: inducing summaries use training rows,
row interaction acts within each row, and ICL queries attend only to training
keys/values. SSMax length is key length, not query count. Native eval/no-grad
is used. This is **source-supported, not a measured guarantee**; payload config,
GPU kernels and dependency behavior still require stage 3.

Do not change training context order: it is part of the frozen protocol.
Do not expose `embed_with_test=True` or any adaptation/transductive mode.
If stage 3 finds systematic query effects, stop and diagnose them; do not
classify them as rounding merely because an allclose can be made to pass.

For M08 freeze external chunks at 128 consecutive rows in existing gateway
order for each full test partition, including the last partial chunk.
Save query identities/order hashes and that chunk policy. All three test
partitions are still evaluated completely; no subsampling or class balancing.
If a future profile deliberately has query-composition effects, its fixed
ordered chunks become part of the predictor definition, predictions cannot be
compared across arbitrary batch sizes as equivalent, and changing chunk size
requires a newly named reviewed protocol before test evaluation.

### Native resource heuristics are not disabled by explicit top-level settings

`batch_size=1` does not disable `InferenceManager`: column/row leading dimensions
can still be split based on current available memory. Multi-batch execution
can halve internal batches after OOM; the single-batch fast path can instead
raise immediately. No observed code reduces context or ensemble count or
changes precision during that recovery.

`offload_mode="cpu"` configures column outputs, not all stages and not CPU
model execution. Row/ICL manager defaults retain GPU outputs. Requested
storage modes can still cascade; with no disk directory, native code may fall
back to CPU/swap under host pressure. Therefore do **not** claim strict
no-fallback, fixed scheduling, all-GPU operation, or memory limits enforced
by these flags.

Keep verbose native diagnostics and Python warnings per operation/run, including
estimated batch sizes, offload reasons and OOM reductions; persist a native
diagnostic log with an explicit reference in JSON metadata. The adapter/smoke
implementation must route these through visible repository logging and retain
them, not swallow them. Native scheduling has no stable public complete-history
API; report that limitation rather than inventing effective settings from
`get_params()`. Treat recovery/swap as a resource issue requiring review before
M08. No project-level automatic retry, precision downgrade, context reduction,
ensemble reduction, or CPU model fallback is permitted.

`n_jobs` temporarily changes process-global torch thread count in native
prediction, restoring it only on the normal path. A thin adapter `finally`
must restore the prior thread count on native failure too. Record this narrowly
coupled safety behavior and test it; do not run adapters concurrently in-process.

## 6. Persistence policy

Use the official inherited API, not a project-defined generic pickle:

```text
estimator.save(path, save_model_weights=True,
               save_training_data=True, save_kv_cache=False)
TabICLClassifier.load(path, device=explicit_device)
```

Proposed artifact name: `model.tabicl_fit`, plus `model.adapter.json` sidecar.
With initial `kv_cache=False`, this stores fitted encoders, ensemble preprocessing,
training context and labels, and embedded CPU model weights/config.
No raw validation or test data are added. Save flags and format must be recorded.
The file is **trusted executable pickle-based state**, not a safe interchange
format; load only locally produced trusted artifacts. Include an artifact hash,
package/version checks and schema/sidecar consistency checks before use.

Embedding weights makes restoration independent of the original checkpoint path
through the inspected reconstruction branch. Test this offline with the
checkpoint temporarily unavailable in an isolated test location; never rename
or remove a shared user checkpoint. An adapter load path must not accidentally
invoke its ordinary constructor's checkpoint-file precondition when restoring
a self-contained artifact. Validate sidecar provenance/version/device, then
use native load and attach the restored estimator without refitting.

Native serialization includes sklearn, torch, numpy and Python version metadata,
but compatibility checking only warns for the first three; add the explicit
TabICL version and pandas version in the project sidecar and require the
verified runtime for the initial loader. Neither cross-version compatibility nor
safe loading of untrusted files is promised.

**Device caveat:** native `load(path, device="cpu")` first unpickles, whose
`__setstate__` resolves and uses the saved device, and only then applies the
requested override. A state saved with `device="cuda:0"` can fail on a CPU-only
host before the override. Stage 2 should support and test same-device restore,
preflight the serialized device from the sidecar, and explicitly reject an
unavailable saved device. Do not advertise universal CPU portability or write
a custom unpickler. Portable CPU-first artifact creation is a separate bounded
follow-up if needed; it is not a blocker for local same-GPU M07 checks.

Keeping weights external (`save_model_weights=False`) is an official alternative,
but then load re-enters `_load_model` and depends on the saved checkpoint
path/cache and download flags. It is not the initial choice.
Cache omission/rebuilding paths need not be certified: there is no cache in
the proposed first profile. In particular, do not infer cache persistence
behavior solely from save flags; inspect and test it before allowing `"repr"`
or `"kv"` profiles.

## 7. Metadata, timing and frozen sample identities

Required adapter metadata: release, model generation, checkpoint repository,
revision, expected and actual checkpoint digest, resolved local path at fit,
artifact digest at save/load, canonical feature order, categorical declaration,
fitted transformed mapping and retained-feature mask, classes, requested and
effective ensemble counts/configurations, fitted normalizer order, preprocessing
description, validation unused, no calibration, seed, constructor settings,
resolved per-stage inference settings, device/GPU name, arithmetic flags,
torch thread counts, cache mode, persistence flags/dependencies, and diagnostic
log reference. Serialize numeric arrays explicitly; no NaN-shaped success data.

The runner already records environment, prepared-file fingerprints, sampling
identities, fit and per-domain prediction times, fixed-threshold metrics,
protected outputs and failed status. Extend environment packages for `"tabicl"`
to include `tabicl`, `torch` (plus Hub/einops/psutil where useful), not CatBoost
by falling through the existing branch. Keep the common metadata keys
`preprocessing` and `compute_settings` used by the runner.

Existing runner timings synchronize via adapter hooks. Its allocator reset
covers fit onward, not separate operation peaks; stage-3 smoke must explicitly
reset/synchronize around each operation and record separate preparation,
prediction, save and reload timings/peaks. No need to redesign all runner timing
for this stage. Distinguish allocator allocated/reserved peaks from total VRAM
and CPU/RAM measurements.

### Frozen matched n2000 identities (L)

The saved CatBoost and TabPFN training-sample files each have exactly 2,000 rows,
all `split=train`, consecutive `sample_order=0..1999`, and matching hashes within
each domain. Hashes use the existing sampler's
`domain<TAB>row_index<TAB>integer<NEWLINE>` identity serialization.

| Domain / display | Ordered identity SHA-256 | Unordered identity SHA-256 |
|---|---|---|
| `source` / Original Synthetic | `7d81147cce93cd5c56b2e66484f9f8902ae4ef1784bab270c32016e43f44d170` | `239741018e0fa7b477a8985fdf7d214c05cd43b708b28ab061946e709b4eb9a0` |
| `kaggle` / Playground S5E12 | `612fcfb9dae65a36b591f44065baef3cf629b028435737db4a5480b09469fe3b` | `d4eca71ee3d52a04818645f1eb01c33ea26eeea0f2bde38bfab8cbb8ebabd63e` |

Local reference files:

- [Original Synthetic CatBoost context](../../results/predictive/catboost_source_strict_common_16_n2000_seed42/training_sample.csv.gz)
  and [TabPFN context](../../results/predictive/tabpfn_v35_source_strict_common_16_n2000_float32_seed42/training_sample.csv.gz).
- [Playground CatBoost context](../../results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/training_sample.csv.gz)
  and [TabPFN context](../../results/predictive/tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024/training_sample.csv.gz).

Stage 3 must reproduce both ordered hashes using `select_training(...,2000,42)`
and compare identity sequences to these references. A matching row count/seed
alone is not sufficient. Preserve both display names and internal IDs.

## 8. Stage-3 smoke and feasibility protocol (predeclared)

This protocol is **not executed**. Tests use train and validation only, no
predictive metrics and no test-loader calls. Refuse existing output directories,
including failed/interrupted attempts.

### Selection and progression

1. Start a fresh process in the named environment with offline Hub access.
   Record package/source/checkpoint identities, settings, CUDA flags/device,
   host RAM/free VRAM, and diagnostic logs. Fix model/sampling seed 42.
2. For each of `source` and `kaggle`, use `strict_common_16`, load only that
   domain's train/val through the existing gateway, require identity disjointness.
   Select **512 training rows** using the existing sampler and seed 42.
   This is a small smoke context, not the matched research context; 512 is
   above the README's 300-row lower pretraining range, unlike a 64-row smoke.
   Select **65 validation queries**, stably sorted by authoritative `row_index`,
   taking the first 65 without looking at labels.
3. Fit once with train only. Preserve deep input copies/dtypes/indexes; verify
   nonmutation. Verify correct rejection of reordered/extra/missing feature
   columns, missing and nonfinite values in separate controlled checks.
   Use small hand-built finite query fixtures for unseen string categories
   without modifying canonical partitions.
4. Predict the identical 65 queries three times; compare with one unchunked
   call, chunks **1, 17, 32, 128** (including partial chunks), a fixed RNG-42
   query permutation restored to original order, and the same sentinel rows
   alone and amid other validation queries. Do not refit between comparisons.
   Change only query arrangement, never training context/order or model settings.
5. Save native fitted state plus sidecar, release the live estimator, reload
   in a fresh offline process on the same GPU, and repeat the fixed prediction.
   Prove embedded-weight restore does not need a checkpoint using an isolated
   checkpoint copy/path, not by altering a shared checkpoint.
6. Escalate to **exact n2000 context for BOTH domains**, checking the hashes
   above before model work. Use the first **257 identity-sorted validation
   queries** so the intended 128-row path exercises 128+128+1. Run numerical,
   persistence and resource checks again; small-context success is insufficient.
7. For resource screening only, extend same-domain validation queries to
   **2,049** in the same identity order with fixed 128-row chunks. One full
   warm-up, then three repetitions. Save predictions for numerical comparisons,
   not scores. An optional separately named comparison of query batches
   64/128/256 may assess throughput if the fixed baseline passes; do not
   silently select a new M08 batch from timing or change the baseline artifact.

For each operation, synchronize immediately before and after the timer, reset
allocator peaks beforehand, and record elapsed wall time, max allocated and
reserved GPU bytes, available device memory and process RSS/host memory.
Report cold preparation separately from cold/warm prediction. Native model
loading belongs to preparation; without KV cache, substantial context computation
recurs during prediction. Serialize operation status before/after work, retaining
exceptions and native memory/OOM diagnostics on failure.

### Numerical acceptance and interpretation

These are pre-run engineering gates, not measured error bounds. FP32 machine
epsilon is about 1.19e-7. Use **absolute error <= 64 * epsilon32
(7.62939453125e-6), rtol=0** for source-supported FP32 equivalences, including
reload. This allows several ulps accumulated through deep inference without
relative tolerance becoming permissive near zero; no theorem guarantees this
budget. It is independent of the TabPFN test constants. Do not enlarge it
after seeing outcomes. A violation blocks certification pending diagnosis.

| Check | Required outcome / pre-run rule | Equivalence status before execution |
|---|---|---|
| Frozen n2000 identity/order, schema and settings | Exact identity hashes/order and requested settings, no tolerance | P contract; saved hashes L |
| Input nonmutation | Exact DataFrame/Series equality including dtype/index after fit/predict | P contract, supported by copied inputs; runtime U |
| Binary outputs | Native shape `(n,2)`, finite real values in [0,1]; adapter `(n,)`, class-aware P(y=1) | V API + P validation; runtime U |
| Probability row sums | Absolute deviation <= `32 * epsilon32` (3.814697265625e-6); reject otherwise before shared validation | V explicit normalization; numerical bound P, runtime U |
| Positive-class selection | Mock reversed classes `[1,0]` selects column 0 exactly; real classes expected `[0,1]`; threshold ties use `>=0.5` | P exact mapping, V LabelEncoder; runtime U |
| Repeated identical inference | Three runs within FP32 absolute budget; report bitwise equality separately | V eval/no-grad, not a determinism guarantee; package advertises nondeterministic tag |
| Query permutation, restored order | Within same FP32 budget; no row-count/order loss | V source-supported for chosen path; runtime U |
| Chunk/sentinel composition | Within same budget; record max/mean difference, per-row residuals and threshold flips | V source-supported only for `["none"]` finite path; runtime U |
| Save/reload same GPU, weights embedded | Within same budget and no network/checkpoint access | V reconstruction path; runtime U |
| Cross-device or cross-version equality | Not certified by this protocol; never reuse same-device passing result as proof | U / explicit limitation |
| Other normalizers / AMP / caching | Not certified; need a separate predeclared profile and checks | U; nonlinear fallback has known control-flow risk |

For comparisons also require no threshold disagreement when both probabilities
are farther than the FP32 budget from 0.5. Count and save all flips; those inside
the boundary band still require explicit reporting and review before M08, not
concealment behind allclose. An equivalence pass alone cannot overrule evidence
of systematic composition-dependent preprocessing. No accuracy/AUC-based
configuration selection is permitted here.

If an explicit AMP profile is later authorized, do not apply the FP32 gate
automatically or treat AMP-vs-FP32 differences as an identity check. Predeclare
its own numerical validation before running it; retain the FP32 reference and
report query effects and threshold sensitivity. This document does not certify
an AMP tolerance in advance of that profile's design.

### Resource acceptance and outputs

Proposed operational gate: both n2000 contexts complete preparation,
fixed-query inference, persistence/reload and three 2,049-query repetitions
without project fallback, native OOM recovery, swap-tier fallback, or nonfinite
outputs. Reserve at least **512 MiB** allocator headroom relative to device
total, and record concurrent non-PyTorch usage; passing allocator peaks alone
does not guarantee safe total usage. If this gate fails, retain the failure and
review a separately named profile; never reduce the required n2000 context
while calling the matched-budget requirement satisfied.

Report median/individual times and throughput. Extrapolate full-test runtimes
only after measurements, clearly as estimates: no runtime/SLA threshold is
claimed here and no test files need be opened to run this screen.

New directories, for example:
`results/smoke/tabicl_v2_source_n512_fp32_none_run01`,
`results/smoke/tabicl_v2_kaggle_n512_fp32_none_run01`,
and corresponding `n2000` directories; any optional timing comparison goes under
`results/benchmarks/tabicl_v2_<domain>_n2000_fp32_none_run01`.

Save config, metadata/status, environment, checkpoint/artifact hashes,
`training_sample.csv.gz`, ordered validation-query identities and hash, raw
probability arrays per check, equivalence statistics and threshold diagnostics,
native fitted artifact/sidecar, per-operation timings/memory, warnings and
native logs. Distinguish execution success, numerical eligibility, resource
eligibility and overall readiness. Any failed mandatory gate exits nonzero and
persists operation/type/message/traceback/settings/memory context. No
success-shaped fallback arrays or broad exception suppression.

## 9. Published limits, risks and remaining checks

The [versioned README][readme] recommends typical use with 300-100,000 training
rows and up to 2,000 features, describes pretraining on 300-48K rows and 2-100
columns, and reports larger-data scaling using CPU/disk offloading. Its
50K-by-100 under-10-second example is on an H100, not an RTX laptop.
The README gives complexity `O(n^2 + n*m^2)`. These are author-reported scope
and performance claims, not hard API limits or an 8-GB memory prediction.
Our 2,000-by-16 problem lies inside the reported pretraining ranges.
The roughly 110-MB checkpoint says little about activations, workspace or
retained context memory. No stage-1 inference estimate is presented as measured.

Outstanding gates, in order:

1. **Before real-model work:** authorized download or discovery of the exact
   checkpoint, local SHA/size verification, then payload configuration validation
   during stage-3 fit. Confirm binary capacity and actual architectural settings.
2. **During stage 2:** implement strict configuration, offline guards and optional
   imports, native diagnostics capture, train-only behavior and same-device
   native persistence; unit-test mocked success and every explicit failure path.
   Test thread restoration even when native prediction fails.
3. **During stage 3:** verify pandas 3 string conversion/ordinal behavior,
   constant-feature handling/effective ensemble, real output shape, numerical
   equivalences, embedded-weight offline restoration, native scheduling and
   n2000 feasibility for both domains. No existing TabPFN result answers these.
4. **Before stage 4/M08:** approve the named none-normalization FP32 profile
   based on feasibility, freeze batch/arithmetic/context/settings, and document
   departure from official default normalizer views. If a different profile is
   necessary, repeat the relevant gates rather than silently broadening allowlists.
5. CPU-only portability, optional cache modes, default power-normalization
   equivalence and alternate precision remain explicitly outside certification.
   They are not prerequisites for same-GPU bounded implementation.

### Exact proposed setup commands for later use -- NOT executed

From the repository root, use the named environment explicitly:

```bash
conda run -n synthetic-tabular-ai python -c \
  'import sys, importlib.metadata as m; print(sys.executable); print(m.version("tabicl"))'
conda run -n synthetic-tabular-ai python -m pip check
```

For a recreated environment **only after the dependency manifests are amended
in a later stage**, if TabICL is absent and the already pinned dependencies
remain satisfied:

```bash
conda run -n synthetic-tabular-ai python -m pip install --no-deps 'tabicl==2.2.0'
conda run -n synthetic-tabular-ai python -m pip check
```

No install is needed in the inspected environment. `--no-deps` is appropriate
only with those dependency checks; a missing requirement must be explicitly
resolved, not ignored. Do not run `pip install -U`, install extras, or replace
the CUDA build to add this adapter.

The following **authorized later acquisition** uses existing huggingface-hub,
fixed revision and file only. It deliberately enables network access for this
one setup operation; adapter/smoke operation remains offline.

```bash
env HF_HUB_OFFLINE=0 conda run --no-capture-output -n synthetic-tabular-ai python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
revision = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
destination = Path("checkpoints/tabicl") / revision
path = hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
    revision=revision,
    local_dir=destination,
    token=False,
)
print(path)
PY
printf '%s  %s\n' \
  'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0' \
  'checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt' \
  | sha256sum --check --strict
stat --format='%s' checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt
```

Expected size: 110368038. Download API behavior is source/signature-based, not
an executed download verification. Do not proceed after a digest/size failure.
Checkpoint paths are already covered by [gitignore](../../.gitignore); no weight
or fitted-state artifact should be committed.

After stage-2 implementation, proposed validation commands:

```bash
HF_HUB_OFFLINE=1 conda run -n synthetic-tabular-ai python -m unittest \
  discover -s tests -p 'test_tabicl*.py' -v
HF_HUB_OFFLINE=1 conda run -n synthetic-tabular-ai python -m \
  src.experiments.smoke_tabicl --help
```

`test_tabicl*.py` and `src.experiments.smoke_tabicl` are **proposed new surfaces,
not currently runnable commands**. The intended smoke CLI should accept
`--config`, `--output-dir`, `--sample-size`, `--query-size`, `--batch-size` and
an explicit mode for the equivalence/resource suite above. Stage-3 invocations
are contingent on that implementation and temporary smoke configs; stage-4
research configs do not yet exist. Use `HF_HUB_OFFLINE=1` plus the adapter's
`allow_auto_download=False` and explicit local path/hash guard. No generic
setup script or full runner command should be used as a substitute.

## 10. Stage-2 handoff

Implement the bounded adapter, config/factory/environment hooks, official
self-contained same-device persistence, diagnostic/resource logging and focused
mock-based integration tests. Prepare the train/validation-only smoke surface
without running it. Leave canonical data, sampling, existing models/results,
reports and checklist completion untouched.

There is no unresolved API fact blocking that implementation. There **are**
unresolved checkpoint-availability, payload, numerical and resource facts
blocking certification and M08. Stage 3 must establish them for both frozen
contexts before stage 4 finalizes configuration/documentation or prepares a PR.

## 11. Stage-2 implementation record

The sections above preserve the stage-1 inspection/design. Stage 2 now adds
[the bounded adapter](../../src/models/tabicl.py),
[mocked adapter/integration tests](../../tests/test_tabicl_adapter.py),
[the deferred smoke command](../../src/experiments/smoke_tabicl.py) and
[mocked smoke tests](../../tests/test_tabicl_smoke.py), with
[explicit none/FP32 smoke configurations](../../configs/smoke).
The [README](../guides/tabicl_usage.md)
documents current commands and limitations. These command/test surfaces now
exist; their hypothetical status in section 9 is historical stage-1 context.

No real TabICL estimator, checkpoint payload, fit or prediction was executed
in stage 2; all integration execution uses fake native estimators/runtime.
No packages were installed/upgraded and no weights downloaded.
Dependency pin changes and final M08 configurations remain stage-4 work.
The none-normalization choice is an explicitly named experimental profile,
not a claim that TabICL universally requires it.

The smoke command supports `--suite equivalence` (n512 by default) and
`--suite feasibility` (n2000, 128-row query chunks, 2049-query timing screen).
The fixed tolerance multipliers are engineering acceptance criteria, not
mathematically guaranteed error bounds. Threshold flips require review even
within the tolerance band. Native diagnostics distinguish detected recovery
from **no recovery detected**, never from a claim of invariant scheduling.
The command records fresh-process reload measurements separately and is designed
to check checkpoint-independent restore by removing only an isolated temporary
copy before restore. Same-device restoration is the implemented scope; a real
round trip has not yet been run.

Stage-2 validation passed 89 targeted tests: 27 new mocked TabICL adapter/smoke
tests plus 62 existing sampling, TabPFN smoke/benchmark and shared-framework
configuration/metrics/mocked-runner tests. Real-model adapter tests were
deliberately excluded; this is not a full-suite claim. Python 3.11 parsing,
Pylance syntax checks for the two new implementation modules, CLI help without
optional model imports, documentation links and whitespace checks also passed.
The installed 2.2.0 runtime passed constructor/fit/predict/save/load signature
inspection without estimator construction.

At the end of stage 2, M07 and M08 remained unchecked. Actual checkpoint SHA/size/payload verification,
real-model numerical behavior, persistence round trip and feasibility for
**both** frozen n2000 contexts remained stage-3 gates.

## 12. Final acceptance and stage-4 handoff

Stage 3 resolved the checkpoint, bounded numerical, same-device persistence,
and two-context resource unknowns; details and limitations are retained in
[the results document](../results/tabicl_stage3_results.md). The saved reports were
rechecked during finalization, not inferred solely from earlier summaries.
No inference, preprocessing, persistence or smoke-acceptance behavior changed
after those runs. Only adapter metadata wording was updated to reference the
bounded evidence without certifying a future run's arbitrary context/settings.
Therefore the completed real-model suites were not rerun.

The final [Original Synthetic](../../configs/tabicl_v2_source_n2000_none_fp32.json)
and [Playground S5E12](../../configs/tabicl_v2_kaggle_n2000_none_fp32.json) research
configurations preserve the actual n2000 smoke configuration except for
experiment/output names. Both load through the real loader, resolve to distinct
unused predictive output directories, and reproduce the exact frozen training
sequences in a train-only sampling check. No research runner was invoked.
Both dependency declarations add only `tabicl==2.2.0`; all existing pins,
including PyTorch/CUDA, remain unchanged.

Final validation in `synthetic-tabular-ai`: **28 focused TabICL tests and
115 full-suite tests passed**, including research-profile preservation and
fresh/output-overwrite isolation. `pip check`, dependency parsing and pin
preservation, smoke CLI help, train-only identity validation, documentation
links/fences, and whitespace checks passed. Real-model smoke results are
separate from mocked integration tests; neither is full predictive evaluation.

Ryan Dozier implemented/performed and reviewed this work with AI assistance;
review is self-review, not an independent approval. PR pending; no commit,
push, PR creation, merge, or M08 evaluation is part of this acceptance.
See the [checklist contribution record](../../PROJECT_CHECKLIST.md#contribution-log).

[tag]: https://api.github.com/repos/soda-inria/tabicl/git/ref/tags/v2.2.0
[pyproject]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/pyproject.toml
[readme]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/README.md
[license]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/LICENSE
[classifier]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_sklearn/classifier.py
[base]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_sklearn/base.py
[preprocessing]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_sklearn/preprocessing.py
[embedding]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/embedding.py
[layers]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/layers.py
[interaction]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/interaction.py
[learning]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/learning.py
[encoders]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/encoders.py
[attention]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/attention.py
[inference]: https://github.com/soda-inria/tabicl/blob/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3/src/tabicl/_model/inference.py
[checkpoint-api]: https://huggingface.co/api/models/jingang/TabICL/revision/4dcd344ece2c00be9e831fdd35bed57b5ad83e19?blobs=true
[card]: https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/README.md
