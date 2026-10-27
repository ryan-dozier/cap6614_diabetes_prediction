# FT-Transformer Stage 3 functional, resource, and feasibility results

Date: 2026-10-26

## Outcome and scope

**Stage 3B passed** for both Original Synthetic (`source`) and Playground
S5E12 (`kaggle`). These were functional compatibility checks on the actual
project training and same-domain validation partitions. They do not measure
research-profile feasibility or model quality relative to other models.

Only the configured domain's `train` and `val` partitions were loaded by
the smoke workflow. No test partition or NHANES data was loaded. Neither
resource nor feasibility mode was run during Stage 3B. M09 and M10 remain incomplete.

## Environment

| Item | Observed |
|---|---|
| Python executable | `/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python` |
| Python | 3.11.17 |
| `rtdl-revisiting-models` | 0.0.2 |
| PyTorch | 2.14.1+cu126; CUDA runtime 12.6 |
| NumPy / pandas / scikit-learn | 2.4.6 / 3.0.6 / 1.9.1 |
| CUDA device | `cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU |

No packages were installed or upgraded.

## Commands and output directories

Both proposed directories were absent before the runs. Commands were run
sequentially with an explicit CUDA device:

```bash
/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_source.json \
  --output-dir results/smoke/ft_transformer_source_functional_cuda \
  --mode functional \
  --device cuda:0

/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_kaggle.json \
  --output-dir results/smoke/ft_transformer_kaggle_functional_cuda \
  --mode functional \
  --device cuda:0
```

Artifacts:

- `results/smoke/ft_transformer_source_functional_cuda/`
- `results/smoke/ft_transformer_kaggle_functional_cuda/`

Each directory contains `smoke.json`, requested/effective configuration
JSON, compressed training/validation/query identities,
`reference_probabilities.npy`, and `model.ft_transformer.pt`.

## Requested and effective profile

The requested JSON files preserve the approved research profile: three
blocks, width 192, eight heads, and a 200-epoch cap. Functional mode
resolved that to the pre-approved small diagnostic profile below; the
explicit `--device cuda:0` overrode the mode's CPU default. The stored
requested/effective JSON and report agree.

| Setting | Effective functional value |
|---|---|
| Mode / device / precision | functional / explicit `cuda:0` / float32 |
| Architecture | 1 block, width 16, 4 attention heads |
| Feed-forward | hidden width 32, ReGLU |
| Attention / feed-forward / residual dropout | 0 / 0 / 0 |
| Optimizer settings | learning rate 0.01; weight decay 0 |
| Training | batch 16; maximum 3 epochs; seed 42 |
| Validation | 32 rows, evaluated every epoch, batch 32 |
| Validation selection | highest ROC-AUC; earliest exact tie |
| Prediction comparisons | 32 query rows; batch sizes 3, 7, and 16 |

No model settings, data, split assignments, or tolerances were changed.

## Populations and identity verification

| Domain | Train rows (class 0 / 1) | Validation rows (class 0 / 1) | Selected epoch / executed | Stop |
|---|---:|---:|---:|---|
| Original Synthetic | 64 (26 / 38) | 32 (13 / 19) | 1 / 3 | `max_epochs` |
| Playground S5E12 | 64 (24 / 40) | 32 (12 / 20) | 1 / 3 | `max_epochs` |

Both populations include both classes. For each run, the selected training
and validation identity files have the recorded row counts and match the
reported unordered and ordered SHA-256 fingerprints. Their identities are
disjoint. The validation query identity file has 32 rows, matches the
reported query fingerprint, and is a subset of the selected validation
identities. The query sample is the complete 32-row functional validation
population in both runs.

| Domain / population | Unordered identity SHA-256 | Ordered identity SHA-256 |
|---|---|---|
| Source training | `6a18ad8a9a83eafda6a6a2265ba45721a240c2b06a22f71bde53f568eec2233c` | `8a94bec86de1b61c3ae191db38501d6a91acfff30deeef10acb053d3c0ff806b` |
| Source validation and query | `565a2912ef6f4eab69a86afad0cee7affeace977912db70b0be63a1d16091c69` | `0b2401c6407bc6ce1b2a6040f5079fe39a3598732b3f89d21580d8f4aaae8650` |
| Kaggle training | `117ee9c7d976d2ddda7f7d4f95af358572b4235e95c4c63f17181ec22f663b02` | `5530e7730e4a37f62517051738ec800fc52d5f7571bfa49a1cf049b7c18cb6c4` |
| Kaggle validation and query | `b19696ca317c25a29302bbc06d4cab82f5f54799004cfb621c939c88ba84ff8a` | `eedf312ff6aa1acbe025ab2e74fb75b475e33b4731ce8f2e228375103037b495` |

Both reports identify only their configured train/validation domain; the
identity artifacts contain no test or NHANES partition.

## Training history and checkpoint selection

All reported training loss, validation loss, ROC-AUC, and durations were
finite. The selected checkpoint in each run is epoch 1, the earliest epoch
with maximum validation ROC-AUC in the recorded history. Both ran the full
three-epoch functional budget, so `max_epochs` is the consistent stopping
reason.

| Domain | Epoch | Train loss | Validation loss | Validation ROC-AUC |
|---|---:|---:|---:|---:|
| Source | 1 | 0.722403 | 0.682736 | 0.676113 |
| Source | 2 | 0.686151 | 0.679429 | 0.295547 |
| Source | 3 | 0.679421 | 0.675403 | 0.615385 |
| Kaggle | 1 | 0.721279 | 0.663782 | 0.783333 |
| Kaggle | 2 | 0.675535 | 0.661625 | 0.762500 |
| Kaggle | 3 | 0.662991 | 0.658946 | 0.750000 |

Validation unseen-category counts were valid per-feature counts, all zero:

| Domain | `ethnicity` | `gender` | `smoking_status` |
|---|---:|---:|---:|
| Source | 0 | 0 | 0 |
| Kaggle | 0 | 0 | 0 |

## Prediction and persistence checks

All checks passed with the configured `rtol=1e-5`, `atol=1e-6`, and zero
threshold-decision disagreements. Reference probabilities were float64
arrays of shape `(32,)`, finite and in `[0, 1]`.

| Check | Source max absolute probability difference | Kaggle max absolute probability difference | Decision disagreements |
|---|---:|---:|---:|
| Reference prediction | 0 | 0 | 0 |
| Repeated prediction | 0 | 0 | 0 |
| Permuted queries, restored order | 2.22843e-8 | 1.45164e-8 | 0 |
| Prediction batch size 3 (partial final batch) | 2.22843e-8 | 2.18362e-8 | 0 |
| Prediction batch size 7 (partial final batch) | 0 | 0 | 0 |
| Prediction batch size 16 | 1.48548e-8 | 2.17698e-8 | 0 |
| Save/reload | 0 | 0 | 0 |
| Query input unchanged | Passed | Passed | Not applicable |

The saved artifact was inspected with restricted tensor loading. Its
format/version, device and precision, feature schema/order, preprocessing
metadata, and finite model tensors were verified. Saved preprocessing
matched both the fitted report metadata and reloaded adapter metadata
exactly, including ordered numerical features and means/scales, categorical
vocabularies/cardinalities, fallback values/indices, and the
`training_mode` unknown-category policy.

The common feature order was:

`age`, `alcohol_consumption_per_week`, `bmi`, `cardiovascular_history`,
`cholesterol_total`, `diastolic_bp`, `ethnicity`, `family_history_diabetes`,
`gender`, `hdl_cholesterol`, `heart_rate`, `hypertension_history`,
`physical_activity_minutes_per_week`, `sleep_hours_per_day`,
`smoking_status`, `systolic_bp`.

The persisted train-fitted numerical means and scales (in the numerical
feature order above) were:

| Numerical feature | Source mean | Source scale | Kaggle mean | Kaggle scale |
|---|---:|---:|---:|---:|
| `age` | 48.3125 | 14.4404758837789 | 52.203125 | 11.959802056655244 |
| `alcohol_consumption_per_week` | 1.875 | 1.2562344526401112 | 1.921875 | 0.9731502886887513 |
| `bmi` | 25.0265625 | 3.0847529372048177 | 26.0421875 | 2.4529831766328423 |
| `cardiovascular_history` | 0.09375 | 0.2914805954090255 | 0.03125 | 0.17399263633843817 |
| `cholesterol_total` | 179.546875 | 30.438939743926284 | 186.5 | 16.705163872288114 |
| `diastolic_bp` | 74.515625 | 8.179609150770897 | 76.765625 | 7.324492703209895 |
| `family_history_diabetes` | 0.203125 | 0.40232478717449166 | 0.15625 | 0.3630921887069453 |
| `hdl_cholesterol` | 55.578125 | 7.909260173010811 | 53.703125 | 6.025154789246082 |
| `heart_rate` | 70.28125 | 8.49534569264253 | 69.78125 | 6.450457227011121 |
| `hypertension_history` | 0.25 | 0.4330127018922193 | 0.21875 | 0.41339864235384227 |
| `physical_activity_minutes_per_week` | 133.03125 | 97.51682046415121 | 79.28125 | 54.88182439057124 |
| `sleep_hours_per_day` | 6.8765625 | 1.0708848834462787 | 6.965625 | 0.8851586633903551 |
| `systolic_bp` | 114.21875 | 14.235155371034768 | 117.59375 | 12.028246378317165 |

Categorical mappings found in the artifacts:

| Feature | Source vocabulary (cardinality; fallback value/index) | Kaggle vocabulary (cardinality; fallback value/index) |
|---|---|---|
| `ethnicity` | Asian, Black, Hispanic, Other, White (5; White / 4) | Asian, Black, Hispanic, Other, White (5; White / 4) |
| `gender` | Female, Male, Other (3; Male / 1) | Female, Male (2; Male / 1) |
| `smoking_status` | Current, Former, Never (3; Never / 2) | Current, Former, Never (3; Never / 2) |

## Timing and memory

Durations are wall-clock operation timings from `smoke.json`. Fit includes
preprocessing, training, validation, checkpoint selection/restoration, and
other fit overhead. GPU memory below is the fit operation's PyTorch
allocator peak after the adapter's explicit pre-operation reset, not a
later prediction/save/reload peak or total process/GPU memory.

| Domain | Total run | Fit | Save | Reload | Fit peak allocated | Fit peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| Source | 24.9140 s | 15.0335 s | 0.0338 s | 0.0602 s | 17,542,656 bytes | 23,068,672 bytes |
| Kaggle | 28.6118 s | 14.2931 s | 0.0321 s | 0.0476 s | 17,542,656 bytes | 23,068,672 bytes |

The fit reports also recorded 7,384,072,192 bytes available at the time of
the query and 8,585,216,000 total bytes on the device. Available memory is
only a point-in-time observation. These tiny-profile measurements do not
estimate n2000/full-validation research runtime or memory; that is Stage 3C.

## Limitations and decision

No smoke run failed. The smoke results establish functional project-data
compatibility for these small selected populations, not convergence,
reproducibility across hardware, predictive superiority, or full-profile
resource feasibility. **Stage 3B passed; proceed to Stage 3C's two-epoch
resource estimates only after reviewing these recorded costs.** Do not
launch the resource estimates as part of this Stage 3B record.

## Stage 3C: approved-profile resource measurements

Date: 2026-10-26 local time (2026-10-27 UTC).

**Stage 3C passed for both domains. Recommendation: proceed locally with
the approved profile for Stage 3D, in a separate execution.** No feasibility
mode or research/test evaluation was launched. No profile revision is
recommended. M09 and M10 remain incomplete.

### Execution boundary, environment, and commands

Branch: `feat/ft-transformer-adapter`. The initial worktree already contained
a modified integration plan and untracked smoke configs, workflow, tests,
and this results document. These changes and the completed Stage 3B
artifacts were preserved. Only this document and the integration plan were
edited for Stage 3C; no source/config/dependency changes, installs, upgrades,
commits, or pushes were performed.

The verified executable was
`/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python`, not the
shell's active `python-3.14` environment. Runtime inspection reconfirmed
Python 3.11.17, `rtdl-revisiting-models==0.0.2`,
`torch==2.14.1+cu126`, CUDA runtime 12.6, NumPy 2.4.6, pandas 3.0.6,
and scikit-learn 1.9.1. CUDA was available with one device:
`cuda:0`, NVIDIA GeForce RTX 4060 Laptop GPU. Initial Torch free/total
memory was 7,440,695,296 / 8,585,216,000 bytes. Initial `nvidia-smi`
reported 0 MiB used of 8,188 MiB and 0% utilization; no compute processes
were reported. Checks between and after runs also reported no compute
processes and 0% utilization. No competing GPU work was observed at these
sampled times; this was not continuous process monitoring. No other process
was terminated.

Both proposed output directories were absent. Source ran first and its
completed report, populations, baseline matches, and persistence were
reviewed before Kaggle started. Exact commands:

```bash
/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_source.json \
  --output-dir results/smoke/ft_transformer_source_resource_cuda_e2 \
  --mode resource --resource-epochs 2 --device cuda:0

/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_kaggle.json \
  --output-dir results/smoke/ft_transformer_kaggle_resource_cuda_e2 \
  --mode resource --resource-epochs 2 --device cuda:0
```

Actual artifact directories:

- [Source resource run](../../results/smoke/ft_transformer_source_resource_cuda_e2)
- [Kaggle resource run](../../results/smoke/ft_transformer_kaggle_resource_cuda_e2)

Each contains `smoke.json`, `requested_config.json`,
`effective_config.json`, `training_identities.csv.gz`,
`training_sample.csv.gz`, `validation_identities.csv.gz`,
`validation_query_identities.csv.gz`, `reference_probabilities.npy`, and
`model.ft_transformer.pt`. Only each configured domain's training and
same-domain validation were loaded. Independent population verification
reloaded only that domain's validation partition. Configured `test_domains`
remain inert in this workflow; no test partition, baseline test prediction,
test metric, or NHANES data was inspected.

### Requested versus effective profile

Both requested configurations retained the approved 200-epoch research
profile. Resource mode deliberately changed `max_epochs` to 2; the other
configuration difference is the fresh output directory. The explicit device
override reaffirmed the requested `cuda:0`. Stored configuration files,
report configurations, mode details, and saved/reloaded model parameters
were checked against independently resolved configurations.

| Setting | Effective value for both resource runs |
|---|---|
| Mode / training budget | `resource`; 2 epochs; truncation explicitly true |
| Training / sampling seeds and rows | 42 / 42; exact existing n2000 sample |
| Schema | `strict_common_16`; ordered schema unchanged |
| Architecture | 3 transformer blocks, width 192, 8 attention heads |
| Feed-forward | ReGLU; multiplier 4/3, hidden width 256 |
| Attention / FFN / residual dropout | 0.2 / 0.1 / 0.0 |
| Loss / weighting | BCEWithLogitsLoss; no class weighting |
| Optimizer | AdamW, upstream `model.make_parameter_groups()`; learning rate 0.0001, weight decay 0.00001 |
| Parameter-group policy | Feature/CLS embeddings, normalization parameters, and biases excluded from weight decay |
| Training / validation batches | 256 / 1024 |
| Validation membership / frequency | Complete fixed same-domain population, every epoch |
| Precision / device | Float32, explicit `cuda:0`; no mixed precision or fallback |
| Preprocessing | Train-only StandardScaler; sorted categorical vocabularies, unseen categories alias training mode |
| Selection / patience | Highest raw finite validation ROC-AUC, earliest exact tie; patience 20, `min_delta=0.001` affects patience only |
| Prediction settings | Configured batch 256; bounded numerical queries use batches 3, 7, 16; threshold 0.5 |

Optimizer construction and selection semantics were verified in the
existing adapter; resource mode does not override them. No architecture,
precision, preprocessing, optimizer, batch size, membership, tolerance, or
stopping-policy change was made.

### Population and matched-sampling verification

| Domain | Available training | Selected training (class 0 / 1) | Full validation (class 0 / 1) |
|---|---:|---:|---:|
| Source | 70,000 | 2,000 (800 / 1,200) | 15,000 (6,000 / 9,000) |
| Kaggle | 490,000 | 2,000 (753 / 1,247) | 105,000 (39,554 / 65,446) |

Both classes were present. Every identity file was checked for row count,
unique identities, contiguous sample order, correct domain/split, and its
recorded unordered and ordered fingerprints. Training and validation
identities were disjoint. The validation identity sequence, summary,
fingerprints, and class counts exactly matched an independently loaded
complete fixed validation partition, not just its expected size.

For each domain, `training_sample.csv.gz` was compared as an entire parsed
DataFrame, including row order, against all three corresponding existing
n2000 baselines: CatBoost, TabPFN float32 (Kaggle batch1024), and TabICL v2
none/fp32/uncached. All six comparisons matched exactly. Their metadata's
sampling counts, class counts, unordered hashes, and ordered hashes also
matched. Only baseline training-sample files and the sampling fields of
metadata were used; no baseline predictions or test metrics were read.

| Domain / population | Unordered identity SHA-256 | Ordered identity SHA-256 |
|---|---|---|
| Source training | `239741018e0fa7b477a8985fdf7d214c05cd43b708b28ab061946e709b4eb9a0` | `7d81147cce93cd5c56b2e66484f9f8902ae4ef1784bab270c32016e43f44d170` |
| Source validation | `b2bd2e47a46738932e9738fcd38cec4af4c10c81b5045ae04eb15dfbc8009033` | `9c8a79f403559c527c5fb0ab8089f8f12c9c64151fd38f18dd9fa0c71af06417` |
| Kaggle training | `d4eca71ee3d52a04818645f1eb01c33ea26eeea0f2bde38bfab8cbb8ebabd63e` | `612fcfb9dae65a36b591f44065baef3cf629b028435737db4a5480b09469fe3b` |
| Kaggle validation | `531c3659e3965ea609266736bb9ff126ac988cd5ef1fb5aa9f6a0098b5b146c6` | `92d1dafc260aa5d25705beb500327a6ccfc95963af34ea05ba933c6ae0851f6b` |
| Source query | `eefc4bbcab384eb5feabee0ed3d277f9fc9f4b974e817b645f6fbbae0caf69e2` | `8f233003147e083bb66729dab22d12550fbf222f9ffb5d29ed85994e74815508` |
| Kaggle query | `4d5a8d66c1ad132e5020049d92c9da9a1ff4f652354287f21bce9a74608e4cad` | `aa3df47ea6af17f8ea21cb8f4a59e4ba6abd7edc97a01061c7571991bc5b8918` |

Each bounded prediction query had 32 rows (13 class 0, 19 class 1).
Query identities matched their recorded validation gateway positions and
were members of the complete validation population. Query bounds apply
only to numerical/persistence diagnostics, never epoch validation.

### Numerical, history, and persistence outcomes

All recorded checks passed at unchanged `rtol=1e-5`, `atol=1e-6`.
Every probability comparison had zero threshold disagreements at 0.5.

| Check | Source maximum absolute difference | Kaggle maximum absolute difference |
|---|---:|---:|
| Reference prediction | 0 | 0 |
| Repeated prediction, batch 7 | 0 | 0 |
| Query permutation, restored order, batch 7 | 0 | 0 |
| Prediction batch 3 (final batch 2) | 7.998305462e-8 | 5.697576622e-8 |
| Prediction batch 7 (final batch 4) | 0 | 0 |
| Prediction batch 16 (two complete batches) | 5.634272504e-8 | 8.391024653e-8 |
| Same-device save/reload | 0 | 0 |
| Query input nonmutation | Passed | Passed |

Reference arrays were finite float64 `(32,)` probabilities in `[0,1]`.
All recorded history fields (losses, validation ROC-AUC, and durations)
were finite. Both runs executed epochs 1 and 2, selected epoch 2 (the
earliest maximum recorded validation ROC-AUC), and stopped by `max_epochs`.
Both output statuses were `completed`, while
`training_budget_truncated=true` and training `truncated=true` remained
explicit. This is diagnostic completion, not convergence or Stage 3D
completion. Two-epoch predictive scores are not model-quality findings.

Restricted `torch.load(..., map_location="cpu", weights_only=True)`
inspection verified artifact format version 1, finite state tensors,
float32, saved `cuda:0`, approved parameters, and training metadata.
Persisted preprocessing exactly matched fitted and reloaded metadata:
ordered numerical features, means/scales, categorical vocabularies and
counts, cardinalities, fallback values/indices, and `training_mode` policy.
Saved, fitted, reloaded, and authoritative feature orders/schema agreed.
No preprocessing refit occurred on reload. Full-validation unseen counts
were zero for all three categorical features in each domain.

### Operation timing and scoped GPU memory

Times below are seconds from the workflow's synchronized operation timers.
Its `total_duration_seconds` is the recorded elapsed run timer, not a
separately timed shell/process launch. UTC start/end timestamp spans were
21.9480 s (source) and 28.6138 s (Kaggle); they differ from the
`perf_counter` totals below and are not substituted into fit projections.

| Measurement | Source | Kaggle |
|---|---:|---:|
| Total workflow elapsed | 20.3845 | 26.9247 |
| Fit | 12.6781 | 11.9596 |
| Epoch 1 training | 6.6091 | 4.1057 |
| Epoch 1 full validation | 0.2896 | 1.9867 |
| Epoch 2 training | 0.5155 | 0.5501 |
| Epoch 2 full validation | 0.2747 | 1.8545 |
| Combined fit overhead | 4.9892 | 3.4626 |
| Save | 0.0959 | 0.0789 |
| Reload | 0.0964 | 0.1232 |

Combined fit overhead retains its existing definition: fit duration minus
the sum of epoch training and validation durations. It includes construction,
checkpoint copies/restoration, and other overhead; it is **not pure
preprocessing time**. First-epoch training startup is inside the epoch 1
training measurement, not this overhead. Outside-fit workflow costs were
7.7064 s / 14.9651 s, including partition loading, identity persistence,
bounded predictions, saving/reloading, and reporting.

The next table uses `operations.<name>.resources.peak_gpu_memory`, captured
after an explicit reset for that operation. Peak byte values were identical
between domains; durations differed.

| Operation | Source seconds | Kaggle seconds | Peak allocated bytes (each) | Peak reserved bytes (each) |
|---|---:|---:|---:|---:|
| Fit | 12.6781 | 11.9596 | 191,702,016 | 301,989,888 |
| Reference prediction, batch 7 | 0.3821 | 0.3679 | 25,860,608 | 301,989,888 |
| Repeated prediction, batch 7 | 0.0858 | 0.1036 | 25,860,608 | 301,989,888 |
| Permuted prediction, batch 7 | 0.0958 | 0.1221 | 25,860,608 | 301,989,888 |
| Prediction batch 3 | 0.2117 | 0.2783 | 25,319,936 | 301,989,888 |
| Prediction batch 7 | 0.0919 | 0.0972 | 25,860,608 | 301,989,888 |
| Prediction batch 16 | 0.0444 | 0.0476 | 27,075,072 | 301,989,888 |
| Save | 0.0959 | 0.0789 | 24,909,824 | 301,989,888 |
| Reload | 0.0964 | 0.1232 | 21,302,272 | 46,137,344 |
| Reloaded prediction, batch 7 | 0.0999 | 0.1029 | 22,253,056 | 48,234,496 |

Fit peaks correspond to 182.82 MiB allocated / 288 MiB reserved.
Both workflows recorded 7,105,150,976 bytes available / 8,585,216,000
total at fit completion, and 7,358,906,368 available / 8,585,216,000
total after reloaded prediction. Reload's operation record contains only
allocator peaks, not free/total. Free memory is a point-in-time query.

The later `model.resource` snapshot shows just 24,909,824 allocated bytes
because prediction/save reset the peaks; **it is not the fit peak**.
Reserved prediction peaks retain the existing caching allocator pool.
PyTorch allocated/reserved peaks do not measure total GPU/process memory:
CUDA contexts, driver/library allocations, other processes, and host RAM
are outside these allocator figures. No host-RAM peak was measured.

### Startup-aware runtime projections and decision

Let `H` be recorded combined fit overhead and `T1,V1,T2,V2` the measured
training/validation epoch costs. The central illustrative fit projection is:

```text
fit(N) = H + (T1 + V1) + (N - 1) * (T2 + V2)
source = 4.989160 + 6.898681 + (N - 1) * 0.790213 seconds
kaggle = 3.462610 + 6.092385 + (N - 1) * 2.404607 seconds
```

This charges observed first-epoch startup once rather than multiplying the
two-epoch wall time. The ranges below hold warm training at `T2` and use
the observed minimum/maximum validation time for subsequent epochs:
`H + T1 + V1 + (N-1) * (T2 + min/max(V1,V2))`. They are sensitivity
ranges, not confidence intervals or guarantees.

| Executed epochs | Source projected fit | Kaggle projected fit |
|---|---:|---:|
| 20 | 26.90-27.19 s | 55.24-57.75 s |
| 50 | 50.61-51.34 s | 127.38-133.86 s (2.12-2.23 min) |
| 200 | 169.14-172.11 s (2.82-2.87 min) | 488.07-514.37 s (8.13-8.57 min) |

For two sequential Stage 3D runs reaching 200 epochs, this is about
10.95-11.44 minutes of fit time combined. Adding the measured outside-fit
workflow costs once per domain gives roughly **11.3-11.8 minutes** for the
two workflows, excluding interpreter startup and unmeasured changes in
reporting/checkpoint costs over a longer history. Reserve scheduling
headroom (for example 15-20 minutes), not an enforced runtime limit.
Early stopping determines actual executed epochs; 200 is a cap, not a
prediction that either run will reach it.

Warm epoch validation accounts for 34.8% of source training-plus-validation
time and 77.1% for Kaggle. Thus Playground's full validation dominates its
steady epoch cost; source validation does not. The population remains
unchanged despite this difference.

There was substantial first-epoch training startup (6.6091 versus 0.5155 s
for source; 4.1057 versus 0.5501 s for Kaggle), but validation differed only
about 5.4% / 7.1% between epochs. The separate startup term and validation
sensitivity ranges support a useful local-versus-cluster decision without
another resource diagnostic. **No extension was run.** Only one warm
training epoch per domain was measured: sustained thermal/power behavior,
contention, later training variability, cumulative checkpoint/reporting
cost, actual patience stop, and longer-run correctness remain uncertain.
The unexplained UTC/performance-timer discrepancy also limits wall-clock
precision, not population or numerical verification.

No run failed and no correction is needed before Stage 3D. Measured memory
headroom and projected runtime support **proceeding locally with the
approved profile**, rather than requiring cluster execution. Stage 3D is
ready to be scheduled separately, with device availability rechecked; its
full-profile stopping behavior and feasibility are not yet established.
These timings imply nothing about accuracy or convergence. Do not mark
M09/M10 complete, revise the profile, or begin test evaluation from this
resource evidence.

## Stage 3D: full-profile local feasibility

Date: 2026-10-26 local time (2026-10-27 UTC).

**Both domains passed full-profile local feasibility. Stage 3 is complete.**
The declared training policy ran without diagnostic overrides and stopped
normally through patience. This establishes local feasibility under the
observed conditions, not predictive superiority or convergence. M09 remains
open for Stage 4 configuration/documentation acceptance; M10 remains
incomplete because no test-domain evaluation occurred.

### Execution conditions and commands

The branch remained `feat/ft-transformer-adapter`. The initial dirty
worktree, completed Stage 3B/3C artifacts, smoke implementation, and configs
were preserved. No repository instructions files were found in the
repository or inspected ancestor locations. Only this evidence document
and the integration plan were edited. No dependency installation/upgrade,
profile/code/data/split change, predictive-runner execution, PR, commit, or
push occurred.

The executable and runtime versions were reverified:
`/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python`,
Python 3.11.17, `rtdl-revisiting-models==0.0.2`,
`torch==2.14.1+cu126`, CUDA runtime 12.6, NumPy 2.4.6, pandas 3.0.6,
and scikit-learn 1.9.1. CUDA was available on the NVIDIA GeForce RTX 4060
Laptop GPU, explicit `cuda:0`. Initial Torch available/total memory was
7,440,695,296 / 8,585,216,000 bytes. Initial `nvidia-smi` showed 8 MiB
used of 8,188 MiB, 0% utilization, and no compute processes. Between and
after experiments, sampled checks also showed no compute processes and
0% utilization. No competing workload or resource failure was observed;
these were point checks, not continuous monitoring. No process was killed.

Both proposed directories were unused. Source completed and passed its
independent population/history/checkpoint audit before Kaggle began.
Exact commands:

```bash
/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_source.json \
  --output-dir results/smoke/ft_transformer_source_feasibility_cuda \
  --mode feasibility --device cuda:0

/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python -m src.experiments.smoke_ft_transformer \
  --config configs/smoke/ft_transformer_kaggle.json \
  --output-dir results/smoke/ft_transformer_kaggle_feasibility_cuda \
  --mode feasibility --device cuda:0
```

Artifacts:

- [Source feasibility run](../../results/smoke/ft_transformer_source_feasibility_cuda)
- [Kaggle feasibility run](../../results/smoke/ft_transformer_kaggle_feasibility_cuda)

Each directory contains the same nine artifact types listed for Stage 3C.
The original `smoke.json` files were not rewritten for post-run audits;
their bytes were checked unchanged after each independent audit.

### Configuration and population provenance

Requested/effective JSON files, the report, independently resolved
feasibility config, and saved/reloaded constructor settings all agreed.
Only the output directory changed from the requested config; the complete
approved model parameters were identical. In particular:

- `mode=feasibility`, maximum 200 epochs; no resource or functional override.
- `strict_common_16`, training seed 42, sampling seed 42, n2000.
- Three blocks, width 192, eight heads, ReGLU hidden width 256 through
  multiplier 4/3, attention/FFN/residual dropout 0.2/0.1/0.0.
- Train-only StandardScaler and sorted train-only categorical vocabularies,
  unseen values mapped to training-mode embeddings.
- Existing AdamW `make_parameter_groups()` policy, learning rate 1e-4,
  weight decay 1e-5; no class weighting.
- Training batch 256; complete fixed validation every epoch, batch 1024.
- Float32, explicit `cuda:0`; no mixed precision, fallback, or automatic
  batch/validation changes.
- Highest raw finite validation ROC-AUC checkpoint, earliest exact tie;
  patience 20 and minimum improvement 0.001 applied only to patience.

Both reports had `status=completed`, `training_budget_truncated=false`,
training `truncated=false`, and epoch cap 200. Patience stopping before
200 is the approved policy, not diagnostic truncation.

| Domain | Selected training (class 0 / 1) | Full validation (class 0 / 1) |
|---|---:|---:|
| Source | 2,000 (800 / 1,200) | 15,000 (6,000 / 9,000) |
| Kaggle | 2,000 (753 / 1,247) | 105,000 (39,554 / 65,446) |

Identity CSVs were checked for domain/split, unique IDs, contiguous sample
order, counts, and independently recomputed ordered/unordered SHA-256
fingerprints. Training and validation were disjoint. Both domains' entire
training and validation identity frames and summaries matched Stage 3C
exactly; their fingerprints are the Stage 3C values recorded above.
An independent same-domain validation load matched the full ordered
validation identity sequence and summary, including class counts.

The entire parsed training-sample DataFrames matched both Stage 3C and all
three corresponding existing n2000 baselines per domain (CatBoost,
TabPFN float32, TabICL v2 none/fp32/uncached). Baseline sampling hashes and
counts matched too; only baseline sampling/provenance metadata was used.
The 32-row query identities and their fingerprints matched the reported
positions within permitted validation. No test partition, baseline test
prediction/metric, or NHANES data was loaded.

### Independent complete-history and patience audit

Every saved history entry was read and audited, not just console summaries.
Epoch numbering was contiguous from 1, all losses/AUCs/durations were
finite, executed epochs were within 1-200, and the selected checkpoint was
the earliest maximum raw AUC in the entire history. Best-AUC metadata
exactly matched the selected history entry.

Patience was independently reconstructed: first AUC initializes reference
and counter zero; a gain of at least 0.001 over the last reset reference
updates it and resets the counter; otherwise increment. Both runs stopped
on the first epoch reaching counter 20.

| Audit item | Source | Kaggle |
|---|---|---|
| Executed / selected epochs | 23 / 3 | 29 / 13 |
| Stop | `patience` | `patience` |
| Patience reset epochs | 1, 2, 3 | 1, 2, 3, 5, 6, 9 |
| Final patience reference | 0.6456892962962963 | 0.6812525872644797 |
| Final counter / first exhaustion epoch | 20 / 23 | 20 / 29 |
| Selected raw validation ROC-AUC | 0.6456892962962963 | 0.6816919093141097 |
| Selected validation BCE | 0.6444024713834127 | 0.6194430263655526 |

Reconstructed reference/counter, epochs, and stop reason exactly matched
saved training metadata. Kaggle illustrates the intentional distinction:
epoch 13 increased the raw best AUC but exceeded the epoch-9 patience
reference by only 0.0004393220496300, below 0.001. Thus epoch 13 is selected
while patience still exhausts at epoch 29, not 33. The implementation
checks patience before the epoch cap, so simultaneous exhaustion would
record `patience`; neither run reached that boundary. No settings were
tuned in response to these scores.

### Numerical checks, selected checkpoint, and persistence

All checks passed with unchanged `rtol=1e-5`, `atol=1e-6` and zero
threshold disagreements at 0.5.

| Probability comparison | Source max absolute difference | Kaggle max absolute difference |
|---|---:|---:|
| Reference / repeat / restored permutation, batch 7 | 0 | 0 |
| Batch 3, partial final batch 2 | 8.169449728e-8 | 4.738891191e-8 |
| Batch 7, partial final batch 4 | 0 | 0 |
| Batch 16, two complete batches | 1.122549098e-7 | 6.385960671e-8 |
| Workflow same-device save/reload | 0 | 0 |
| Independent audit reload versus saved reference | 0 | 0 |

Input nonmutation passed. Reference probabilities were finite float64
arrays of shape `(32,)` in `[0,1]`. Restricted
`torch.load(..., map_location="cpu", weights_only=True)` verified format
version 1, finite state tensors, float32 floating tensors, schema/order,
approved architecture/constructor, exact installed dependency versions,
device `cuda:0`, and training/checkpoint metadata. Persisted preprocessing
matched fitted, reported-reloaded, and independently reloaded preprocessing
exactly: numeric means/scales/order, category vocabularies/counts,
cardinalities, fallback values/indices, and policy. Full-validation unseen
category counts were zero for all categorical features in both domains.

The saved selected model was independently loaded using
`FTTransformerAdapter.load(..., device="cuda:0")`, with one GPU model alive
at a time. Full same-domain validation logits were recomputed through the
adapter's scoped inference/backend settings in **batches of 1024**.
ROC-AUC used the implementation's float64 stable-sigmoid probabilities.
BCE used those raw float32 logits and float32 labels with
`binary_cross_entropy_with_logits`, sample-weighted across the same
1024-row batches, including the final partial batch. No public prediction
batch-256 substitution or validation subset was used.

| Selected-checkpoint audit | Source | Kaggle |
|---|---:|---:|
| Recomputed full-validation ROC-AUC | 0.6456892962962963 | 0.6816919093141097 |
| Absolute AUC difference from selected epoch | 0 | 0 |
| Recomputed full-validation BCE | 0.6444024713834127 | 0.6194430263655526 |
| Absolute BCE difference from selected epoch | 0 | 0 |
| Independent model load duration | 1.5126 s | 0.9426 s |
| Recompute duration, including preprocessing/metrics | 3.7544 s | 5.6211 s |

The extra audit durations are separate from original workflow timings.
GPU references were released and cache emptied between domain audits.
The source audit emitted a read-only NumPy label-buffer warning from its
temporary audit code; labels were never mutated and metrics matched
exactly. The Kaggle audit explicitly copied label arrays before tensor
conversion to avoid that warning. This required no repository correction,
profile change, or rerun of an experiment.

### Sustained runtime and operation-scoped memory

Durations are workflow `perf_counter` seconds, not separately measured
process-launch elapsed times. Fit includes all epochs, preprocessing,
construction, checkpoint selection/restoration, and other fit overhead.
Combined fit overhead retains the Stage 3C definition: fit minus summed
training/validation epoch durations, not pure preprocessing.

| Measurement | Source | Kaggle |
|---|---:|---:|
| Total workflow / fit | 33.8038 / 26.7523 s | 90.4664 / 78.1768 s |
| First epoch training / validation | 5.1540 / 0.2778 s | 5.6982 / 1.8892 s |
| Later training median (range) | 0.5026 (0.4494-0.5912) s | 0.5213 (0.4583-0.5952) s |
| Later validation median (range) | 0.2696 (0.2461-0.2925) s | 1.8602 (1.7483-1.9849) s |
| Summed training / validation | 16.2156 / 6.2116 s | 20.3058 / 54.3952 s |
| Validation fraction, all / later epochs | 27.70% / 34.91% | 72.82% / 78.23% |
| Combined fit overhead | 4.3252 s | 3.4759 s |
| Save / workflow reload | 0.1367 / 0.0939 s | 0.1209 / 0.0789 s |
| Fit peak allocated / reserved | 191,702,016 / 304,087,040 bytes | 191,702,016 / 304,087,040 bytes |

Fit peaks are **182.82 MiB allocated / 290 MiB reserved** for both domains,
using `operations.fit.resources`, not later allocator snapshots.
Available/total memory at fit completion was 7,103,053,824 /
8,585,216,000 bytes for each run. After workflow reload prediction it was
7,358,906,368 / 8,585,216,000 bytes. Separate full-validation checkpoint
audits peaked at 151,309,312 allocated / 226,492,416 reserved bytes each,
with 7,189,037,056 bytes available at query. All figures retain their
operation scope; allocator peaks exclude CUDA/driver/library and other
process allocations and are not total GPU/process or host-RAM peaks.

Kaggle validation dominated sustained epoch costs (78.23% after epoch 1);
source validation did not. No OOM, non-finite failure, numerical failure,
or fallback occurred. No thermal throttling or contention diagnosis is
asserted: no continuous power/temperature/process trace was collected.

### Comparison with Stage 3C and limitations

The unchanged Stage 3C calculation was evaluated at the **actual** executed
epoch count: `H3C + T1_3C + V1_3C + (N-1)*(T2_3C + V2_3C)`, with the
previous validation min/max sensitivity range.

| Domain / actual epochs | Stage 3C projected fit | Actual fit | Difference from central projection |
|---|---:|---:|---:|
| Source / 23 | 29.2725-29.6007 s | 26.7523 s | -8.61% |
| Kaggle / 29 | 76.8840-80.5848 s | 78.1768 s | +1.68% |

Source was 2.5202 s below the central projection: 1.4669 s less first-epoch
training-plus-validation startup, 0.6640 s less combined fit overhead,
0.2805 s less cumulative later training, and 0.1088 s less later validation.
Kaggle was 1.2929 s above it: 1.4951 s more first-epoch cost, 0.0133 s more
fit overhead, 0.7948 s less later training, and 0.5793 s more later
validation. These measured components explain the differences without
claiming a hardware cause. The original sensitivity range was not a
confidence interval and did not encompass startup variability.

Both full policies stopped well before 200 epochs, so feasibility is
established for the declared patience-based policy as executed, not a
200-epoch thermal soak or proof of convergence. Observed sustained
training/validation rates support local execution. The UTC timestamp spans
(35.4111 s source; 93.7186 s Kaggle) again differ from recorded performance
timer totals; wall-clock precision remains limited as noted in Stage 3C.
No predictive-quality or cross-model/test comparison follows from this
feasibility evidence.

### Checkpoint reuse and remaining work

Inspection of `src/experiments/runner.py` and its configuration contract
found that the predictive runner always creates a fresh adapter, calls
`fit`, saves it, and then evaluates. There is no supported FT
evaluation-only checkpoint-load option or provenance-aware reuse path.
`FTTransformerAdapter.load` supports same-device audit/inference, but
does not by itself provide the predictive runner's complete result
lifecycle/provenance. **Later predictive runs will retrain unless a
separate, reviewed reuse feature is added.** No resume/reuse feature was
implemented in Stage 3D.

No experiment correction is required. Stage 4 still needs the two research
configs, accepted-profile/configuration review, directly related
README/checklist/documentation updates, and M09 acceptance verification.
M09 is not marked complete here. M10 remains a separate future predictive
test-evaluation task; none was launched. Stage 3B and Stage 3C evidence is
retained above.
