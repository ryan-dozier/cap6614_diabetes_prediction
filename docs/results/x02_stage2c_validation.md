# X02 Stage 2C development smoke validation

Status: **Complete for Stage 2C development-only verification; not a
research evaluation.** No canonical test features or outcomes were used.

## Environment and inputs

The `synthetic-tabular-ai` Conda environment used Python 3.11.17, NumPy
2.4.6, pandas 3.0.6, scikit-learn 1.9.1, CatBoost 1.2.10, PyTorch
2.14.1+cu126, rtdl-revisiting-models 0.0.2, TabPFN 9.2.0, and TabICL 2.2.0.
CUDA was available on an NVIDIA GeForce RTX 4060 Laptop GPU (8,585,216,000
bytes total; 7,440,695,296 bytes free at preflight). No checkpoints were
downloaded. Local checkpoint content passed the existing pinned SHA-256 checks:

- TabPFN: `/home/noob/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors`:
  `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
- TabICL: `checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt`:
  `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.

The manifest assignment fingerprint was
`fea3d3c14d7d32fb2ee371981b2e2c907cc9b1083ac56f4dfd5fdedaaacfb59d`;
the 220-specification matrix fingerprint was
`77d2429fbbbe840f5585ab4a66823d9423b9841026b9e0c504245a15ac937f24`.
The code revision recorded in the runs was `538c821`. The three CatBoost
positive-budget runs recorded a clean worktree; the remaining runs recorded a
dirty worktree while Stage 2C verification work was in progress. The Stage 2B
runner and adapters were not changed during these checks.

## Prespecified persistence criteria

Reload the identical development query using the run's recorded prediction
batch size, precision, and device. FT-Transformer and TabPFN use the existing
`rtol=1e-5, atol=1e-6` smoke criteria. TabICL uses its established absolute
criterion `64 * float32 epsilon = 7.62939453125e-6`, with zero relative
tolerance. CatBoost's native CPU save/load has no existing project tolerance;
this check uses `rtol=0, atol=1e-12` as a strict FP64 serialization engineering
criterion. All models additionally require zero disagreements at threshold 0.5.
These checks assess persistence, not batch-size invariance or predictive merit.

## Execution and artifacts

Runs were sequential; the first was CatBoost mixed with Original Synthetic.
All used seed 42, budget 250, and 64 development queries from canonical NHANES
validation outside the reserved selection subset. CatBoost and FT-Transformer
used a development cap of 5 iterations/epochs. Frozen models retained the
pinned inference profiles and did not receive selection examples.

Completed runs are under
`results/adaptation/x01_v1/development/stage2c_20261120/`; per-run logs, except
the initial console run, are under the sibling `stage2c_20261120_logs/`.
Reload reports and logs are in sibling `stage2c_20261120_verification/` and
`stage2c_20261120_verify_logs/`. They are outside the hashed run directories.
All output locations are local results artifacts and are not overwritten.

The following command pattern reproduces the twelve executions and the extra
anchor with equivalent settings from the repository root. Choose a **new**
output root for a future repeat; the recorded root already exists. The first
execution shown corresponds to the initial path check. Commands fit models and
are for development only.

```bash
PY=/home/noob/miniconda3-pbe/envs/synthetic-tabular-ai/bin/python
MANIFEST=results/adaptation/x01_v1/manifest
RUNS=results/adaptation/x01_v1/development/stage2c_20261120
CHECKS=results/adaptation/x01_v1/development/stage2c_20261120_verification
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

$PY -m src.experiments.adaptation_runner execute --manifest-dir "$MANIFEST" \
  --run-id catboost_mixed_source_B250_S1800_seed42 --mode development \
  --output-root "$RUNS" --training-cap 5 --query-size 64

# Remaining eleven runs, sequentially; do not repeat the first run.
for model in catboost ft_transformer tabpfn tabicl; do
  for suffix in real_only_none_B250_S0 mixed_source_B250_S1800 mixed_kaggle_B250_S1800; do
    run="${model}_${suffix}_seed42"
    if [ "$run" = catboost_mixed_source_B250_S1800_seed42 ]; then continue; fi
    case "$model" in
      catboost|ft_transformer) set -- --training-cap 5 ;;
      tabpfn) set -- --checkpoint /home/noob/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors ;;
      tabicl) set -- --checkpoint checkpoints/tabicl/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt ;;
    esac
    $PY -m src.experiments.adaptation_runner execute --manifest-dir "$MANIFEST" \
      --run-id "$run" --mode development --output-root "$RUNS" --query-size 64 "$@"
  done
done

$PY -m src.experiments.adaptation_runner execute --manifest-dir "$MANIFEST" \
  --run-id catboost_synthetic_only_source_B0_S2000_seed42 --mode development \
  --output-root "$RUNS" --training-cap 5 --query-size 64

# Verify each completed run in a separate, new output directory.
for run_dir in "$RUNS"/*; do
  run=${run_dir##*/}
  $PY -m src.experiments.adaptation_verify --manifest-dir "$MANIFEST" \
    --run-dir "$run_dir" --output-dir "$CHECKS/$run"
done

$PY -m src.experiments.adaptation_runner inventory --manifest-dir "$MANIFEST" \
  --mode development --output-root "$RUNS"
```

TabICL's reference configuration already identifies the local checkpoint; the
explicit path above is an equivalent deployment override. FT, TabPFN and
TabICL ran on `cuda:0`; CatBoost ran on CPU. Recorded metadata contains the
full effective settings, timings, hashes and label accounting for each run.

## Results

All rows below completed and passed same-batch reload verification. The
reported maximum difference is the maximum absolute probability difference;
all had zero threshold-0.5 decision disagreements. Fit/selection are row
counts, not research evaluation samples.

| Model | Condition | Source | Fit | Selection | Max difference | Decision flips |
|---|---|---|---:|---:|---:|---:|
| CatBoost | real-only | none | 200 | 50 | 0 | 0 |
| CatBoost | mixed | Original Synthetic | 2,000 | 50 | 0 | 0 |
| CatBoost | mixed | Playground | 2,000 | 50 | 0 | 0 |
| FT-Transformer | real-only | none | 200 | 50 | 0 | 0 |
| FT-Transformer | mixed | Original Synthetic | 2,000 | 50 | 0 | 0 |
| FT-Transformer | mixed | Playground | 2,000 | 50 | 0 | 0 |
| TabPFN | real-only | none | 200 | 0 | 0 | 0 |
| TabPFN | mixed | Original Synthetic | 2,000 | 0 | 0 | 0 |
| TabPFN | mixed | Playground | 2,000 | 0 | 0 | 0 |
| TabICL | real-only | none | 200 | 0 | 0 | 0 |
| TabICL | mixed | Original Synthetic | 2,000 | 0 | 0 | 0 |
| TabICL | mixed | Playground | 2,000 | 0 | 0 | 0 |

The additional CatBoost Original Synthetic anchor completed and reloaded with
zero difference and zero flips: 2,000 synthetic fitting rows, all 15,000
source-validation selection rows, and no NHANES adaptation labels. It checks
the task-trained anchor's distinct full-source selection path; the remaining
anchor combinations are covered by Stage 2B fixture tests.

Verification reconstructed the ordered domain-qualified identities and checked
membership, disjointness, class counts, feature exclusion, saved prediction
order, finite probabilities in [0, 1], checkpoint and artifact hashes, and
configuration and label accounting against the manifest and canonical inputs.
Mixed fits contain 1,800 synthetic rows followed by 200 NHANES rows. Each
positive-budget task-trained run used 200 NHANES fit and 50 selection labels;
each frozen run used 200 context labels and reserved 50 unused. Every run used
64 additional, disjoint NHANES development-query labels for pilot metrics.
These artifacts are marked ineligible for research comparisons. A post-reload
inventory classified all 13 present runs as completed with valid hashes.
There were no failed runs, integration fixes, or retries.

Elapsed run times were 1.1–3.6 s for CatBoost (anchor 2.5 s), 20.1–28.6 s
for FT-Transformer, 21.7–31.7 s for TabPFN, and 13.8–15.4 s for TabICL.
Recorded peak PyTorch GPU-reserved memory was 184.5–232.8 MB for FT,
1,065.4–2,382.4 MB for TabPFN, and 172.0–501.2 MB for TabICL. These
allocator measurements are not total GPU usage; CatBoost GPU peaks are not
applicable. These are laptop smoke observations, not Newton runtime estimates,
grid-feasibility evidence, or model-performance comparisons.

Stage 2D reporting and Stage 2E Newton feasibility remain pending. A final X01
protocol/feasibility freeze has not been approved; research test execution
remains gated.
