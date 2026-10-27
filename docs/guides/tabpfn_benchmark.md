# Validation-only TabPFN inference benchmark

Run commands from the repository root with `synthetic-tabular-ai` active unless
a section explicitly creates or activates the environment.
See the [documentation index](../README.md) for onboarding and related guides.

`src.experiments.benchmark_tabpfn` measures prediction throughput and numerical
agreement across batch sizes. It is **not a predictive research result**: it
loads only configured train and same-domain validation partitions through
`src.data`, never requests a test partition, calculates no predictive metrics,
and does not change research configurations or their saved outputs.

Completed Original Synthetic and Playground S5E12 benchmarks each used 2,000
training rows, 2,049 same-domain validation queries, explicit float32, batches
256/512/1024, and three
measured repetitions after one full-query warm-up per case on the RTX 4060 Laptop
GPU. Saved reports are under
`results/benchmarks/tabpfn_source_n2000_float32/` and
`results/benchmarks/tabpfn_kaggle_n2000_float32/`.

| Training/query dataset | Batch 256 median (s) | Batch 1024 median (s) | Measured speedup |
|---|---:|---:|---:|
| Original Synthetic | 21.8232 | 8.3923 | 2.6004x |
| Playground S5E12 | 17.7591 | 6.8744 | 2.5834x |

Median throughput increased from 93.8909 to 244.1513 rows/s for Original
Synthetic and from 115.3775 to 298.0624 rows/s for Playground S5E12. Batch 512
medians were 12.1719 s
(168.3391 rows/s) and 10.6220 s (192.9017 rows/s), respectively.
**Batch 1024 was the fastest numerically eligible tested setting in both domains.**
All measured repetitions passed the unchanged `rtol=1e-5, atol=1e-6` criterion,
with maximum reference probability differences of `1.1920928955078125e-7`
(Original Synthetic) and `1.4901161193847656e-7` (Playground S5E12), and zero threshold disagreements.
For batch 1024, both domains recorded peak allocated memory of 2,346,739,200
bytes and peak reserved memory of 2,982,150,144 bytes (approximately 2.78 GiB
reserved). These are process PyTorch allocator peaks, not total GPU usage.

Based on this computational evidence, use **batch 1024 for upcoming n2000
float32 research runs**, without changing the context, seed, checkpoint,
ensemble, precision, fit mode, threshold, or evaluation settings. These are
inference benchmarks, **not new predictive results**. Fixed case order, laptop
thermal/cache effects, and same-domain validation queries limit interpretation:
neither the same speedup on complete cross-domain evaluation workloads nor
equivalence under different settings or hardware is guaranteed.

The explicit batch-1024 configurations are available. The Playground S5E12
run has completed; the Original Synthetic batch-1024 rerun is optional:

```bash
python3 -m src.experiments.runner --config configs/tabpfn_source_n2000_float32_batch1024.json
```

The completed Playground S5E12 batch-1024 TabPFN output is
`results/predictive/tabpfn_v35_kaggle_strict_common_16_n2000_float32_seed42_batch1024/`;
its matched n2000 CatBoost output is
`results/predictive/catboost_kaggle_strict_common_16_n2000_seed42/`.
Do not rerun either completed configuration into its existing protected output.

With `synthetic-tabular-ai` activated, the completed Original Synthetic benchmark was run
from the project root as follows (use a fresh output name to repeat it):

```bash
python3 -m src.experiments.benchmark_tabpfn \
  --config configs/tabpfn_source_n2000_float32.json \
  --output-dir results/benchmarks/tabpfn_source_n2000_float32 \
  --query-size 2049 \
  --batch-sizes 256 512 1024 \
  --repetitions 3
```

The query size, batch sizes, and repetition count shown are the defaults.
The command requires explicit float32, an explicit CUDA device, a configured
training sample budget and sampling seed, and batch size 256 among unique
positive batch sizes. The local checkpoint must already be available; there
are no downloads, CPU fallbacks, or automatic changes to model settings.
Use a **new directory under `results/benchmarks/`** each time; existing directories
are refused, including interrupted attempts. Research output directories are
never reused.

The context is selected with the existing deterministic training sampler.
Validation queries are sorted by authoritative identity (`row_index` for
Original Synthetic/Playground S5E12, `seqn` for NHANES 2015–2016), then the first `query_size` rows are used.
Selection does not use validation labels. Full train/validation identities must
be disjoint. Training context, query order, feature order, checkpoint, model
seed, ensemble count, precision, fit mode, and memory-saving settings are held
fixed across cases.

Batch 256 always executes first, followed by the other sizes in requested order.
Each case creates one fresh adapter, prepares its context once, executes one
full-query warm-up, and then performs the requested measured repetitions
without refitting. The shared batching implementation preserves query order
and the last partial chunk: 2,049 queries yield 8+1, 4+1, and 2+1 chunks for
batches 256, 512, and 1024 respectively. The adapter is released and unused CUDA
allocations are cleared between cases, not between measured repetitions.

Prediction timing uses a monotonic timer with CUDA synchronization before and
after prediction and peak counters reset before each repetition. It excludes
loading, construction/checkpoint hashing, context preparation, warm-up,
serialization, comparisons, and console output. Context preparation and
warm-up have separate timings. Memory peaks describe the **process's PyTorch
allocator, not total GPU usage**. Fixed case order, cache state, and laptop
thermals may affect timings; small timing differences are not statistically
established speedups.

Every measured vector is checked for exact row count, one dimension, finite
values, and probabilities in `[0, 1]`. The first measured batch-256 vector is
the fixed reference for **all** measured repetitions, including later
batch-256 repetitions. Reports include maximum/mean absolute differences,
`np.allclose(rtol=1e-5, atol=1e-6)`, and threshold disagreements using
`probability >= config.threshold`. Probabilities are not rounded. Matching
classifications alone do not establish probability equivalence; the tolerance
cannot be relaxed through CLI options.

Outputs, saved incrementally with atomic JSON/CSV replacement:

- `benchmark.json`: UTC timestamps, arguments, resolved configuration,
  package/device/checkpoint metadata, sampling/query fingerprints, feature
  order, operation failures, measurements, agreement and eligibility.
- `benchmark_summary.csv`: one row per requested size, including failed or
  unattempted cases; context/warm-up times, median throughput/duration,
  minimum/maximum duration, maximum measured allocator peaks, and speedup
  relative to batch 256's median duration.
- `benchmark_repetitions.csv`: one row per attempted measured repetition,
  with its status, available timing/memory, agreement, and probability path.
- `training_sample.csv.gz` and `validation_queries.csv.gz`: selected row
  provenance. Query provenance contains identifiers and query order, not
  features or labels.
- `probabilities_batch<SIZE>_repetition<N>.npy`: each measured probability
  vector, saved without pickle; load with `np.load(path, allow_pickle=False)`.

Only a case with all requested measured repetitions completed and passing
agreement is numerically eligible. The report identifies the fastest eligible
measured case, but never rewrites a research config or launches an experiment.
OOM reports identify the operation, batch size, repetition (if applicable),
and error, preserving completed measurements without altered-setting retries.
Reference OOM or disagreement stops the benchmark. Candidate OOM may continue
after successful cleanup; candidate disagreement retains its measurements,
excludes the case, and continues. Unexpected errors and KeyboardInterrupt
preserve available diagnostics and propagate; reporting/cleanup failures do
not replace the original error.

Top-level statuses and exit behavior:

- `running`: in progress, or the process ended before final diagnostics.
- `completed`: every requested case completed and passed; exit 0.
- `completed_with_issues`: reference passed but a candidate had OOM or failed
  agreement; **exit 1**, even if another candidate passed.
- `failed`: reference or unexpected execution failure; nonzero exit.

Per-case execution status and numerical eligibility are recorded separately,
with case statuses `completed`, `out_of_memory`, `agreement_failed`, `failed`,
or `not_run` (and `running` while active).
