# Candidate A adaptation reporting

Stage 2D implements the approved [E05 analysis](../plans/e05_evaluation_protocol.md);
it does not itself produce research results or approve the pending X01 freeze.
The reporting command reads saved research predictions; it does not fit models,
load checkpoints, or load canonical test features or labels.

## Inputs and command

After Newton feasibility and an actual final protocol freeze have been approved,
use the project environment and supply the validated Stage 2A manifest, approved
freeze, explicit research run root, and a **new** report directory:

```bash
python -m src.reporting.adaptation \
  --manifest-dir results/adaptation/x01_v1/manifest \
  --freeze /path/to/approved-freeze.json \
  --run-root results/adaptation/x01_v1/research/approved-runs \
  --output-dir results/reports/adaptation/approved-report
```

These are illustrative paths, not a claim that approval or research runs exist.
The run root must be within `results/adaptation/x01_v1/research/`. The report
directory must be separate from the run root and must not exist. See
[X01 Section 10](../plans/x01_adaptation_protocol.md#10-x02-stage-2b-matrix-and-single-run-execution)
for the freeze record and execution gate. Artificial approval-shaped fixtures
belong only in temporary test directories, never in the real results tree.

The reporter validates the approved grid, manifest, freeze, effective scientific
profile, checkpoint hashes recorded by runs, run artifacts and hashes, label
accounting, and ordered fitting/selection/query provenance. It reads canonical
split-assignment identities, not test feature rows or labels. Saved predictions
must contain the exact NHANES test cohort, binary labels, finite probabilities,
and matching 0.5 decisions. Prediction rows are aligned by domain and participant
ID; labels must agree across runs. The observed cohort must have 545 negatives
and 101 positives. Device and checkpoint-location overrides and code revisions
are recorded, not required to be identical; scientific settings and checkpoint
content hashes must match the approved profile.

An unapproved directory or incompatible artifact causes an error rather than
being silently dropped. The output's `inventory.csv` identifies incompatible
run artifacts and `report_status.json` records failure. Missing, running,
failed, and incomplete runs remain visible in the inventory. No existing report
or run artifacts are overwritten, including failed reports; choose a fresh
output directory on retry.

## Output files

- `inventory.csv`: each approved run's identity, status, and diagnostic.
- `run_metrics.csv`: one row per completed run with AUC, average precision,
  Brier score, log loss, fixed-threshold metrics, confusion counts, ECE, and
  10-bin calibration JSON. A run appears once even when reused in contrasts.
- `comparisons.csv`: all approved contrast cells, expected and available paired
  seeds, missing run IDs, and complete/incomplete status.
- `paired_differences.csv`: per-seed metrics and signed improvements for available
  pairs, explicitly marked when their cell is incomplete.
- `summary.csv`: **complete five-seed cells only**, one row per metric,
  containing both condition means and sample SDs, mean paired improvement,
  paired sample SD, and percentile interval.
- `bootstrap_effects.csv`: all bootstrap mean paired effects for complete cells;
  `bootstrap_indices.npy` records the common participant draws when any cell is
  complete. The `.npy` rows contain indices into the ascending canonical test-ID
  order, with negatives followed by positives within each draw.
- `report.md`: concise primary AUC summary, incomplete cells, and limitations.
- `report_metadata.json`: input and code hashes, run metadata hashes and
  deployment provenance, software, bootstrap settings/index hash, counts, and
  reporting time. `report_status.json` marks completion or failure.

An empty research root yields a completed **incomplete** report with an
inventory and no five-seed summaries or intervals. Available partial seed
metrics do not make a comparison complete. A failed report directory is not a
completed report.

## Calculation and interpretation

Mixed versus real-only is primary; mixed versus the corresponding synthetic-only
anchor is a supporting **pipeline contrast**, not an isolated synthetic-utility
effect. AUC and AP use mixed minus comparator; Brier and log loss use comparator
minus mixed, so positive effects are improvements. Metrics are computed for
each seed before averaging; predictions are never averaged into an ensemble.
The real-only run is reused across sources and the anchor across budgets.

One NumPy `default_rng` stream (seed 20261112) makes 2,000 draws, sampling
participants with replacement within outcome classes. All runs, conditions,
seeds, and metrics use the same draws. Each replicate averages the five
within-seed effects, and the 2.5th and 97.5th percentiles use linear
interpolation. The point estimate is the original cohort's mean paired effect.
The existing predictive metric implementations and probability handling are
used. These are pointwise intervals conditional on the fitted models and the
observed cohort/class composition, not full training-sample uncertainty or
simultaneous intervals. The unweighted participant bootstrap ignores NHANES
survey design and does not support national-population inference; crossing zero
does not establish equivalence.

Reporting time is separate from run time. The full 220-run bootstrap has not
been benchmarked here; runtime depends on the number of completed runs and
available hardware.

## Fixture validation

The reporting regression suite exercises the complete report-generation path
with five artificial paired seeds, participant-varying imperfect predictions,
and six fixed stratified draws. It independently computes AUC from positive/
negative pairs, average precision from score thresholds, Brier and log loss
from their formulas, and percentile bounds by linear interpolation. For all
four metrics it checks each saved bootstrap effect, the point estimate, and
both interval bounds; each fixture interval has nonzero width. The fixture
distinguishes per-seed effects from an averaged-probability ensemble and from
unpaired participant draws. A separate artificial test retains the prescribed
2,000-replicate path. No real research predictions were evaluated.
