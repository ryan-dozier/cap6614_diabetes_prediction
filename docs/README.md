# Project documentation

Start with [project onboarding](../Readme.md#quick-start) for installation,
data setup, and a first experiment. This index covers detailed workflows, design
records, and written results reviews.

## Guides

Instructions for current workflows:

- [Environment setup and maintenance](guides/environment_setup.md)
- [Data preparation, provenance, and analysis](guides/data_and_analysis.md)
- [Shared predictive experiments and artifacts](guides/predictive_experiments.md)
- [FT-Transformer usage](guides/ft_transformer_usage.md)
- [TabPFN usage](guides/tabpfn_usage.md)
- [TabICL setup and usage](guides/tabicl_usage.md)
- [TabPFN inference benchmarking](guides/tabpfn_benchmark.md)
- [Comparison reporting](guides/reporting.md)

## Workflow shortcuts

- [Dataset provenance and raw-file recovery](guides/data_and_analysis.md#raw-data-and-provenance)
- [Pinned TabICL checkpoint download and verification](guides/tabicl_usage.md#pinned-checkpoint-setup)
- [Full preparation and analysis command sequence](guides/data_and_analysis.md#running-the-current-analysis-pipeline)
- [Four-model comparison report](guides/reporting.md#four-model-comparison-report-e04b)

## Plans

Design decisions and implementation-stage acceptance records. Historical sections
retain their original scope; use each document's current-status notes.

- [Research questions and hypotheses — approved (R04)](plans/research_questions.md)
- [FT-Transformer integration plan](plans/ft_transformer_integration_plan.md)
- [TabICL integration plan](plans/tabicl_integration_plan.md)

## Results reviews

Tracked written analyses and validation records:

- [FT-Transformer predictive results](results/ft_transformer_predictive_results.md)
- [FT-Transformer feasibility evidence](results/ft_transformer_stage3_results.md)
- [TabICL predictive results](results/tabicl_predictive_results.md)
- [TabICL feasibility evidence](results/tabicl_stage3_results.md)

These documents are maintained research narratives. Project scripts do not generate
them. The repository-root `results/` directory holds generated predictions,
metrics, reports, figures, and fitted artifacts; those outputs are ignored by Git.
Moving these written reviews does not change generated-output locations.
The reporting CLI accepts `--output-dir` under `results/reports/` and refuses
existing output directories.

See [PROJECT_CHECKLIST.md](../PROJECT_CHECKLIST.md) for ownership, contributions,
commit/PR evidence, and remaining research tasks.
