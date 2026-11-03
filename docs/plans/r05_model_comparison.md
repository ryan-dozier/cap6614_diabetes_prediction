# R05: comparison of implemented models

Prepared: November 2, 2026.
Status: **Initial Review Complete**
Prepared with AI assistance using the supplied papers and repository configurations.

## Purpose and source boundaries

This note compares learning procedures, information access, and implementation
choices for the four-model matched-n2000 experiments. It supports the approved
[research questions](research_questions.md), especially RQ1/RQ2, and provides
background for the adaptation and mitigation decisions in RQ3/RQ4.

Published benchmark results motivate model selection; they do not establish
performance on our diabetes domains. Local runs are development evidence.
Final experiments are planned for Newton HPC.

The Nature 2025 TabPFN paper describes an earlier generation than the implemented
TabPFN-3.5. The original TabICL paper explains architectural ideas inherited by
TabICLv2; it is not a fifth implemented comparator. The ordering of these papers
does not order the release dates of every checkpoint currently used.

## Learning procedure and information access

| Aspect | CatBoost | FT-Transformer | TabPFN-3.5 | TabICLv2 |
|---|---|---|---|---|
| Model family | Boosted decision trees | Task-trained feature transformer | Pretrained tabular foundation model | Pretrained tabular foundation model |
| External pretrained weights in our experiment | None | None | Yes; synthetic-task pretraining | Yes; synthetic-task pretraining |
| Use of the 2,000 sampled rows | Fit trees and categorical processing | Fit preprocessing, embeddings, and neural parameters | Labeled inference context and fitted preprocessing | Labeled inference context and fitted preprocessing |
| Neural parameter updates on diabetes data | Not applicable | Yes | None in current adapter | None in current adapter |
| Main interaction mechanism | Feature splits and tree combinations | Attention among features within each row | In-context prediction over table inputs; exact generation documented separately | Column embedding, row compression, then attention across row representations |
| Same-domain validation labels | Used for early stopping and best-model selection | Used for early stopping and checkpoint selection | Unused by current predictive fit | Unused by current predictive fit |
| Current target-domain adaptation | None | None | None | None |
| Candidate future adaptation | Refit on real-only or mixed data; weighting if selected | Train/refit or fine-tune with budgeted target labels | Change labeled context; fine-tuning requires separate implementation/version verification | Change labeled context; fine-tuning requires separate implementation/version verification |

Model support for an operation is distinct from its implementation here.
Providing NHANES training examples as context is supervised adaptation and consumes
labels even when neural parameters stay fixed. A calibration-only baseline also
consumes calibration labels. No target validation or test labels may be treated as
free adaptation data.

## Contributions and limitations of the source papers

### CatBoost

The paper introduced ordered categorical target statistics and ordered boosting
to reduce leakage-driven prediction shift during training. Its ordering principle
uses preceding observations under permutations to avoid problematic dependence
on an example's own target when constructing statistics or gradient estimates.
It reported competitive results against contemporary boosted-tree systems.
This internal prediction-shift mechanism is distinct from the generator-to-generator
and synthetic-to-observed shifts studied in this project.
Source: [Prokhorenkova et al.](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
BibTeX key: ProkhorenkovaCatBoost.

CatBoost provides the conventional task-trained reference. Its paper does not
establish that every current library configuration uses ordered boosting: our
configurations do not explicitly set boosting_type. Refer to saved effective
parameters before attributing a result to that particular algorithmic variant.

### FT-Transformer

The paper contributed a controlled comparison of tabular neural architectures,
a strong ResNet baseline, and a feature-tokenizing transformer. Numerical and
categorical features become embeddings whose interactions are learned through
within-row self-attention. Its experiments found strong neural performance while
retaining the conclusion that boosted trees and neural models had no universal
winner. Source:
[Gorishniy et al.](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html).
BibTeX key: gorishniy2021fttransformer.

Our model is trained from scratch for the selected diabetes domain. This gives a
task-trained transformer comparison alongside pretrained in-context transformers.
It does not isolate attention's benefit because embeddings, optimization,
regularization, and training procedures differ from CatBoost as well.

### TabPFN: foundational paper and implemented generation

The Nature paper advances synthetic-task pretraining for a general-purpose
tabular prediction algorithm, extending the earlier TabPFN with broader task and
data support. Labeled examples condition predictions through in-context learning.
Its small-data benchmark results support studying this approach, rather than
assuming it succeeds under our domain shifts.
Source: [Hollmann et al.](https://www.nature.com/articles/s41586-024-08328-6).
BibTeX key: hollmann2025tabpfn.

Our implementation uses the later base TabPFN-3.5 checkpoint. The official model
card identifies synthetic-task pretraining and the checkpoint filename; the release
report distinguishes base, Fast, Plus, and Thinking variants. We use the base model,
not the full collection of service variants. This note does not transfer the
Nature paper's architecture limits or benchmark numbers directly to v3.5, and
does not attribute a v3.5 improvement to a specific architectural change without
checking the full generation report.
Sources: [release report](https://priorlabs.ai/technical-reports/tabpfn-3-5)
and [model card](https://huggingface.co/Prior-Labs/tabpfn_3_5).
BibTeX keys: priorlabs2026tabpfn35, priorlabs2026tabpfn35modelcard.

### Original TabICL and TabICLv2

Original TabICL made larger-context prediction more practical by computing
distribution-aware column embeddings, aggregating features into compact row
representations, and performing dataset-level in-context prediction over those
representations. Synthetic priors included tree-based generators, with a training
curriculum increasing table size. It reported competitive classification accuracy
and efficiency on TALENT. The compression-then-ICL design therefore predates v2.
Source: [Qu et al., original TabICL](https://arxiv.org/abs/2502.05564).
BibTeX key: qu2025tabicltabularfoundationmodel.

TabICLv2 retains this structure while adding repeated feature grouping, early
target-aware embeddings, query-aware scalable softmax, a richer synthetic
generator, and changes to pretraining including Muon. These changes aim to improve
representation quality and attention over larger contexts; v2 also supports
regression, although our experiment uses its classification checkpoint.
Its benchmark comparisons are evidence about the evaluated configurations and
datasets, not proof of superiority over our TabPFN-3.5 configuration.
Source: [Qu et al., TabICLv2](https://proceedings.mlr.press/v306/qu26a.html).
BibTeX key: qu2026tabiclv2.

## Implemented profiles versus paper descriptions and defaults

These are explicit project choices, not claims that the published method always
uses them. This is not an exhaustive package-default audit.

| Model | Current profile | Consequence for interpretation |
|---|---|---|
| CatBoost | CPU; depth 7; learning rate 0.05; at most 1,000 iterations; 100-round early stopping; native categorical handling | A specified reference configuration, not an extensive hyperparameter search; verify effective boosting_type in saved artifacts |
| FT-Transformer | rtdl-revisiting-models 0.0.2; 3 blocks, width 192, 8 heads; ReGLU; AdamW at 0.0001, weight decay 0.00001; float32; maximum 200 epochs, patience 20, min_delta 0.001 | Task-trained local profile; raw best validation AUC selects the restored checkpoint, while min_delta governs the separate patience counter |
| TabPFN | Base v3.5 checkpoint tabpfn-v3.5-20260909.safetensors; 4 estimators; float32; fit_preprocessors; memory-saving enabled | No gradient fine-tuning; fitted preprocessing is not neural training; report actual run-specific query batch size |
| TabICL | Package 2.2.0; classifier v2-20260212 at revision 4dcd344ece2c00be9e831fdd35bed57b5ad83e19; 8 estimators; normalization none; FP32; uncached; native ensemble batch 1; query batch 128; column output stored on CPU | An explicit none-normalization/uncached profile; results should not be described as the paper's complete default inference recipe |

FT numerical scaling and categorical mappings are learned from training rows.
CatBoost receives unscaled numerical features and native categorical inputs.
The foundation-model adapters apply their documented training-context preprocessing.
Differences in preprocessing and ensemble size are part of the compared pipelines.

TabPFN source runs exist with query batches 256 and 1,024; the four-model report
manifest is the authority for which saved run participates in a particular report.
Do not silently replace a historical result with a newer configuration's output.

Local evidence:
[CatBoost adapter](../../src/models/catboost.py),
[FT source config](../../configs/ft_transformer_source_n2000.json),
[TabPFN source batch-1024 config](../../configs/tabpfn_source_n2000_float32_batch1024.json),
[TabICL source config](../../configs/tabicl_v2_source_n2000_none_fp32.json),
and [four-model report manifest](../../configs/reports/predictive_n2000_seed42_with_ft_transformer.json).
See also the [predictive workflow](../guides/predictive_experiments.md) and
[FT results review](../results/ft_transformer_predictive_results.md).

## What the comparison can establish

The matched experiments compare complete prediction pipelines using the same
2,000 sampled training identities within each domain, the same strict_common_16
feature set, and matching evaluation populations. They can measure how those
pipelines perform when trained or conditioned on Original Synthetic versus
Playground S5E12, and how ranking and probability quality transfer to NHANES.

The nominal n2000 budget counts training/context rows, not every accessible label.
CatBoost and FT-Transformer also use full same-domain validation partitions for
selection: 15,000 rows for Original Synthetic and 105,000 for Playground S5E12.
The current foundation-model adapters do not use these validation labels during
predictive fit. Therefore these are matched-training-row comparisons, not equal
total-label-access experiments.

Cross-model differences cannot isolate the causal effect of synthetic pretraining,
architecture, attention, or prior diversity. These factors vary jointly with
optimization, preprocessing, validation access, and ensemble policy. Domain
distinguishability alone also does not establish harmful or causal artifacts.
A controlled ablation or intervention is needed for a stronger attribution claim.

Strong NHANES ROC-AUC does not establish calibrated probabilities or clinical
usefulness. Evaluate probability-quality metrics separately. Current development
test results have already been inspected; repeating on HPC with the same held-out
participants does not create a newly unseen confirmatory test.

## Review and next actions

- [x] Draft the learning-procedure, pretraining, context, and validation-access comparison.
- [x] Record explicit local profiles and distinguish them from paper-level claims.
- [x] Identify comparison limitations and candidate adaptation interfaces.
- [ ] Reconcile any resulting corrections with the final report's methods section.
- [ ] Critically compare the closest few-shot adaptation and artifact-aware
      mitigation papers before selecting the R06 contribution.

This document does not mark R05, R06, or the final related-work section complete.
