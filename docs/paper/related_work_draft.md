# Related work

Draft prepared November 9, 2026; reviewed and approved by Ryan Dozier
for the current planning milestone. Prepared with AI assistance.
Citation commands use keys in the accompanying references.bib.
The final paragraph describes planned experiments and should be updated when
the experimental scope and results are finalized.

## Tabular prediction and foundation models

Gradient-boosted decision trees provide a strong reference for tabular
prediction. CatBoost introduced ordered target statistics for categorical
features and ordered boosting to address leakage-driven prediction shift
during fitting \cite{ProkhorenkovaCatBoost}. This mechanism concerns differences
introduced by the learning procedure, while our study examines differences
between synthetic training domains and an observed evaluation population.
FT-Transformer takes a neural approach by embedding numerical and categorical
features and applying attention among feature tokens within each record.
Gorishniy et al. found that this architecture and a carefully tuned ResNet
provided strong neural baselines, while their comparison with boosted trees
identified no universally superior model family
\cite{gorishniy2021fttransformer}. These studies motivate retaining both a
tree-based reference and a task-trained transformer when evaluating newer
pretrained approaches.

Tabular foundation models learn prediction procedures across datasets before
being applied to a downstream task. Hollmann et al. trained TabPFN on synthetic
prediction tasks and used labeled examples as context for predicting queries.
Their Nature study extended the earlier TabPFN to a broader range of tabular
problems and reported strong performance on small-data benchmarks
\cite{hollmann2025tabpfn}. Original TabICL addressed scalability by constructing
distribution-aware column embeddings, compressing features into row
representations, and performing in-context prediction over those representations
\cite{qu2025tabicltabularfoundationmodel}. TabICLv2 builds on this design with
early target-aware embeddings, repeated feature grouping, query-aware attention
scaling, a richer synthetic prior, and revised pretraining
\cite{qu2026tabiclv2}. Our experiments use TabICLv2 and the later base
TabPFN-3.5 checkpoint, whose generation is documented separately from the Nature
paper \cite{priorlabs2026tabpfn35,priorlabs2026tabpfn35modelcard}.
Comparing these models with CatBoost and FT-Transformer evaluates complete
prediction pipelines that differ in architecture, pretraining, and fitting
procedure.

## Synthetic data and downstream utility

Manousakas and Aydöre examined synthetic tabular data for sharing,
summarization, augmentation, and class balancing. Their augmentation experiment
trained generators on subsets of real training data, added an equal number of
generated rows, and compared downstream prediction with real-only training.
Across the evaluated datasets and generators, augmentation did not consistently
improve predictive performance \cite{manousakas2023usefulnesssynthetictabulardata}.
This result motivates treating the benefit of synthetic examples as an
empirical question and retaining a baseline that uses the same available real
examples.

Generator design can nevertheless change downstream utility. TAEGAN combines
a masked autoencoder generator with self-supervised warmup, sampling designed
for imbalance and skew, and losses intended to preserve distributions and
relationships. Its evaluation compares synthetic-data utility across eight
datasets and several generative baselines \cite{liTAEGAN}.
ReTabSyn uses direct preference optimization to favor observed rows over
perturbed alternatives, with target perturbations emphasizing feature-outcome
relationships and feature perturbations discouraging implausible
co-occurrences. Its experiments include small training samples, imbalanced
outcomes, and shifts created by demographic subgroup splits
\cite{lin2026retabsynrealistictabulardata}.
Together, these studies show why the source and construction of synthetic
examples matter when interpreting downstream results.

The present study evaluates two existing synthetic diabetes tables acquired
separately from NHANES. This setup differs from training a generator on the
available NHANES participants and evaluating the resulting samples on a
held-out partition of the same survey dataset. Holding each synthetic source
fixed allows us to ask whether its examples add value beyond a specified
budget of labeled NHANES participants. Comparing real-only, synthetic-only,
and mixed conditions also makes the dependence on the chosen synthetic
source observable.

## Adaptation and context selection

Adaptation of tabular foundation models can change the labeled examples
provided as context or update pretrained weights. Rubachev et al. compared
TabPFNv2 fine-tuning procedures with frozen inference and task-trained
models. They reported improvements on academic benchmarks alongside less
stable performance on temporally shifted datasets, showing that the
evaluation setting matters when assessing adaptation
\cite{rubachev2025finetuning}. Their experiments motivate evaluating frozen
context-based adaptation before introducing the additional optimization
and selection choices required by task-specific fine-tuning.

Localized TabICLv2 provides a related example of changing inference context.
Guta retrieves training neighbors by cosine similarity in the model's learned
row-representation space, then supplies the selected rows to the in-context
prediction stage. The paper also fine-tunes the row-representation and
prediction stages on generated synthetic tasks. Its TabArena experiments
compare full-context inference, localized inference, and localized inference
with those additional weight updates
\cite{guta2026localizedtabiclv2scalingtabular}. These results establish a
precedent for studying context composition and the accuracy-cost trade-off.
In our proposed experiment, the context intervention concerns the proportion
of synthetic and labeled NHANES examples, with the available real-label
budget explicitly counted.

## Distribution shift and feature-removal controls

Distribution-shift methods provide tools for characterizing differences
between the synthetic tables and NHANES. Rabanser et al. evaluated shift
detection methods, including domain-discriminating approaches
\cite{rabanser2019failingloudly}. Bickel et al. studied learning under
covariate shift, where feature distributions change while the conditional
outcome relationship is assumed to remain stable
\cite{bickel2009covariateshift}. That assumption is consequential for our
setting because differences in sampling and variable definitions can affect
the relationship between harmonized predictors and reported diagnosis.
Domain discrimination can identify features that separate the datasets,
but determining whether those features impair prediction requires a
downstream comparison.

Feature removal offers one such comparison. Hooker et al. evaluated
importance-based removal by retraining models after inputs were removed and
comparing against random removal \cite{HookerBenchmarkInterpretability}.
Their image-based evaluation supplies a methodological precedent for our
proposed tabular follow-up. We would compare domain-ranked feature removal
with the full feature set, repeated random removal of the same number of
features, and a source-predictive selection baseline. Refitting each pipeline
on the retained features would test whether the ranking rule improves
transfer while controlling for the number of available inputs.

## Evaluation and position of this study

Ranking performance and probability quality capture different aspects of
transfer. Guo et al. studied confidence calibration and post-processing
methods, while Gneiting and Raftery established a framework for evaluating
probabilistic predictions through proper scoring rules
\cite{GuoConfidenceCalibration,gneiting2007properscoring}.
These perspectives motivate evaluating ROC-AUC alongside probability-quality
measures such as Brier score and log loss. Paired evaluation on the same
participants addresses differences between predictions, while repeated
training or context draws address variability in the learning procedure.
The latter concern is emphasized by Bouthillier et al.'s analysis of
benchmark variance \cite{bouthillier2021variance}.

Our planned contribution is an empirical comparison of how two existing
synthetic diabetes sources transfer to NHANES through task-trained and
pretrained tabular models. The primary adaptation experiment asks whether
adding synthetic examples improves prediction beyond the same limited
set of labeled NHANES participants. A bounded follow-up would examine
domain-informed feature removal with matched controls. This design connects
synthetic-data utility, context-based adaptation, and distribution-shift
analysis in a common evaluation population. It can establish the benefits
and limitations of the tested sources and procedures without attributing
cross-model differences to any single architectural or pretraining choice.
