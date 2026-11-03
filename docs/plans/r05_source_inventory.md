# R05 source and citation inventory

Prepared: October 28, 2026.
Status: **Bibliography expanded October 30, 2026; critical related-work review remains incomplete.**
Owner: Ryan Dozier. Prepared with AI assistance.
Scope: the [approved research questions](research_questions.md), implemented model
profiles, three project datasets, and methods needed to design R06/E05/X01.

An identified source is not a completed critical review. The unchecked items below
require reading, extracting supported claims and limitations, and adding verified
bibliographic metadata. Suggested citation keys are provisional; the BibTeX file is located in the
[bibliography](../paper/references.bib).

## 1. Implemented models: required sources

- [x] Save bibliography entries for the four implemented model papers and the
      TabPFN-3.5 technical report/model card (October 30, 2026).

| Topic / suggested key | Primary source to read and cite | What it should support |
|---|---|---|
| TabPFN foundations: `hollmann2025tabpfn` | Hollmann et al., [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6), Nature, 2025 | Synthetic-task pretraining, in-context prediction, and the foundation-model framing. This paper alone does not document our v3.5 checkpoint. |
| Exact TabPFN generation: `priorlabs2026tabpfn35` | Prior Labs, [TabPFN-3.5 technical report](https://priorlabs.ai/technical-reports/tabpfn-3-5), September 2026; [model card](https://huggingface.co/Prior-Labs/tabpfn_3_5) | Generation-specific architecture/protocol and weight access/license. Record the exact local checkpoint revision/hash separately. |
| TabICLv2: `qu2026tabiclv2` | Qu et al., [TabICLv2: A better, faster, scalable, and open tabular foundation model](https://arxiv.org/abs/2602.11139), 2026 | Synthetic pretraining, in-context learning, and v2 architectural/training changes. Cite the paper version actually read. |
| FT-Transformer: `gorishniy2021fttransformer` | Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959), NeurIPS 2021; [author implementation](https://github.com/yandex-research/rtdl-revisiting-models) | Feature tokenization, task-trained transformer baseline, and implementation provenance. Record our pinned rtdl-revisiting-models 0.0.2 profile. |
| CatBoost: `ProkhorenkovaCatBoost` | Prokhorenkova et al., [CatBoost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), NeurIPS 2018 | Conventional reference model, ordered boosting, and categorical handling. |

- [x] Read the method papers and write a short comparison of training/context,
      external pretraining, validation-label access, and adaptation support.
- [x] Cite official software/checkpoint records as well as papers where exact
      version, local preprocessing, query batching, or persistence matters.
- [x] Separate claims about the published default from our nondefault profiles:
      especially TabICL none-normalization/FP32/uncached and the documented
      TabPFN float32 batch settings.
- [x] Do not use a publication's benchmark ranking as evidence of superiority on
      our domains or as a causal comparison of pretraining.

The [model comparison draft](r05_model_comparison.md), prepared November 2, 2026,
records these distinctions and their experimental implications.
- [x] Create the model-comparison draft with local configuration/source links.
- [x] Review the draft and reconciles it with his paper-reading notes,
      including generation-specific TabPFN-3.5 details.

## 2. Dataset sources: required citations and provenance

- [x] Save citations for Original Synthetic, Playground S5E12, and NHANES 2015–2016.
- [x] Save DIQ_I and the other 13 component codebooks referenced by the preparation
      script, plus the CDC analytic-guidelines index.
- [x] Verify remaining Kaggle publication/version metadata and access conditions;
      complete dataset-specific provenance and mapping review below.

| Source / suggested key | Primary reference | Required record |
|---|---|---|
| Original Synthetic: `thallaDiabetesDataset` | [Kaggle Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset) | Publisher as displayed, title, URL, available version/update date, actual access date, license/access conditions, filename, and verified raw SHA-256. |
| Playground S5E12: `kaggleS5E12` | [Competition data page](https://www.kaggle.com/competitions/playground-series-s5e12/data) and its linked overview/rules | Organizer/title, labeled train.csv, available release/version information, generation description, access conditions, and raw-file hash. |
| NHANES 2015–2016: `cdcNhanes2015_2016` | [CDC cycle documentation](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015) | CDC/NCHS dataset citation, cycle, downloaded component files, access dates, and public-use conditions. |
| NHANES target: `cdcDiqI` | [DIQ_I diabetes questionnaire codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DIQ_I.htm) | Exact DIQ010 question, response codes, exclusions, and how the script maps it to diagnosed_diabetes. Distinguish reported diagnosis from lab-defined diabetes. |
| NHANES measurement and survey design | [CDC analytic guidelines](https://wwwn.cdc.gov/nchs/nhanes/analyticguidelines.aspx) plus the codebook for each component actually used | Measurement definitions, eligible ages, units, skip patterns, missingness, participant linkage, examination/fasting subsamples, and applicable weights/design variables. |

Kaggle URLs resolved during this inventory, but their page bodies were not
extractable by the browsing tool. Reopen them in a browser to verify displayed
metadata and terms; this inventory does not certify their license or reproduce
the generation description. The Original Synthetic acquisition/hash verification
is existing local evidence, not newly performed here. Its exact acquisition URL
does not establish every upstream author or generation step. Do not cite the
related Hugging Face dataset as the acquired file.

- [ ] Audit script-to-codebook mappings for all retained features, including DEMO,
      BMX, BPX, cholesterol/lipid components, and questionnaire components.
- [ ] Identify exact component citations for common_20/common_22 only where used;
      distinguish approximate derived mappings from directly measured features.
- [ ] Record which components were merely downloaded/inspected versus used in the
      final cohort. Do not cite every downloaded file as if it entered predictors.
- [ ] Record why the primary cohort and sensitivity cohorts differ.
- [ ] Reconcile primary source descriptions with
      [preparation and provenance](../guides/data_and_analysis.md) and
      [the preparation implementation](../../scripts/prepare_nhanes.py).
- [ ] Share detailed provenance/measurement work with D07; R05 should cite the
      data correctly without duplicating the full appendix.
- [ ] BRFSS is not in the current predictive design. Cite it only if it is actually
      analyzed or discussed; do not imply it was an evaluation domain.

## 3. Approaches and evaluation: source checklist

- [x] Save the listed starting sources for synthetic utility, shift detection,
      tree/neural baselines, covariate/label shift, calibration, ROC/PR/AP,
      feature-removal controls, and paired ROC-AUC uncertainty.
- [x] Add initial adaptation reading candidates: Rubachev et al. (2025) on
      tabular foundation-model fine-tuning and Bickel et al. (2009) on
      discriminative covariate-shift adaptation.

| Topic | Starting source | Why it is needed / boundary |
|---|---|---|
| Synthetic fidelity versus downstream utility (RQ1/RQ4) | Hansen et al., [Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark](https://papers.nips.cc/paper/2023/hash/6aa9a05b929fb08ff46a58cab6cf860d-Abstract-Datasets_and_Benchmarks.html), NeurIPS 2023 | Motivate studying predictive utility and feature relationships rather than marginal similarity alone. Its scope does not establish our method's novelty. |
| Dataset-shift detection and domain classification (RQ1) | Rabanser et al., [Failing Loudly](https://papers.neurips.cc/paper_files/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html), NeurIPS 2019 | Justify domain distinguishability as a diagnostic; distinguish detectable shift from harmful shift or causal artifacts. |
| Tree versus neural baselines | Grinsztajn et al., [Why do tree-based models still outperform deep learning on typical tabular data?](https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html), NeurIPS 2022 | Explain retaining a strong tree reference. Historical benchmarks are not a verdict on current foundation models. |
| Covariate shift / importance weighting | Sugiyama et al., [Covariate Shift Adaptation by Importance Weighted Cross Validation](https://www.jmlr.org/papers/v8/sugiyama07a.html), JMLR 2007 | Candidate weighting/adaptation rationale and assumptions. Do not assume the conditional label relationship is invariant across our datasets. |
| Prevalence/label shift (RQ2) | Lipton et al., [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html), ICML 2018 | Distinguish changed class proportions from pure label shift. A prevalence difference does not establish unchanged feature distributions conditional on class. |
| Calibration and calibration-only adaptation (RQ2/RQ3) | Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017 | Reliability/ECE and a calibration baseline if selected. Calibration data and labels must be budgeted; no test fitting. |
| ROC versus precision–recall evaluation | Davis and Goadrich, [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874), ICML 2006 | Metric interpretation under class imbalance. Keep within-population comparisons explicit. |
| Actual average-precision computation | [Official scikit-learn model-evaluation documentation](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst) | Cite software/version for our implementation; AP is not trapezoidal PR area. Use version-matched documentation in the final bibliography where available. |
| Feature-removal and retraining controls (RQ4) | Hooker et al., [A Benchmark for Interpretability Methods in Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html), NeurIPS 2019 | Background on evaluating removal-based claims and retraining. This is not a tabular synthetic-transfer method; find closer related work before choosing our intervention. |
| Paired ROC-AUC uncertainty | DeLong et al., [Comparing the areas under two or more correlated receiver operating characteristic curves](https://pubmed.ncbi.nlm.nih.gov/3203132/), Biometrics 1988 | Candidate paired-AUC method. Does not account for training-seed variation or settle survey-design inference. Final choice belongs to E05. |

The following source needs are still open; identifying them is part of R05:

- [ ] **Few-shot tabular target adaptation:** find directly relevant recent work
      on pretrained tabular context adaptation, fine-tuning, and real-only versus
      synthetic-plus-real comparisons. Use implemented model papers as starting
      points, not as the entire adaptation literature.
- [ ] **Closest artifact-aware mitigation methods:** search for tabular
      domain-aware feature/context selection and density-ratio/sample weighting;
      compare to ordinary feature selection and random controls. This is the key
      search needed to justify R06.
- [x] **Proper scoring rules / Brier and log loss:** add a foundational source
      for their interpretation and the distinction between probability error
      and calibration alone: Gneiting and Raftery (2007),
      `gneiting2007properscoring`.
- [ ] Read the scoring-rule source and write the project-specific metric
      interpretation; adding the citation does not complete that review.
- [x] **Bootstrap and repeated-run foundations:** add Efron (1979),
      `efron1979bootstrap`, and Bouthillier et al. (2021),
      `bouthillier2021variance`.
- [ ] **Paired bootstrap and repeated-run protocol:** review participant-level
      resampling, training/sampling variability, uncertainty reporting, survey
      design, and multiplicity; add method-specific sources as needed. Select
      methods under E05 rather than assuming the foundational citations suffice.
- [ ] **Feature-level shift statistics:** document the definitions/software used
      for KS, standardized mean difference, Wasserstein distance, categorical
      total variation, and Jensen–Shannon distance/divergence. Cite a methods
      source when the statistic or interpretation is central.
- [ ] **Harmonization, selection, and measurement limitations:** use the NHANES
      codebooks/guidelines first; add methodological literature for any stronger
      claims about complete-case selection or cross-dataset label comparability.
- [ ] **Explanation-based analysis:** source SHAP/permutation approaches only if
      selected for E07; do not promise an attribution method merely because it
      appeared in an earlier proposal.

### Reviewed adaptation and mitigation draft

- [x] Compare primary studies on augmentation, fine-tuning, context retrieval,
      weighting, shift detection, and feature removal.
- [x] Record the proposed empirical contribution, candidate interventions,
      and their comparison controls.
- [x] Ryan Dozier reviewed the draft and approved moving forward on November 3,
      2026. Candidate A is the core experiment and Candidate B is a bounded
      follow-up.

Evidence is in the [adaptation and mitigation review](r05_adaptation_mitigation_review.md).
Detailed protocol checks and final paper citations remain open.

## 4. Optional background: does not expand implementation scope

- SAINT: [paper](https://arxiv.org/abs/2106.01342). Read/cite if explaining its
  deferral or comparing self-supervised/row-attention approaches.
- TabM: [official implementation and paper links](https://github.com/yandex-research/tabm).
  Background for a recent task-trained neural alternative.
- Mitra: [NeurIPS 2025 paper](https://arxiv.org/abs/2510.21204).
  Relevant adjacent work on synthetic-prior diversity; consider it when assessing
  the research gap, even if it is not implemented.
- Original TabICL: [ICML 2025 paper](https://arxiv.org/abs/2502.05564).
  Cite for historical development where useful; the v2 citation is still required
  for our implemented model generation.

A paper can be useful related work without becoming an experiment. SAINT remains
deferred, and this inventory does not select any additional model.

## 5. Record for each reviewed source

- [x] Review and clean the current 40 BibTeX entries, including authors, titles,
      available venue/year/version metadata, DOI/stable URLs, entry types,
      duplicate keys, and formatting.
- [x] Save the current paper, technical-report, model-card, dataset/codebook,
      and software-documentation citations in the shared bibliography.
- [ ] Verify bibliographic metadata for any further sources selected during
      reading; finish missing dataset publication metadata and any exact
      software/checkpoint revision citations required by final claims.
- [ ] Write 2–4 sentences on the supported claim, method, data/evaluation scope,
      limitations, and relevance to an R04 question.
- [ ] Record how it differs from our proposed intervention; identify missing
      comparisons and assumptions rather than declaring novelty from a title.
- [ ] Track reading/review ownership and actual access dates. Today's inventory
      date is not the datasets' original download date.
- [ ] Add the citation to the exact claim it supports in the report/design.
      Repository hashes, cohort counts, and experimental outcomes require our
      own artifact/code evidence in addition to external citations.
- [ ] Record current model-weight and dataset conditions separately from the
      bibliography. A citation is not a license verification.

The initial pass should prioritize the implemented models, three data sources,
synthetic-utility/shift work, and the closest adaptation/mitigation papers.
R05 is complete only after those sources are critically compared, a bibliography
is prepared, and the research gap informs R06. A list of URLs alone is insufficient.

## Bibliography update: October 30, 2026

Bibliography sourcing and cleanup: Ryan Dozier, with AI assistance.
Completed checkboxes record sourcing work; they do not imply the papers were read.

The bibliography now includes the TabPFN-3.5 report/model card, all 14 component
codebooks referenced by the NHANES preparation script, the CDC analytic-guidelines
index, Failing Loudly, and sources on fine-tuning, discriminative covariate-shift
adaptation, proper scoring rules, bootstrap foundations, and benchmark variance.

Additional reading candidates are rubachev2025finetuning (TabPFNv2 fine-tuning;
not evidence of synthetic-to-NHANES few-shot transfer) and bickel2009covariateshift
(domain-discriminative adaptation under covariate-shift assumptions; not an
artifact-specific tabular mitigation method). The scoring-rule, bootstrap, and
benchmark-variance sources provide starting points, not a final E05 protocol.

Adding entries does not complete the unchecked reading, mapping-audit, novelty,
or protocol tasks above. The closest artifact-aware tabular mitigation literature,
feature-statistic citations, and method-specific sources still need selection as
R06/E05/X01 develop. Cite component codebooks according to the actual cohort and
retained measurements. TRIGLY_I is relevant to lipid sensitivity analyses.
