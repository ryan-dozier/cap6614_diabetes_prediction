from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SOURCE_RAW_PATH = RAW_DIR / "source" / "diabetes_dataset.csv"
KAGGLE_RAW_PATH = RAW_DIR / "kaggle_s5e12" / "train.csv"
NHANES_RAW_DIR = RAW_DIR / "nhanes_2015_2016"

NHANES_PREPARATION_DIR = RESULTS_DIR / "nhanes_preparation"
NHANES_INSPECTION_DIR = RESULTS_DIR / "nhanes_inspection"
DOMAIN_CLASSIFIER_DIR = RESULTS_DIR / "domain_classifier"
DISTRIBUTION_SHIFT_DIR = RESULTS_DIR / "distribution_shift"
DOMAIN_FEATURE_ABLATION_DIR = RESULTS_DIR / "domain_feature_ablation"
HARMONIZED_DOMAIN_COMPARISON_DIR = RESULTS_DIR / "harmonized_domain_comparison"