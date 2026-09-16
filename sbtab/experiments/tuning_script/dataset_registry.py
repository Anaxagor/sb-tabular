"""
Centralized dataset registry to ensure tuning and evaluation scripts
use the exact same dataset files and keys.
"""
from pathlib import Path

DATASETS_PATH = Path("../../data/datasets/datasets_mixed.pkl")

DATASET_KEYS = [
    "Adult",
    "Credit Approval",
    "Online Shoppers",
    "Eucalyptus",
    "Forest Fires",
    "Insurance",
    "House Sales",
    "Cardiovascular Disease",
    "Churn Modelling",
    "Auto MPG",
    "Diamonds",
    "Real Estate",
    "Stroke Prediction",
    "Palmer Penguins"
]

KEY_NORMALIZER = lambda k: str(k).strip().lower().replace(" ", "_")
