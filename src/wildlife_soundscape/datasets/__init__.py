"""Dataset contracts, validation, and benchmark recommendations."""

from .registry import RECOMMENDED_DATASETS, RecommendedDataset
from .splitting import SplitRatios, assign_group_splits
from .validation import DatasetIssue, ValidationReport, validate_dataset

__all__ = [
    "DatasetIssue",
    "RECOMMENDED_DATASETS",
    "RecommendedDataset",
    "SplitRatios",
    "ValidationReport",
    "assign_group_splits",
    "validate_dataset",
]
