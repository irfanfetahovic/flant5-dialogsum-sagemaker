"""
Reusable dataset utilities for dialog summarization.
"""

from datasets import load_dataset
import json
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_dialogsum_subset(train_size: int = 125, val_size: int = 32) -> Tuple:
    """
    Load DialogSum dataset with optional subsetting.

    Args:
        train_size: Number of training examples
        val_size: Number of validation examples

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading DialogSum dataset (train: {train_size}, val: {val_size})")
    dataset = load_dataset("knkarthick/dialogsum")

    train_subset = dataset["train"].select(
        range(min(train_size, len(dataset["train"])))
    )
    val_subset = dataset["validation"].select(
        range(min(val_size, len(dataset["validation"])))
    )

    logger.info(f"Loaded train: {len(train_subset)}, val: {len(val_subset)}")
    return train_subset, val_subset


def save_jsonl(dataset_split, filename: str):
    """
    Convert HuggingFace dataset to JSONL format.

    Args:
        dataset_split: HuggingFace dataset split
        filename: Output filename
    """
    logger.info(f"Saving dataset to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset_split:
            obj = {
                "instruction": "Summarize the dialog",
                "input": example["dialogue"].replace("\n", " "),
                "output": example["summary"].replace("\n", " "),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(dataset_split)} examples to {filename}")


def load_jsonl(filename: str) -> List[dict]:
    """
    Load JSONL file.

    Args:
        filename: Input filename

    Returns:
        List of dictionaries
    """
    examples = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples
