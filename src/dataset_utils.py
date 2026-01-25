"""
Reusable dataset utilities for dialog summarization.
"""

from datasets import load_dataset
import json
from typing import Tuple, List
import logging

# Main application configures logging (logging.basicConfig), which affects all module loggers
logger = logging.getLogger(__name__)


def load_samsum_subset(train_size: int = 1000, val_size: int = 150) -> Tuple:
    """
    Load SAMSum dataset with optional subsetting.

    Args:
        train_size: Number of training examples (default: 1000)
        val_size: Number of validation examples (default: 150)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info(f"Loading SAMSum dataset (train: {train_size}, val: {val_size})")
    dataset = load_dataset("samsung/samsum")

    train_subset = dataset["train"].select(
        range(min(train_size, len(dataset["train"])))
    )
    val_subset = dataset["validation"].select(
        range(min(val_size, len(dataset["validation"])))
    )

    logger.info(f"Loaded train: {len(train_subset)}, val: {len(val_subset)}")
    return train_subset, val_subset


# This function is used in prepare_dataset.py
# JSONL = JSON Lines; Each line is a complete JSON object
# JSONL is needed because LLM fine-tuning / instruction-tuning pipelines expect one JSON object per line, which is streamable, appendable, and memory-efficient — perfect for big datasets.
# dataset_split is either train or val split (subset) from HuggingFace dataset
def save_jsonl(dataset_split, filename: str, prompt_template: str = None):
    """
    Convert HuggingFace dataset to JSONL format with pre-formatted prompts.

    Args:
        dataset_split: HuggingFace dataset split
        filename: Output filename
        prompt_template: Template for formatting input. If None, uses default.
                        Placeholder: {dialogue}
    """
    if prompt_template is None:
        prompt_template = (
            "Summarize the following conversation:\n\n{dialogue}\n\nSummary:"
        )

    logger.info(f"Saving dataset to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset_split:
            formatted_input = prompt_template.format(dialogue=example["dialogue"])
            obj = {
                "input": formatted_input.replace("\n", " "),
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
