"""
Prepare SAMSum dataset and upload to S3.
Run this BEFORE training.
"""

import os
import logging
import argparse
from src.dataset_utils import load_samsum_subset, save_jsonl
from src.sagemaker_config import initialize_sagemaker
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Prepare and upload dataset."""

    # Initialize SageMaker
    session, role, bucket, region = initialize_sagemaker()

    # Load dataset
    train_data, val_data = load_samsum_subset(
        train_size=args.train_size, val_size=args.val_size
    )

    # Create local directory
    os.makedirs("data/jsonl", exist_ok=True)

    # Save to JSONL
    train_path = "data/jsonl/train.jsonl"
    val_path = "data/jsonl/val.jsonl"

    save_jsonl(train_data, train_path, args.prompt_template)
    save_jsonl(val_data, val_path, args.prompt_template)

    # Upload to S3
    logger.info(f"Uploading to S3 bucket: {bucket}")
    s3 = boto3.client("s3", region_name=region)

    s3_prefix = args.s3_prefix

    s3.upload_file(train_path, bucket, f"{s3_prefix}/train.jsonl")
    logger.info(f"Uploaded train.jsonl to s3://{bucket}/{s3_prefix}/train.jsonl")

    s3.upload_file(val_path, bucket, f"{s3_prefix}/val.jsonl")
    logger.info(f"Uploaded val.jsonl to s3://{bucket}/{s3_prefix}/val.jsonl")

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SAMSum dataset")
    parser.add_argument(
        "--train-size", default=1000, type=int, help="Number of training examples"
    )
    parser.add_argument(
        "--val-size", default=150, type=int, help="Number of validation examples"
    )
    parser.add_argument("--s3-prefix", default="llm", help="S3 prefix for dataset")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Prompt template for formatting input. Placeholder: {dialogue}. Default: 'Summarize the following conversation:\\n\\n{dialogue}\\n\\nSummary:'",
    )

    args = parser.parse_args()
    main(args)
