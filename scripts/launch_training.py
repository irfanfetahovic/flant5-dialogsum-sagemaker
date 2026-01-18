"""
Launch SageMaker training job.
"""

import argparse
import logging
import yaml
from sagemaker.huggingface import HuggingFace
from src.sagemaker_config import initialize_sagemaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Launch training job."""

    # Load configuration
    config = load_config()

    # Initialize SageMaker
    session, role, bucket, region = initialize_sagemaker()

    logger.info(f"Launching training job: {args.job_name}")

    # Get config values
    model_name = config["model"]["name"]
    instance_type = config["training"]["instance_type"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    s3_prefix = args.s3_prefix

    # Create estimator
    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="src",
        instance_type=instance_type,
        instance_count=config["training"]["instance_count"],
        role=role,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        use_spot_instances=config["training"]["use_spot"],
        max_wait=config["training"]["max_wait"],
        hyperparameters={
            "model-name": model_name,
            "num-epochs": epochs,
            "batch-size": batch_size,
            "learning-rate": learning_rate,
            "lora-r": config["model"]["peft"]["r"],
            "lora-alpha": config["model"]["peft"]["lora_alpha"],
            "lora-dropout": config["model"]["peft"]["dropout"],
        },
    )

    # Launch training
    logger.info(f"Launching job with config:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Instance: {instance_type}")
    logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"  Dataset: s3://{bucket}/{s3_prefix}/")

    estimator.fit(
        {
            "train": f"s3://{bucket}/{s3_prefix}/train.jsonl",
            "validation": f"s3://{bucket}/{s3_prefix}/val.jsonl",
        }
    )

    logger.info(f"Training job complete: {estimator.latest_training_job.name}")
    logger.info(f"Model artifacts saved to: {estimator.model_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--job-name", required=True, help="Training job name")
    parser.add_argument("--s3-prefix", default="llm", help="S3 prefix for dataset")

    args = parser.parse_args()
    main(args)
