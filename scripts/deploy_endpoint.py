"""
Deploy a SageMaker real-time inference endpoint.
"""

import argparse
import logging
from sagemaker.huggingface import HuggingFaceModel
from src.sagemaker_config import initialize_sagemaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Deploy endpoint from trained model artifacts."""

    # Initialize SageMaker
    session, role, bucket, region = initialize_sagemaker()

    logger.info(f"Deploying endpoint: {args.endpoint_name}")
    logger.info(f"Model data: {args.model_data}")

    # Create HuggingFace model
    model = HuggingFaceModel(
        model_data=args.model_data,
        role=role,
        entry_point="inference.py",
        source_dir="src",
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
    )

    # Deploy endpoint
    logger.info(f"Deploying to {args.instance_type} (this may take 5-10 minutes)...")
    predictor = model.deploy(
        instance_type=args.instance_type,
        initial_instance_count=args.instance_count,
        endpoint_name=args.endpoint_name,
    )

    logger.info(f"✓ Endpoint deployed successfully!")
    logger.info(f"  Endpoint name: {predictor.endpoint_name}")

    # Test endpoint if requested
    if args.test:
        logger.info("Testing endpoint...")
        test_input = {
            "inputs": "Summarize the following conversation:\n\nPerson A: Hi, how are you?\nPerson B: I'm good, thanks! How about you?\nPerson A: Doing well!\n\nSummary:"
        }

        result = predictor.predict(test_input)
        logger.info(f"Test result: {result}")

    logger.info("\nEndpoint is ready for use!")
    logger.info(f"\nTo delete this endpoint later, run:")
    logger.info(
        f"  aws sagemaker delete-endpoint --endpoint-name {predictor.endpoint_name}"
    )
    logger.info(
        f"  aws sagemaker delete-endpoint-config --endpoint-config-name {predictor.endpoint_name}"
    )

    return predictor.endpoint_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SageMaker endpoint")
    parser.add_argument(
        "--model-data",
        required=True,
        help="S3 URI to model artifacts (e.g., s3://bucket/path/output/model.tar.gz)",
    )
    parser.add_argument(
        "--endpoint-name",
        default="flan-t5-dialogsum-endpoint",
        help="Name for the endpoint",
    )
    parser.add_argument(
        "--instance-type", default="ml.m5.xlarge", help="Instance type for endpoint"
    )
    parser.add_argument(
        "--instance-count", default=1, type=int, help="Number of instances"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run a test prediction after deployment"
    )

    args = parser.parse_args()
    main(args)
