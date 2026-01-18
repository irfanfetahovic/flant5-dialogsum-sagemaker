"""
SageMaker configuration module for production use.
Handles session initialization, role management, and error handling.
"""
import sagemaker
import os
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def get_sagemaker_session() -> sagemaker.Session:
    """
    Initialize and return SageMaker session.
    
    Returns:
        sagemaker.Session: Initialized session
        
    Raises:
        RuntimeError: If session initialization fails
    """
    try:
        session = sagemaker.Session()
        logger.info(f"SageMaker session initialized in region: {session.boto_region_name}")
        return session
    except Exception as e:
        logger.error(f"Failed to initialize SageMaker session: {e}")
        raise RuntimeError(f"Failed to initialize SageMaker session: {e}")


def get_execution_role() -> str:
    """
    Get execution role from SageMaker or environment variables.
    
    Returns:
        str: IAM role ARN
        
    Raises:
        ValueError: If role cannot be determined
    """
    try:
        # First, try SageMaker Studio/Notebook instance
        role = sagemaker.get_execution_role()
        logger.info(f"Using SageMaker execution role: {role}")
        return role
    except ValueError:
        # Fallback to environment variable (for local development)
        role = os.getenv("SAGEMAKER_ROLE_ARN")
        if not role:
            raise ValueError(
                "Not in SageMaker environment and SAGEMAKER_ROLE_ARN not set. "
                "Set SAGEMAKER_ROLE_ARN environment variable or run in SageMaker Studio."
            )
        logger.info(f"Using role from SAGEMAKER_ROLE_ARN: {role}")
        return role


def initialize_sagemaker() -> Tuple[sagemaker.Session, str, str, str]:
    """
    Initialize all SageMaker components.
    
    Returns:
        Tuple[sagemaker.Session, str, str, str]: (session, role, bucket, region)
    """
    session = get_sagemaker_session()
    role = get_execution_role()
    bucket = session.default_bucket()
    region = session.boto_region_name
    
    logger.info(f"AWS Region: {region}")
    logger.info(f"Default S3 Bucket: {bucket}")
    logger.info(f"Execution Role: {role}")
    
    return session, role, bucket, region
