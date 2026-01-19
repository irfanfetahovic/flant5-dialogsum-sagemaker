# SageMaker Deployment Guide

## 0. Prerequisites
- AWS credentials configured (`aws configure`) and a default region (e.g., `AWS_REGION` env var).
- IAM role with SageMaker + S3 permissions; when running locally set `SAGEMAKER_ROLE_ARN` to that role.
- An S3 bucket you can write to; by default the SDK will use the SageMaker default bucket.
- Python 3.10+ and the project requirements installed.

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Prepare and upload the dataset
Runs DialogSum sampling, writes JSONL locally, and uploads to S3.
```bash
python scripts/prepare_dataset.py \
  --train-size 125 \
  --val-size 32 \
  --s3-prefix llm
```
Outputs go to `s3://<your-bucket>/llm/train.jsonl` and `s3://<your-bucket>/llm/val.jsonl`.

## 3. Launch training
Uses `config.yaml` for model + training hyperparameters and the S3 data prefix above.
```bash
python scripts/launch_training.py \
  --job-name flan-t5-dialogsum-$(date +%Y%m%d%H%M%S) \
  --s3-prefix llm
```
Note the `model_uri` (e.g., `s3://.../output/model.tar.gz`) printed when the job finishes; you'll need it to deploy.

## 4. Deploy a real-time endpoint

### Option A: Using the deployment script (recommended)
```bash
python scripts/deploy_endpoint.py \
  --model-data s3://your-bucket/path/to/output/model.tar.gz \
  --endpoint-name flan-t5-dialogsum-endpoint \
  --test
```

This will deploy the endpoint and optionally run a test prediction.

### Option B: Manual Python session
In a Python session (Studio, notebook, or local with credentials):
```python
from sagemaker.huggingface import HuggingFaceModel
from src.sagemaker_config import initialize_sagemaker

session, role, bucket, region = initialize_sagemaker()

model = HuggingFaceModel(
    model_data="s3://your-bucket/path/to/output/model.tar.gz",  # from training job
    role=role,
    entry_point="inference.py",
    source_dir="src",
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
)

predictor = model.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
)
```

### Quick test
```python
result = predictor.predict({"inputs": "Summarize: Person A: Hi. Person B: Hello!"})
print(result)
```

### Cleanup to avoid charges
```bash
# Delete endpoint and config
aws sagemaker delete-endpoint --endpoint-name flan-t5-dialogsum-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name flan-t5-dialogsum-endpoint
```

Or in Python:
```python
predictor.delete_endpoint(delete_endpoint_config=True)
predictor.delete_model()
```

## 5. Batch/offline inference (optional)
Instead of an endpoint, create a SageMaker batch transform job using the same `model_data` URI if you only need offline scoring.

## 6. Cost notes
- Training with spot instances is enabled in `config.yaml`; adjust if you need on-demand.
- Endpoints bill per-hour; delete them when not in use.
