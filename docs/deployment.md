# SageMaker Deployment Guide

## 0. Prerequisites

### AWS Setup (Required)

**1. IAM User (for YOU to run scripts locally):**
- Create an IAM user in AWS Console with programmatic access
- Generate access keys (Access Key ID + Secret Access Key)
- Attach policy: `AmazonSageMakerFullAccess` + S3 permissions
- Install AWS CLI and run `aws configure` with these credentials
- Set default region (e.g., `us-east-1`)

**2. SageMaker Execution Role (for SageMaker to run jobs):**
- AWS will auto-create this role when you first use SageMaker, OR
- Create manually in IAM Console with trust policy for `sagemaker.amazonaws.com`
- Attach policies: `AmazonSageMakerFullAccess` + S3 access
- (Optional) Set `SAGEMAKER_ROLE_ARN` env var to the role ARN, or SDK will find/create it

**3. S3 Bucket:**
- An S3 bucket you can write to (SageMaker SDK will use default bucket if not specified)

**4. Local Environment:**
- Python 3.10+ and project dependencies installed

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Prepare and upload the dataset
Runs SAMSum sampling, writes JSONL locally, and uploads to S3.
```bash
python scripts/prepare_dataset.py \
  --train-size 1000 \
  --val-size 150 \
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

## 4. Deploy a real-time endpoint using the deployment script

```bash
python scripts/deploy_endpoint.py \
  --model-data s3://your-bucket/path/to/output/model.tar.gz \
  --endpoint-name flan-t5-dialogsum-endpoint \
  --test
```

This will deploy the endpoint and optionally run a test prediction.

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

## 5. Batch/offline inference (optional)
Instead of an endpoint, create a SageMaker batch transform job using the same `model_data` URI if you only need offline scoring.

## 6. Cost notes
- Training with spot instances is enabled in `config.yaml`; adjust if you need on-demand.
- Endpoints bill per-hour; delete them when not in use.
