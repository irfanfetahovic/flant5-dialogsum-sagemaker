# FLAN-T5-Base Dialog Summarization with SageMaker

> **Looking for business overview and ROI?** See the [Client Showcase](docs/FREELANCE_SHOWCASE.md) | [Try Live Demo](#) | [Get a Quote](mailto:irfanfetahovic@gmail.com)
>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10-3.11](https://img.shields.io/badge/python-3.10--3.11-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/transformers-4.36.0-orange)](https://huggingface.co/transformers/)

**Technical Documentation** - A production-ready implementation of **FLAN-T5-Base** fine-tuned on Amazon SageMaker for multi-turn dialog summarization using **LoRA/PEFT**. This project demonstrates ML engineering best practices and cloud-native deployment patterns.

## Overview

This project demonstrates LLM fine-tuning best practices with:
- **Efficient Training**: LoRA adapter tuning (~6.8M trainable params vs 248M total)
- **Cloud-Native**: Full SageMaker integration with spot instances for cost optimization
- **Production Architecture**: Modular design, configuration management, and error handling
- **Real-world Task**: Abstractive dialog summarization from the SAMSum dataset

### Example

**Input Dialogue:**
```
Tom: Hi Lisa, how are you doing today?
Lisa: Pretty good! I've been working on the new project all morning.
Tom: That's great! How's it going so far?
Lisa: Really well. We've made a lot of progress. The team has been fantastic.
```

**Generated Summary:**
```
Lisa has been working on a new project and has made good progress with a fantastic team.
```

## Project Structure

```
├── config.yaml                    # Configuration file (training hyperparameters)
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment variables template
├── CONTRIBUTING.md                # Contributing guidelines
├── Dockerfile                     # Docker container definition
├── Procfile                       # Heroku deployment configuration
├── docs/
│   ├── deployment.md              # Deployment guide
│   └── FREELANCE_SHOWCASE.md      # Client-facing business case
├── Makefile                       # Build automation
├── pytest.ini                     # Pytest configuration
├── README.md                      # This file
├── .github/
│   └── workflows/                 # CI/CD workflows
├── app/
│   ├── demo_app.py                # Streamlit demo application
│   ├── README.md                  # Demo app documentation
│   └── .streamlit/                # Streamlit configuration
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application (production)
│   ├── models.py                  # Pydantic request/response models
│   ├── requirements.txt           # API-specific dependencies
│   └── README.md                  # API deployment guide
├── notebooks/
│   └── sagemaker_training.ipynb   # SageMaker Studio notebook
├── src/
│   ├── __init__.py
│   ├── sagemaker_config.py        # Reusable SageMaker initialization
│   ├── dataset_utils.py           # Dataset loading & formatting
│   ├── train.py                   # Training script (SageMaker entry point)
│   └── inference.py               # Inference functions & utilities
├── scripts/
│   ├── prepare_dataset.py         # Prepare & upload dataset to S3
│   ├── launch_training.py         # Launch SageMaker training job
│   ├── deploy_endpoint.py         # Deploy model endpoint
│   ├── evaluate.py                # Evaluate model with ROUGE metrics
│   ├── example_inference.py       # Example inference usage
│   └── benchmark.py               # Performance benchmarking
├── tests/
│   ├── __init__.py
│   ├── test_dataset_utils.py      # Dataset utilities tests
│   └── test_inference.py          # Inference functions tests
└── data/                          # Auto-created during dataset preparation
    └── jsonl/
        ├── train.jsonl
        └── val.jsonl
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Environment                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Prepare Dataset                                   │   │
│  │    • Load SAMSum from HuggingFace                    │   │
│  │    • Convert to JSONL format                        │   │
│  │    • Upload to S3 (1000 train, 200 val samples)     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    AWS SageMaker                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Training Job                                      │   │
│  │    • Load FLAN-T5-Base (248M params)                │   │
│  │    • Apply LoRA tuning (6.8M trainable params)      │   │
│  │    • 3 epochs, batch size 4                         │   │
│  │    • ml.m5.2xlarge instance (spot)                  │   │
│  │    • Training time: ~3-4 hours for 1000 samples     │   │
│  └──────────────────────────────────────────────────────┘   │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Model Inference (Real-time or Batch)            │   │
│  │    • Load fine-tuned FLAN-T5-Base                   │   │
│  │    • LoRA weights integrated                         │   │
│  │    • Generate summaries (max 200 tokens)            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- AWS account with SageMaker access (only for training)
- AWS credentials configured (`aws configure`) (only for training)
- ~2GB disk space

### Try the Demo (No Training Required!)

You can run the Streamlit demo immediately without any training:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
streamlit run app/demo_app.py
```

The app will use the base FLAN-T5 model (downloads automatically on first run). While not as accurate as a fine-tuned model, it will still generate reasonable summaries.

### Use the Production API (FastAPI)

For production deployments and client integrations, use the FastAPI endpoint:

```bash
# Install API dependencies
pip install fastapi uvicorn pydantic

# Run API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API documentation and interactive testing:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

**Model Loading Options:**
```bash
# Base model (default)
uvicorn api.main:app --port 8000

# With fine-tuned LoRA weights (local)
export PEFT_WEIGHTS_PATH=/path/to/adapter
uvicorn api.main:app --port 8000

# With fine-tuned LoRA weights (S3)
export PEFT_WEIGHTS_PATH=s3://bucket/path
export AWS_ACCESS_KEY_ID=key
export AWS_SECRET_ACCESS_KEY=secret
pip install boto3
uvicorn api.main:app --port 8000
```

See [api/README.md](api/README.md) for deployment options.

### Full Training & Deployment

For production-quality results, follow these steps:

#### 1. Clone & Setup

```bash
git clone https://github.com/irfanfetahovic/flant5-dialogsum-sagemaker.git
cd flan-t5-base-dialogsum-sagemaker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

#### 2. Prepare Dataset

```bash
# Download SAMSum and upload to S3
python scripts/prepare_dataset.py --train-size 1000 --val-size 150
```

This will:
- Download SAMSum dataset from HuggingFace
- Convert to JSONL format
- Upload to your S3 bucket

#### 3. Launch Training

```bash
# Via CLI
python scripts/launch_training.py --job-name my-samsum-job --s3-prefix llm

# Or open the notebook in SageMaker Studio
# notebooks/sagemaker_training.ipynb
```

#### 4. Run Inference

```bash
python scripts/example_inference.py \
  --model-id google/flan-t5-base \
  --peft-weights s3://your-bucket/model-artifacts/path

# Output:
# Dialogue: Tom and Lisa discuss project progress...
# Summary: Lisa reports good progress on the new project.
```

#### 5. Evaluate Model

```bash
python scripts/evaluate.py \
  --model-id google/flan-t5-base \
  --peft-weights s3://your-bucket/model-artifacts/path \
  --num-samples 150
```

#### 6. Deploy to Production

Choose your deployment option:

**Option A: SageMaker Endpoint** (Real-time inference)
```bash
python scripts/deploy_endpoint.py \
  --model-data s3://your-bucket/model-artifacts/path/model.tar.gz \
  --endpoint-name dialogsum-endpoint \
  --instance-type ml.m5.xlarge
```

**Option B: FastAPI on AWS ECS/EC2** (Containerized deployment)
```bash
# Build and deploy Docker container
docker build -t dialogsum-api:latest .
docker tag dialogsum-api:latest {account}.dkr.ecr.us-east-1.amazonaws.com/dialogsum-api:latest
docker push {account}.dkr.ecr.us-east-1.amazonaws.com/dialogsum-api:latest

# Deploy to ECS or EC2
# See api/README.md for detailed deployment instructions
```

See [docs/deployment.md](docs/deployment.md) and [api/README.md](api/README.md) for detailed deployment guides.

#### 7. (Optional) Benchmark Performance

```bash
python scripts/benchmark.py \
  --model-id google/flan-t5-base \
  --peft-weights s3://your-bucket/model-artifacts/path \
  --num-samples 50
```

## Results & Performance

### Model Efficiency

| Metric | Value |
|--------|-------|
| Base Model Parameters | 248M |
| LoRA Trainable Parameters | 6.8M (2.7%) |
| Training Time | ~3-4 hours (3 epochs, 1000 samples) |
| Instance Type | ml.m5.2xlarge (spot) |
| Estimated Cost | ~$0.40 |
| Dataset | SAMSum (1000 training, 200 validation) |

### Evaluation Metrics (100 test samples)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.462 |
| ROUGE-2 | 0.218 |
| ROUGE-L | 0.398 |

**Notes:**
- Metrics computed on validation set from SAMSum (1000 training samples)
- SAMSum dataset produces higher baseline ROUGE scores than DialogSum
- ROUGE scores show good alignment with human-written reference summaries
- Actual performance will be better than small 125-sample baseline

## Key Features

**Efficient Fine-tuning**
- LoRA adaptation reduces trainable parameters by 97.3%
- Faster training and inference compared to full model tuning
- Lower memory footprint

**Cloud-Native Design**
- Full AWS SageMaker integration
- Spot instance support for cost optimization
- S3-hosted datasets
- Automatic model versioning

**Well-Structured**
- Modular architecture with separation of concerns
- Basic error handling and logging
- Configuration management via YAML
- Environment variable support for credentials
- CI/CD pipeline for code quality

**Easy Integration**
- Clean API for inference (`summarize_dialogue()`, `batch_summarize()`)
- Support for both local and SageMaker environments
- Pre-built evaluation scripts with ROUGE metrics

## Testing & Monitoring

### Run Tests

```bash
pytest tests/ -v

# Output:
# tests/test_dataset_utils.py::test_save_and_load_jsonl PASSED
# tests/test_inference.py::test_summarize_dialogue PASSED
```

### Experiment Tracking with MLflow

Track training experiments locally:

```bash
# Install dependencies
pip install mlflow

# Run training with MLflow tracking
python src/train.py --use-mlflow

# View results in MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

MLflow tracks:
- Hyperparameters (learning rate, batch size, LoRA config)
- Training metrics (loss, runtime)
- Model parameters (trainable vs total)

### Performance Benchmarking

Measure inference latency and throughput:

```bash
python scripts/benchmark.py --model-id google/flan-t5-base --num-samples 50
```

Benchmark metrics:
- **Latency**: Average, P50, P95 response times
- **Throughput**: Summaries per second
- **Memory**: Model size and RAM usage
- **Token speed**: Tokens generated per second

## Configuration

Edit `config.yaml` to customize:

```yaml
aws:
  region: us-east-1
  bucket: llm-training-bucket

dataset:
  name: samsung/samsum         # SAMSum dataset (higher quality)
  train_size: 1000             # 1000 training samples
  val_size: 150                # 150 validation samples

model:
  name: google/flan-t5-base
  peft:
    r: 8                    # LoRA rank
    lora_alpha: 16          # LoRA scaling
    target_modules: ["q", "v"]
    dropout: 0.05

training:
  instance_type: ml.m5.2xlarge
  epochs: 3
  batch_size: 4               # Increased from 2 for efficiency
  learning_rate: 1e-3
  use_spot: true              # Cost optimization (~70% cheaper)

inference:
  max_new_tokens: 200
  num_beams: 1
```

## Documentation

### Quick Links
- **[api/README.md](api/README.md)** - FastAPI deployment guide
- **[docs/deployment.md](docs/deployment.md)** - Deployment guide and best practices
- **[docs/FREELANCE_SHOWCASE.md](docs/FREELANCE_SHOWCASE.md)** - Business overview & ROI
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contributing guidelines


### Key Dependencies

- `torch==2.1.0` - PyTorch framework
- `transformers==4.36.0` - HuggingFace Transformers
- `datasets==2.16.0` - HuggingFace Datasets (requires `pyarrow>=15.0.0`)
- `peft==0.7.1` - Parameter-Efficient Fine-Tuning
- `sagemaker==2.168.0` - AWS SageMaker SDK

## Environment Variables

Copy `.env.example` to `.env` and fill with your values:

```bash
cp .env.example .env
```

```
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
AWS_REGION=us-east-1
AWS_BUCKET=llm-training-bucket
```

## How It Works

### Fine-tuning with LoRA

This project uses **Low-Rank Adaptation (LoRA)** to efficiently fine-tune FLAN-T5-Base:

1. **Load pretrained model** (248M params)
2. **Apply LoRA adapters** to Q and V attention layers
3. **Freeze base model weights** - only train LoRA weights (~6.8M params)
4. **Save lightweight adapters** - adapters are only ~7MB instead of 1GB+

### Inference Pipeline

1. Load base FLAN-T5 model
2. Load LoRA weights from S3
3. Merge weights for inference
4. Generate summary

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Production Readiness

This project demonstrates **production-grade architecture patterns** with working implementations of key ML engineering practices. It serves as a strong foundation for production deployments.

### What's Included:
- ✅ Good code structure and modularity  
- ✅ Basic CI/CD for code quality  
- ✅ Configuration management  
- ✅ Basic logging  
- ✅ Error handling fundamentals
- ✅ Unit test coverage for core functionality
- ✅ Cloud deployment patterns (SageMaker, S3)
- ✅ Production API (FastAPI)
- ✅ Docker support

### Additional Considerations for Enterprise Scale:
For large-scale enterprise deployments, you may want to enhance:
- **Advanced Monitoring** - Model drift detection, A/B testing frameworks
- **Enhanced Security** - AWS Secrets Manager integration, advanced authentication
- **Infrastructure as Code** - Terraform/CloudFormation for automated provisioning
- **Comprehensive Testing** - Load tests, integration tests, end-to-end tests
- **Advanced Observability** - Distributed tracing, custom metrics dashboards
- **Compliance** - Industry-specific certifications (HIPAA, SOC2)

**Current state is suitable for:** Small-to-medium deployments, proof-of-concepts, and organizations with in-house DevOps support.

See [docs/deployment.md](docs/deployment.md) and [api/README.md](api/README.md)for deployment guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.


