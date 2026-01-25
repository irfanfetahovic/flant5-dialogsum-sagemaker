# Quick Reference: FastAPI Model Loading

## TL;DR - Get Started in 2 Minutes

### Option 1: Base Model (Fastest)
```bash
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000
# Visit: http://localhost:8000/docs
```

### Option 2: Fine-Tuned Model (Local LoRA)
```bash
export PEFT_WEIGHTS_PATH=/path/to/adapter
uvicorn api.main:app --port 8000
# Visit: http://localhost:8000/docs
```

### Option 3: Fine-Tuned Model (S3)
```bash
export PEFT_WEIGHTS_PATH=s3://bucket/path
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
pip install boto3
uvicorn api.main:app --port 8000
# Visit: http://localhost:8000/docs
```

---

## Environment Variables Cheat Sheet

```bash
# Model Selection
MODEL_ID=google/flan-t5-base              # Default

# LoRA Adapter (optional)
PEFT_WEIGHTS_PATH=/local/path/adapter     # Local LoRA
# OR
PEFT_WEIGHTS_PATH=s3://bucket/path/adapter  # S3 LoRA

# AWS Credentials (required if using S3)
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/...
AWS_REGION=us-east-1                      # Default: us-east-1
```

---

## API Endpoints Summary

```bash
# Health Check
GET  /health
GET  /health/detailed

# Inference
POST /summarize           # Single dialogue
POST /batch               # Multiple dialogues

# Documentation
GET  /docs                # Swagger UI
GET  /redoc               # ReDoc
```

---

## Deployment Commands

### Heroku
```bash
heroku create app-name
heroku config:set MODEL_ID=google/flan-t5-base
heroku config:set PEFT_WEIGHTS_PATH=s3://bucket/path
heroku config:set AWS_ACCESS_KEY_ID=key AWS_SECRET_ACCESS_KEY=secret
git push heroku main
curl https://app-name.herokuapp.com/health
```

### Docker
```bash
docker build -t dialog-api .
docker run -e MODEL_ID=google/flan-t5-base -e PEFT_WEIGHTS_PATH=/path/to/adapter -p 8000:8000 dialog-api
```

### Google Cloud Run
```bash
gcloud run deploy api --image gcr.io/project/dialog-api \
  --set-env-vars MODEL_ID=google/flan-t5-base \
  --set-env-vars PEFT_WEIGHTS_PATH=s3://bucket/path \
  --memory 4Gi
```

---

## Testing

```bash
# Single summarization
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "A: Hi!\nB: Hello!", "max_tokens": 200}'

# Batch summarization
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"dialogues": ["A: Hi!\nB: Hello!"], "max_tokens": 200}'

# Health check
curl http://localhost:8000/health/detailed
```

---

## Common Scenarios

| Goal | Command | Time |
|------|---------|------|
| **Quick demo** | `uvicorn api.main:app` | 2 min |
| **Use fine-tuned model** | `export PEFT_WEIGHTS_PATH=...` + uvicorn | 30 sec |
| **Deploy to Heroku** | `git push heroku main` | 5 min |
| **Deploy to Cloud Run** | `gcloud run deploy` | 10 min |
| **Test API locally** | `curl http://localhost:8000/health` | 1 sec |

---

## Performance Estimates

| Configuration | Startup | Inference | Memory |
|---|---|---|---|
| Base model | 2-3 min | 500ms | 4GB |
| Base + Local LoRA | 30 sec | 600ms | 4GB |
| Base + S3 LoRA | 1-2 min | 600ms | 4GB |

*Startup time is first-run only (includes model download)*



