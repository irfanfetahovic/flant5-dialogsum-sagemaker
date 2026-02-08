# FastAPI API Deployment Guide

## Local Development

### 1. Run API Locally (Base Model)

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**Interactive documentation:** `http://localhost:8000/docs`

### 2. Run API with Fine-Tuned Model (LoRA)

```bash
# Option A: Load LoRA weights from local filesystem
export MODEL_ID=google/flan-t5-base
export PEFT_WEIGHTS_PATH=/path/to/adapter

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option B: Load LoRA weights from S3 (requires boto3 and AWS credentials)
export MODEL_ID=google/flan-t5-base
export PEFT_WEIGHTS_PATH=s3://your-bucket/path/to/adapter

pip install boto3  # If not already installed
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option C: Use different base model
export MODEL_ID=google/flan-t5-large  # or any HuggingFace model ID
export PEFT_WEIGHTS_PATH=/path/to/adapter

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Environment Variables (can be set via export command, .env file, or platform config):**
- `MODEL_ID`: HuggingFace model ID (default: `google/flan-t5-base`)
- `PEFT_WEIGHTS_PATH`: Path to LoRA adapter weights (local filesystem or S3 path)
- `API_KEY`: Optional API key for request authentication

### 3. Test the API

```bash
# Single summarization
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue": "Customer: Hi, I need help with my order.\nAgent: Sure! What is the order number?",
    "max_tokens": 200
  }'

# Batch summarization
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "dialogues": [
      "Customer: Hi.\nAgent: Hello!",
      "User: How much?\nBot: $99/month"
    ],
    "max_tokens": 200
  }'

# Health check (basic)
curl http://localhost:8000/health

# Health check (detailed with model config)
curl http://localhost:8000/health/detailed
```

### 4. API Documentation

Once running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Deployment Options

### Option 1: Heroku (Easiest, Free Tier Available)

```bash
# 1. Create Procfile (already in repo)
cat Procfile
# Output: web: uvicorn api.main:app --host 0.0.0.0 --port $PORT

# 2. Create Heroku app
heroku create your-app-name

# 3. (Optional) Set environment variables for fine-tuned model
heroku config:set MODEL_ID=google/flan-t5-base
heroku config:set PEFT_WEIGHTS_PATH=s3://your-bucket/path/to/adapter
heroku config:set AWS_ACCESS_KEY_ID=your_key
heroku config:set AWS_SECRET_ACCESS_KEY=your_secret

# 4. Deploy
git push heroku main

# 5. Test
curl https://your-app-name.herokuapp.com/health
curl https://your-app-name.herokuapp.com/health/detailed
```

**Cost:** Free tier (limited), $7+/month for production  
**Setup time:** 5 minutes

### Option 2: AWS Lambda (Serverless)

Using Mangum for ASGI->Lambda adapter:

```bash
# 1. Install Mangum
pip install mangum

# 2. Create lambda handler
# In api/lambda_handler.py:
# from mangum import Mangum
# from api.main import app
# handler = Mangum(app)

# 3. Set environment variables in AWS Lambda console or via AWS CLI:
aws lambda update-function-configuration \
  --function-name dialog-summarizer \
  --environment Variables="{MODEL_ID=google/flan-t5-base,PEFT_WEIGHTS_PATH=s3://your-bucket/path}"

# 4. Deploy with serverless framework or AWS SAM
serverless deploy

# Or use AWS Lambda console (zip file upload)
```

**Cost:** ~$0.20/million requests + data transfer  
**Setup time:** 15-30 minutes

**Cost:** ~$0.20/million requests + data transfer

### Option 3: Docker + AWS ECS/EC2 (Recommended)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Deploy:
```bash
# Build
docker build -t dialogsum-api:latest .

# Run locally
docker run -p 8000:8000 dialogsum-api:latest

# Push to ECR and deploy to ECS
aws ecr create-repository --repository-name dialogsum-api
docker tag dialogsum-api:latest {account}.dkr.ecr.us-east-1.amazonaws.com/dialogsum-api:latest
docker push {account}.dkr.ecr.us-east-1.amazonaws.com/dialogsum-api:latest
```

**Cost:** ~$10-30/month for EC2, ~$0-5/month for ECS Fargate

### Option 4: Google Cloud Run

```bash
# 1. Create Dockerfile (above)

# 2. Deploy
gcloud run deploy dialogsum-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# API available at: https://dialogsum-api-{hash}.a.run.app
```

**Cost:** ~$0.24/million requests, generous free tier

### Option 5: Railway (Simple Alternative)

```bash
# Just connect your GitHub repo
# Railway auto-deploys on push
# Auto-generates API URL
```

**Cost:** $5/month free tier, pay-as-you-go after

## Production Security

Before deploying to production:

### 1. Add Authentication

```python
# In api/main.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/summarize")
async def summarize_single(
    request: SummarizeRequest,
    credentials: HTTPAuthCredentials = Depends(security)
):
    # Validate token
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # ... rest of code
```

### 2. Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/summarize")
@limiter.limit("100/minute")
async def summarize_single(request: SummarizeRequest):
    # ... code
```

### 3. Enable HTTPS

Most cloud providers auto-enable HTTPS. For local testing:

```bash
# Generate self-signed cert
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Run with SSL
uvicorn api.main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```

### 4. CORS Configuration

Update `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Monitoring

### 1. Logging

Already configured in `api/main.py`. Check logs:

```bash
# Heroku
heroku logs --tail

# AWS Lambda
aws logs tail /aws/lambda/dialogsum-api --follow

# Local
# Logs printed to console
```

### 2. Health Checks

API includes `/health` endpoint for monitoring:

```bash
# Setup monitoring alerts
# PagerDuty, Datadog, CloudWatch, etc.

curl -X GET https://your-api.com/health \
  -H "Authorization: Bearer {token}"
```

### 3. Performance Metrics

Track in your deployment platform:
- Request latency (p50, p95, p99)
- Error rate
- Requests per second

## Cost Comparison

| Platform | Monthly Cost | Request Cost |
|----------|------------|--------------|
| Heroku (free) | $0 | ~$0.01/request |
| Heroku (production) | $7-50 | Included |
| AWS Lambda | $0 (first 1M) | $0.0000002/req |
| Google Cloud Run | $0 (first 2M) | $0.0000004/req |
| Railway | $5/month | Included |
| EC2 t3.micro | $10/month | Included |

**Recommendation for freelance:** Start with **Railway** or **Google Cloud Run** (free tier, simple deployment)

## Example: Deploy to Railway (Simplest)

1. Push code to GitHub
2. Go to railway.app
3. Connect GitHub repo
4. Railway auto-deploys
5. Get API URL automatically
6. Done!

## Next Steps

1. **Test locally:** `uvicorn api.main:app --reload`
2. **Choose platform:** Railway/Cloud Run for simplicity, Lambda for scale
3. **Add authentication:** Use API tokens/keys
4. **Monitor:** Set up error alerts
5. **Document:** Provide client with API docs link (`/docs`)

## Client Integration Example

```python
# Client code to use your API
import requests

API_URL = "https://your-api.example.com"
API_TOKEN = "your-secret-token"

response = requests.post(
    f"{API_URL}/summarize",
    json={
        "dialogue": "Customer: Hi...\nAgent: Hello...",
        "max_tokens": 200
    },
    headers={"Authorization": f"Bearer {API_TOKEN}"}
)

summary = response.json()["summary"]
print(summary)
```
