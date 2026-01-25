# Dialog Summarization API

FastAPI service for AI-powered conversation summarization using FLAN-T5.

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

## Deployment Using Docker + AWS ECS/EC2

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

## Production Enhancements (Optional)

Recommended security features for production deployment:

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

By default, CORS allows all origins. For production, restrict to specific domains in `api/main.py`:

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

Already configured in `api/main.py`. Logs are written to stdout/stderr.

```bash
# Local development
# Logs printed to console

# Docker (local)
docker logs -f <container_id>

# AWS ECS/Fargate
# Logs automatically sent to CloudWatch Logs
aws logs tail /ecs/dialogsum-api --follow

# Or view in AWS Console: CloudWatch > Log groups > /ecs/dialogsum-api

# EC2 (Docker)
# SSH into instance and view container logs
ssh ec2-user@<instance-ip>
docker logs -f <container_id>
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

## Next Steps

1. **Test locally:** `uvicorn api.main:app --reload`
2. **Add authentication:** Use API tokens/keys
3. **Monitor:** Set up error alerts
4. **Document:** Provide client with API docs link (`/docs`)

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
