# Production Implementation Guide

This guide provides step-by-step instructions to implement three critical production features currently missing from the codebase:

1. **Model Monitoring** - Track model performance, data drift, and inference metrics
2. **Production Deployment Automation** - Automate deployment with Infrastructure as Code
3. **Secret Management Integration** - Securely manage credentials and API keys

---

## 1. Model Monitoring Implementation

### 1.1 Overview

Currently, the project has:
- ✅ Basic logging in `api/main.py`
- ✅ Health check endpoint `/health`
- ❌ No performance metrics collection
- ❌ No data drift detection
- ❌ No alert system
- ❌ No metrics dashboard

### 1.2 Implementation: Prometheus Metrics + CloudWatch

#### Step 1: Install Dependencies

```bash
pip install prometheus-client boto3 python-json-logger
```

Update `requirements.txt`:
```
prometheus-client==0.19.0
python-json-logger==2.0.7
boto3==1.34.0
```

#### Step 2: Create Monitoring Module

Create file: `src/monitoring.py`

```python
"""
Monitoring and observability module for inference pipeline.
Collects metrics for Prometheus and CloudWatch.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info
import boto3
from pythonjsonlogger import jsonlogger

# Prometheus metrics
inference_counter = Counter(
    'inference_total',
    'Total number of inferences',
    ['model_id', 'status']
)

inference_duration = Histogram(
    'inference_duration_seconds',
    'Inference duration in seconds',
    ['model_id'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

active_requests = Gauge(
    'active_requests',
    'Number of requests currently being processed'
)

error_counter = Counter(
    'error_total',
    'Total number of errors',
    ['model_id', 'error_type']
)

model_info = Info(
    'model_info',
    'Information about the loaded model'
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device']
)

model_memory_usage = Gauge(
    'model_memory_usage_mb',
    'Model memory usage in MB'
)

queue_depth = Gauge(
    'inference_queue_depth',
    'Current inference queue depth'
)


class StructuredLogger:
    """Structured logging for JSON-formatted logs."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter for structured logs
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = jsonlogger.JsonFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_inference(
        self,
        model_id: str,
        status: str,
        duration_ms: float,
        output_length: int,
        error: Optional[str] = None
    ):
        """Log inference event with structured data."""
        self.logger.info(
            "inference_event",
            extra={
                "timestamp": datetime.utcnow().isoformat(),
                "model_id": model_id,
                "status": status,
                "duration_ms": duration_ms,
                "output_length": output_length,
                "error": error,
            }
        )


class ModelMonitor:
    """Monitor operational metrics (latency, errors, GPU, memory).
    
    Note: This class handles INFRASTRUCTURE monitoring only.
    For data/model drift detection, use EvidentlyMonitor (Section 1.3).
    """
    
    def __init__(self, model_id: str, cloudwatch_enabled: bool = False):
        self.model_id = model_id
        self.cloudwatch_enabled = cloudwatch_enabled
        self.logger = StructuredLogger(__name__)
        
        if cloudwatch_enabled:
            self.cloudwatch_client = boto3.client('cloudwatch')
        
        # Track recent metrics for operational summary
        self.inference_times = []
        self.output_lengths = []
        self.token_counts = []
    
    def record_inference(
        self,
        duration_ms: float,
        output_length: int,
        token_count: int,
        status: str = "success",
        error: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """Record inference metrics and update Prometheus counters."""
        
        # Update Prometheus metrics
        inference_counter.labels(
            model_id=self.model_id,
            status=status
        ).inc()
        
        inference_duration.labels(
            model_id=self.model_id
        ).observe(duration_ms / 1000.0)
        
        # Track errors by type
        if error and error_type:
            error_counter.labels(
                model_id=self.model_id,
                error_type=error_type
            ).inc()
        
        # Track for drift detection
        self.inference_times.append(duration_ms)
        self.output_lengths.append(output_length)
        self.token_counts.append(token_count)
        
        # Keep only last 100 samples for drift detection
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
            self.output_lengths.pop(0)
            self.token_counts.pop(0)
        
        # Log structured event
        self.logger.log_inference(
            model_id=self.model_id,
            status=status,
            duration_ms=duration_ms,
            output_length=output_length,
            error=error
        )
        
        # Send to CloudWatch if enabled
        if self.cloudwatch_enabled and status == "success":
            self._send_to_cloudwatch(duration_ms, output_length)
    
    def _send_to_cloudwatch(self, duration_ms: float, output_length: int):
        """Send metrics to AWS CloudWatch."""
        try:
            self.cloudwatch_client.put_metric_data(
                Namespace='FlanT5DialogSum',
                MetricData=[
                    {
                        'MetricName': 'InferenceDuration',
                        'Value': duration_ms,
                        'Unit': 'Milliseconds',
                        'Dimensions': [
                            {
                                'Name': 'ModelId',
                                'Value': self.model_id
                            }
                        ]
                    },
                    {
                        'MetricName': 'OutputLength',
                        'Value': output_length,
                        'Unit': 'Count',
                        'Dimensions': [
                            {
                                'Name': 'ModelId',
                                'Value': self.model_id
                            }
                        ]
                    }
                ]
            )
        except Exception as e:
            self.logger.logger.warning(f"Failed to send metrics to CloudWatch: {e}")
    
    def set_model_info(self, model_config: Dict[str, str]):
        """Set model information metadata."""
        model_info.info(model_config)
    
    def update_gpu_metrics(self):
        """Update GPU utilization metrics if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Get memory usage as proxy for utilization
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
                    
                    # Calculate utilization percentage
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**2
                    utilization = (memory_allocated / total_memory * 100) if total_memory > 0 else 0
                    
                    gpu_utilization.labels(device=f"cuda:{i}").set(utilization)
                    
                    # Also update model memory gauge
                    model_memory_usage.set(memory_allocated)
        except Exception as e:
            self.logger.logger.warning(f"Failed to update GPU metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        if not self.inference_times:
            return {}
        
        return {
            "total_inferences": len(self.inference_times),
            "avg_inference_time_ms": sum(self.inference_times) / len(self.inference_times),
            "avg_output_length": sum(self.output_lengths) / len(self.output_lengths),
            "min_inference_time_ms": min(self.inference_times),
            "max_inference_time_ms": max(self.inference_times),
        }


```

**Note:** Alerting is handled by Prometheus AlertManager (see Step 4), not in application code.

#### Step 3: Update FastAPI to Use Monitoring

Update `api/main.py` to integrate monitoring:

```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.monitoring import ModelMonitor, active_requests
from fastapi.responses import Response
import time

# Initialize monitoring
monitor = ModelMonitor(model_id=os.getenv("MODEL_ID", "google/flan-t5-base"))

# Set model info on startup
@app.on_event("startup")
async def startup_monitoring():
    """Initialize monitoring on startup."""
    monitor.set_model_info({
        'model_id': os.getenv("MODEL_ID", "google/flan-t5-base"),
        'version': '1.0.0',
        'lora_merged': str(os.getenv("PEFT_WEIGHTS_PATH") is not None)
    })

# Add route for Prometheus metrics
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Middleware to track active requests
@app.middleware("http")
async def track_active_requests(request: Request, call_next):
    """Track active requests and update GPU metrics."""
    active_requests.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests.dec()
        # Periodically update GPU metrics
        monitor.update_gpu_metrics()

# Update inference endpoints to record metrics
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize dialogue with monitoring."""
    start_time = time.time()
    status = "success"
    error = None
    error_type = None
    output_length = 0
    
    try:
        summary = summarize_dialogue(request.dialogue, _model, _tokenizer)
        output_length = len(summary.split())
    except ValueError as e:
        status = "error"
        error = str(e)
        error_type = "validation_error"
        logger.error(f"Validation error: {error}")
        raise HTTPException(status_code=400, detail=error)
    except RuntimeError as e:
        status = "error"
        error = str(e)
        error_type = "runtime_error"
        logger.error(f"Runtime error: {error}")
        raise HTTPException(status_code=500, detail=error)
    except Exception as e:
        status = "error"
        error = str(e)
        error_type = "unknown_error"
        logger.error(f"Inference failed: {error}")
        raise HTTPException(status_code=500, detail=error)
    finally:
        duration_ms = (time.time() - start_time) * 1000
        monitor.record_inference(
            duration_ms=duration_ms,
            output_length=output_length,
            token_count=len(summary.split()) if status == "success" else 0,
            status=status,
            error=error,
            error_type=error_type
        )
    
    return SummarizeResponse(summary=summary, id=str(uuid.uuid4()))
```

#### Step 3.1: Secure the /metrics Endpoint

**⚠️ Security Warning:** The `/metrics` endpoint exposes sensitive system information and should be protected in production.

**Security Risks of Unprotected `//metrics`:**
- System architecture and infrastructure details
- Performance bottlenecks and vulnerabilities
- Request rates and usage patterns
- Memory/CPU consumption patterns
- Error rates and error types
- Technology stack information

**Recommended Protection Strategy: Network Isolation + API Key Authentication**

Update `api/main.py` to add configurable authentication:

```python
from fastapi import Security, HTTPException, Request, Depends, Header
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.monitoring import ModelMonitor, active_requests
from fastapi.responses import Response
import os
import time

# Initialize monitoring
monitor = ModelMonitor(model_id=os.getenv("MODEL_ID", "google/flan-t5-base"))

# Metrics endpoint security configuration
METRICS_API_KEY = os.getenv("METRICS_API_KEY")
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
METRICS_REQUIRE_AUTH = os.getenv("METRICS_REQUIRE_AUTH", "true").lower() == "true"

api_key_header = APIKeyHeader(name="X-Metrics-Key", auto_error=False)

async def verify_metrics_access(
    request: Request,
    api_key: str = Security(api_key_header),
    authorization: str = Header(None)
):
    """Verify metrics access - supports both X-Metrics-Key and Bearer token."""
    if not PROMETHEUS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    # If auth not required, allow access
    if not METRICS_REQUIRE_AUTH:
        return True
    
    # Check if API key is configured
    if not METRICS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Metrics auth enabled but METRICS_API_KEY not set"
        )
    
    # Try Bearer token first (Prometheus standard)
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        if token == METRICS_API_KEY:
            return True
    
    # Try X-Metrics-Key header (custom, for manual testing)
    if api_key and api_key == METRICS_API_KEY:
        return True
    
    # No valid auth found
    raise HTTPException(
        status_code=403,
        detail="Invalid or missing metrics API key"
    )

# Set model info on startup
@app.on_event("startup")
async def startup_monitoring():
    """Initialize monitoring on startup."""
    monitor.set_model_info({
        'model_id': os.getenv("MODEL_ID", "google/flan-t5-base"),
        'version': '1.0.0',
        'lora_merged': str(os.getenv("PEFT_WEIGHTS_PATH") is not None)
    })

# Protected metrics endpoint
@app.get("/metrics")
async def metrics(authorized: bool = Depends(verify_metrics_access)):
    """Expose Prometheus metrics (protected).
    
    Accepts authentication via:
    - Authorization: Bearer <token> header (Prometheus standard)
    - X-Metrics-Key: <token> header (for manual testing with curl)
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Middleware to track active requests
@app.middleware("http")
async def track_active_requests(request: Request, call_next):
    """Track active requests and update GPU metrics."""
    active_requests.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        active_requests.dec()
        # Periodically update GPU metrics
        monitor.update_gpu_metrics()

# Update inference endpoints to record metrics
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize dialogue with monitoring."""
    start_time = time.time()
    status = "success"
    error = None
    error_type = None
    output_length = 0
    
    try:
        summary = summarize_dialogue(request.dialogue, _model, _tokenizer)
        output_length = len(summary.split())
    except ValueError as e:
        status = "error"
        error = str(e)
        error_type = "validation_error"
        logger.error(f"Validation error: {error}")
        raise HTTPException(status_code=400, detail=error)
    except RuntimeError as e:
        status = "error"
        error = str(e)
        error_type = "runtime_error"
        logger.error(f"Runtime error: {error}")
        raise HTTPException(status_code=500, detail=error)
    except Exception as e:
        status = "error"
        error = str(e)
        error_type = "unknown_error"
        logger.error(f"Inference failed: {error}")
        raise HTTPException(status_code=500, detail=error)
    finally:
        duration_ms = (time.time() - start_time) * 1000
        monitor.record_inference(
            duration_ms=duration_ms,
            output_length=output_length,
            token_count=len(summary.split()) if status == "success" else 0,
            status=status,
            error=error,
            error_type=error_type
        )
    
    return SummarizeResponse(summary=summary, id=str(uuid.uuid4()))
```

**Environment Configuration:**

Update `.env` file:

```bash
# Metrics Security
PROMETHEUS_ENABLED=true
METRICS_REQUIRE_AUTH=true  # Set to false for local dev only
METRICS_API_KEY=your-random-secret-key-here  # Generate with: openssl rand -hex 32 (PowerShell: -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_}))
```

**Configure Prometheus to Use API Key:**

Update `prometheus.yml` to use Bearer token authentication (Prometheus standard):

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Load alert rules
rule_files:
  - 'alerts.yml'

scrape_configs:
  - job_name: 'flan-t5-api'
    static_configs:
      - targets: ['host.docker.internal:8000']  # API running on host
    metrics_path: '/metrics'
    # Add Bearer token authentication (standard Prometheus method)
    authorization:
      type: Bearer
      credentials: 'your-random-secret-key-here'  # Same as METRICS_API_KEY
```

**Note:** The API now accepts both `Authorization: Bearer <token>` (Prometheus standard) and `X-Metrics-Key` header for flexibility.

**Alternative: Network-Level Isolation (Production Best Practice)**

When running in Docker Compose, keep the API and Prometheus on the same private network without exposing the metrics port publicly:

```yaml
# docker-compose.yml (production setup)
version: '3.8'

services:
  api:
    build: .
    environment:
      - PROMETHEUS_ENABLED=true
      - METRICS_REQUIRE_AUTH=false  # Can disable auth when using network isolation
    networks:
      - monitoring  # Private network
    ports:
      - "8000:8000"  # Only expose main API publicly
    # Port 8000/metrics NOT publicly exposed, only accessible within docker network
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - monitoring  # Can access api:8000/metrics internally
    # Only expose if you need external access (not recommended)
    # ports:
    #   - "9090:9090"

networks:
  monitoring:
    driver: bridge
```

With this setup, Prometheus accesses `http://api:8000/metrics` internally without public exposure.

**Security Best Practices:**

1. ✅ **Development:** Set `METRICS_REQUIRE_AUTH=false` for easy testing
2. ✅ **Production:** Always set `METRICS_REQUIRE_AUTH=true` and use strong API keys
3. ✅ **Defense in Depth:** Use both network isolation AND authentication
4. ✅ **Key Rotation:** Rotate `METRICS_API_KEY` periodically
5. ✅ **Secrets Management:** Store keys in AWS Secrets Manager or environment variables (never in code)

#### Step 4: Setup CloudWatch Integration (Optional)

Create file: `scripts/setup_monitoring.py`

```python
"""
Setup CloudWatch dashboard for model monitoring.
"""

import boto3
import json

def create_cloudwatch_dashboard():
    """Create dashboard for FlanT5 monitoring."""
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["FlanT5DialogSum", "InferenceDuration", {"stat": "Average"}],
                        ["...", {"stat": "p99"}]
                    ],
                    "period": 60,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Inference Latency"
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["FlanT5DialogSum", "OutputLength", {"stat": "Average"}]
                    ],
                    "period": 60,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "Output Length"
                }
            }
        ]
    }
    
    cloudwatch.put_dashboard(
        DashboardName="FlanT5DialogSumMonitoring",
        DashboardBody=json.dumps(dashboard_body)
    )
    print("CloudWatch dashboard created successfully!")

if __name__ == "__main__":
    create_cloudwatch_dashboard()
```

#### Step 5: Setup Grafana and Prometheus Stack

Grafana provides beautiful, interactive dashboards for Prometheus metrics.

**Two Deployment Options:**

1. **Option A: Fully Containerized** (Recommended) - All services in Docker
2. **Option B: Hybrid Setup** - API on host, monitoring in Docker

---

**Option A: Fully Containerized Setup (Recommended)**

This approach containerizes **everything** (API + Monitoring) in a single Docker Compose stack.

✅ **Advantages:**
- Complete isolation and reproducibility
- Easy deployment to any environment
- Consistent networking between services
- Production-ready architecture
- Simple scaling with Docker Swarm/Kubernetes

**See [DOCKER_SETUP.md](DOCKER_SETUP.md) for complete containerized setup guide.**

**Key files created:**
- `docker-compose.yml` - All services (API, Prometheus, Grafana, AlertManager)
- `prometheus.yml` - Targets containerized API at `api:8000`
- `.env` - Environment variables for all services

**Quick start:**
```powershell
# Copy environment template
Copy-Item .env.example .env

# Edit .env with your METRICS_API_KEY

# Start all services
docker-compose up -d

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

---

**Option B: Hybrid Setup (API on Host, Monitoring in Docker)**

Use this if you want to run your API directly on the host (for development/debugging) while keeping monitoring in containers.

Create file: `docker-compose.grafana.yml`

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus-alerts.yml:/etc/prometheus/alerts.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - monitoring
    depends_on:
      - alertmanager

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager-data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    networks:
      - monitoring

volumes:
  prometheus-data:
  alertmanager-data:
  grafana-data:

networks:
  monitoring:
```

Create file: `prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# Load alert rules
rule_files:
  - 'alerts.yml'

scrape_configs:
  - job_name: 'flan-t5-api'
    static_configs:
      - targets: ['host.docker.internal:8000']  # API running on host
    metrics_path: '/metrics'
    
    # Optional: Add authentication if METRICS_REQUIRE_AUTH=true (see Step 3.1)
    # Uncomment below if using API key authentication:
    # authorization:
    #   type: Bearer
    #   credentials: 'your-random-secret-key-here'  # Must match METRICS_API_KEY
```

Create file: `prometheus-alerts.yml`

```yaml
groups:
  - name: model_performance
    interval: 30s
    rules:
      # High latency alert
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile latency is {{ $value }}s (threshold: 5s)"

      # Error rate alert
      - alert: HighErrorRate
        expr: rate(error_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
          component: model
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # GPU memory alert
      - alert: HighGPUUtilization
        expr: gpu_utilization_percent > 90
        for: 10m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "GPU utilization is high"
          description: "GPU {{ $labels.device }} at {{ $value }}%"

      # Memory usage alert
      - alert: HighMemoryUsage
        expr: model_memory_usage_mb > 4096
        for: 5m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}MB (threshold: 4096MB)"

      # Active requests piling up
      - alert: RequestQueueBacklog
        expr: active_requests > 10
        for: 3m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "Request queue backlog detected"
          description: "{{ $value }} requests are currently being processed"

      # Model endpoint down
      - alert: ModelEndpointDown
        expr: up{job="flan-t5-api"} == 0
        for: 1m
        labels:
          severity: critical
          component: infrastructure
        annotations:
          summary: "Model API endpoint is down"
          description: "The /metrics endpoint is not responding"
```

Create file: `alertmanager.yml`

```yaml
global:
  # Slack webhook (replace with your webhook URL)
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

# Route alerts to different receivers based on severity
route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  
  routes:
    # Critical alerts go to PagerDuty and Slack
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    
    # All critical alerts also to Slack
    - match:
        severity: critical
      receiver: 'slack-critical'
    
    # Warning alerts only to Slack
    - match:
        severity: warning
      receiver: 'slack-warnings'

# Receivers - notification channels
receivers:
  - name: 'default'
    slack_configs:
      - channel: '#ml-monitoring'
        title: 'FLAN-T5 Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'

  - name: 'slack-critical'
    slack_configs:
      - channel: '#ml-alerts-critical'
        title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#ml-alerts-warnings'
        title: '⚠️ WARNING: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  # Email notifications (optional)
  - name: 'email-team'
    email_configs:
      - to: 'ml-team@company.com'
        from: 'alertmanager@company.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'
        headers:
          Subject: 'FLAN-T5 Alert: {{ .GroupLabels.alertname }}'

# Inhibition rules - suppress certain alerts when others are firing
inhibit_rules:
  # If endpoint is down, don't alert on high latency/errors
  - source_match:
      alertname: 'ModelEndpointDown'
    target_match_re:
      alertname: '(HighInferenceLatency|HighErrorRate)'
    equal: ['job']
```

**To configure notifications:**

1. **Slack:** Get webhook URL from Slack → Apps → Incoming Webhooks
2. **PagerDuty:** Get service key from PagerDuty → Services → Integrations
3. **Email:** Use SMTP credentials (Gmail app password recommended)

Create file: `grafana/provisioning/datasources/prometheus.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

Create file: `grafana/provisioning/dashboards/dashboard.yml`

```yaml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

Create file: `grafana/dashboards/flan-t5-monitoring.json`

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {
              "tooltip": false,
              "viz": false,
              "legend": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": true
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 5000
              }
            ]
          },
          "unit": "ms"
        }
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 2,
      "options": {
        "legend": {
          "calcs": ["mean", "max"],
          "displayMode": "table",
          "placement": "bottom"
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "rate(inference_duration_seconds_sum[5m]) / rate(inference_duration_seconds_count[5m]) * 1000",
          "legendFormat": "Average Latency",
          "refId": "A"
        }
      ],
      "title": "Inference Latency (ms)",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "short"
        }
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "id": 3,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "values": false,
          "calcs": ["lastNotNull"],
          "fields": ""
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "sum(rate(inference_total[5m]))",
          "legendFormat": "Total Requests/sec",
          "refId": "A"
        }
      ],
      "title": "Throughput (requests/sec)",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "bars",
            "fillOpacity": 100,
            "gradientMode": "none",
            "hideFrom": {
              "tooltip": false,
              "viz": false,
              "legend": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": true
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          }
        }
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 8
      },
      "id": 4,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom"
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "sum by(status) (rate(inference_total[5m]))",
          "legendFormat": "{{status}}",
          "refId": "A"
        }
      ],
      "title": "Requests by Status",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "yellow",
                "value": 0.5
              },
              {
                "color": "red",
                "value": 0.8
              }
            ]
          },
          "unit": "percentunit"
        }
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 8
      },
      "id": 5,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "values": false,
          "calcs": ["lastNotNull"],
          "fields": ""
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "data_drift_score",
          "legendFormat": "Drift Score",
          "refId": "A"
        }
      ],
      "title": "Data Drift Score",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "red",
                "value": 4096
              }
            ]
          },
          "unit": "decmbytes"
        }
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 16
      },
      "id": 6,
      "options": {
        "orientation": "horizontal",
        "reduceOptions": {
          "values": false,
          "calcs": ["lastNotNull"],
          "fields": ""
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "model_memory_usage_mb",
          "legendFormat": "Memory Usage (MB)",
          "refId": "A"
        }
      ],
      "title": "Model Memory Usage",
      "type": "gauge"
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": ["ml", "inference", "flan-t5"],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "FLAN-T5 DialogSum Monitoring",
  "uid": "flan_t5_monitoring",
  "version": 0
}
```

#### Step 6: Start and Test the Monitoring Stack

**Start all services:**

```bash
# Start Prometheus, AlertManager, and Grafana
docker-compose -f docker-compose.grafana.yml up -d

# Check services are running
docker-compose -f docker-compose.grafana.yml ps

# View logs
docker-compose -f docker-compose.grafana.yml logs -f
```

**Access the dashboards:**

- **Grafana:** http://localhost:3000 (admin/admin)
- **Prometheus:** http://localhost:9090
- **AlertManager:** http://localhost:9093

**Test the setup:**

```bash
# 1. Start your API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 2. Check metrics endpoint
curl http://localhost:8000/metrics

# 3. Generate some traffic
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "John: Hey! How are you?\nMary: Great, thanks!"}'

# 4. Check Prometheus is scraping metrics
# Open http://localhost:9090/targets - should show "flan-t5-api" as UP

# 5. Verify alerts are loaded
# Open http://localhost:9090/alerts - should show all alert rules

# 6. Check AlertManager
# Open http://localhost:9093 - should show no active alerts (if all is healthy)

# 7. Test an alert by triggering high latency (optional)
# Run load test to trigger HighInferenceLatency alert
```

**Configure Slack notifications:**

1. Create a Slack webhook:
   - Go to https://api.slack.com/apps
   - Create new app → Incoming Webhooks → Add New Webhook
   - Copy webhook URL

2. Update `alertmanager.yml`:
   ```yaml
   global:
     slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
   ```

3. Restart AlertManager:
   ```bash
   docker-compose -f docker-compose.grafana.yml restart alertmanager
   ```

**Monitoring Architecture:**

```
API (:8000) → /metrics endpoint
     ↓
Prometheus (:9090) → scrapes every 15s
     ↓                 evaluates alert rules
     ├→ Grafana (:3000) → visualizations
     └→ AlertManager (:9093) → Slack/Email/PagerDuty
```

---

**Option C: AWS Managed Grafana (Enterprise)**

For enterprise deployments, use AWS Managed Grafana with managed Prometheus.

Create file: `scripts/setup_aws_grafana.py`

```python
"""
Setup AWS Managed Grafana workspace.
"""

import boto3
import json

def create_grafana_workspace():
    """Create AWS Managed Grafana workspace."""
    grafana = boto3.client('grafana')
    
    response = grafana.create_workspace(
        accountAccessType='CURRENT_ACCOUNT',
        authenticationProviders=['AWS_SSO'],
        permissionType='SERVICE_MANAGED',
        workspaceDataSources=['PROMETHEUS', 'CLOUDWATCH'],
        workspaceName='flan-t5-monitoring',
        workspaceDescription='Monitoring for FLAN-T5 DialogSum model',
        workspaceRoleArn='arn:aws:iam::ACCOUNT_ID:role/grafana-role',
        tags={
            'Project': 'flan-t5-dialogsum',
            'Environment': 'production'
        }
    )
    
    workspace_id = response['workspace']['id']
    workspace_endpoint = response['workspace']['endpoint']
    
    print(f"Grafana workspace created!")
    print(f"Workspace ID: {workspace_id}")
    print(f"Endpoint: https://{workspace_endpoint}")
    
    return workspace_id


def configure_prometheus_datasource(workspace_id: str):
    """Configure Prometheus as data source."""
    grafana = boto3.client('grafana')
    
    # Update workspace to include Prometheus
    response = grafana.update_workspace_configuration(
        workspaceId=workspace_id,
        configuration=json.dumps({
            "datasources": {
                "prometheus": {
                    "url": "http://your-prometheus-endpoint:9090",
                    "access": "proxy"
                }
            }
        })
    )
    
    print("Prometheus datasource configured!")


if __name__ == "__main__":
    workspace_id = create_grafana_workspace()
    configure_prometheus_datasource(workspace_id)
```

**Grafana Dashboard Features:**

1. **Inference Latency** - Real-time latency tracking with P50, P95, P99
2. **Throughput** - Requests per second gauge
3. **Request Status** - Success vs error rates
4. **Data Drift Score** - Visual drift indicator
5. **Memory Usage** - Model memory consumption
6. **Custom Alerts** - Configure alerts for thresholds

**Setup Alerts in Grafana:**

```yaml
# grafana/provisioning/alerting/alerts.yml
apiVersion: 1

groups:
  - name: model_alerts
    interval: 1m
    rules:
      - alert: HighLatency
        expr: rate(inference_duration_seconds_sum[5m]) / rate(inference_duration_seconds_count[5m]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "Average latency is {{ $value }}s"
      
      - alert: DataDrift
        expr: data_drift_score > 0.7
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Data drift detected"
          description: "Drift score is {{ $value }}"
      
      - alert: HighMemoryUsage
        expr: model_memory_usage_mb > 4096
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Model using {{ $value }}MB"
```

#### Step 7: Production Deployment Checklist

**Before deploying to production:**

1. ✅ **Generate strong API key:**
   ```powershell
   # PowerShell
   -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
   
   # Or use openssl (Git Bash/WSL)
   openssl rand -hex 32
   ```

2. ✅ **Configure environment variables:**
   ```bash
   # .env (production)
   PROMETHEUS_ENABLED=true
   METRICS_REQUIRE_AUTH=true  # MUST be true in production
   METRICS_API_KEY=<your-generated-key>
   CLOUDWATCH_ENABLED=true  # If using AWS
   SNS_TOPIC_ARN=arn:aws:sns:us-east-1:ACCOUNT_ID:model-alerts
   ```

3. ✅ **Update Prometheus config with API key:**
   - Edit `prometheus.yml`
   - Add your key to `authorization.credentials`

4. ✅ **Configure alerting:**
   - Set up Slack webhook in `alertmanager.yml`
   - Test alerts with sample traffic

5. ✅ **Set up dashboard access:**
   - Change Grafana admin password (default: admin/admin)
   - Configure SSO if needed

6. ✅ **Network security:**
   - Use network isolation in Docker
   - Don't expose Prometheus/Grafana ports publicly
   - Use reverse proxy with HTTPS for Grafana

7. ✅ **Test the full pipeline:**
   ```bash
   # Verify metrics endpoint requires auth
   curl http://localhost:8000/metrics  # Should return 403
   
   # Test with valid auth
   curl -H "Authorization: Bearer YOUR_KEY" http://localhost:8000/metrics  # Should work
   
   # Verify Prometheus can scrape
   # Check http://localhost:9090/targets
   ```

#### Step 8: Monitoring and Troubleshooting

**Common Issues:**

**Problem: Prometheus shows "DOWN" for flan-t5-api target**
```
Solution:
1. Check API is running: curl http://localhost:8000/health
2. Verify metrics endpoint: curl http://localhost:8000/metrics
3. Check Prometheus config target address
4. If using Docker: ensure 'host.docker.internal:8000' is correct
   - Linux: Use 'host.docker.internal' or API container name
   - Windows/Mac: 'host.docker.internal' should work
```

**Problem: 403 Forbidden on /metrics**
```
Solution:
1. Verify METRICS_API_KEY is set in .env
2. Check Prometheus authorization config matches API key
3. Test manually: curl -H "Authorization: Bearer YOUR_KEY" http://localhost:8000/metrics
```

**Problem: No data in Grafana dashboards**
```
Solution:
1. Verify Prometheus is scraping: http://localhost:9090/targets
2. Check Prometheus has data: http://localhost:9090/graph
   - Query: inference_total
3. Generate test traffic to API
4. Verify Grafana datasource connection
```

**Problem: Alerts not triggering**
```
Solution:
1. Check AlertManager is running: http://localhost:9093
2. Verify alert rules loaded: http://localhost:9090/alerts
3. Check alert thresholds are realistic
4. Test with artificial load to trigger alerts
```

**Health Check Commands:**
```bash
# Check all services
docker-compose -f docker-compose.grafana.yml ps

# View logs
docker-compose -f docker-compose.grafana.yml logs prometheus
docker-compose -f docker-compose.grafana.yml logs grafana
docker-compose -f docker-compose.grafana.yml logs alertmanager

# Restart services
docker-compose -f docker-compose.grafana.yml restart

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job:.labels.job, health:.health}'
```

**Monitoring Best Practices:**

1. **Set realistic alert thresholds** based on your SLA requirements
2. **Configure multiple notification channels** (Slack + Email + PagerDuty)
3. **Monitor the monitoring** - set up alerts if Prometheus/Grafana go down
4. **Regular maintenance:**
   - Review and adjust alert rules monthly
   - Clean up old metrics data (Prometheus retention)
   - Update dashboards based on actual usage patterns
5. **Document your alerts** - add context to help on-call engineers

#### Step 9: Environment Variables Reference

Update `.env.example` with all monitoring variables:

```bash
# ============================================
# Monitoring Configuration
# ============================================

# Enable/disable Prometheus metrics
PROMETHEUS_ENABLED=true

# Require API key for /metrics endpoint (MUST be true in production)
METRICS_REQUIRE_AUTH=true

# API key for metrics authentication (generate with: openssl rand -hex 32)
# Used for both X-Metrics-Key header and Bearer token
METRICS_API_KEY=your-secret-key-here

# CloudWatch Integration (optional)
CLOUDWATCH_ENABLED=false
AWS_REGION=us-east-1
SNS_TOPIC_ARN=arn:aws:sns:us-east-1:ACCOUNT_ID:model-alerts

# Grafana Configuration (optional - for AWS Managed Grafana)
GRAFANA_ENABLED=false
GRAFANA_URL=https://your-workspace.grafana-workspace.us-east-1.amazonaws.com
GRAFANA_API_KEY=your_grafana_api_key
```

---

### 1.2.1 Summary: What You've Built

You now have a **production-ready monitoring stack**:

✅ **Infrastructure Metrics** (Prometheus)
- Real-time latency tracking (P50, P95, P99)
- Request throughput and error rates
- GPU utilization and memory usage
- Active request tracking

✅ **Visualization** (Grafana)
- Interactive dashboards
- Custom metrics views
- Historical data analysis

✅ **Alerting** (AlertManager)
- Multi-channel notifications (Slack, Email, PagerDuty)
- Severity-based routing
- Alert inhibition rules

✅ **Security**
- Bearer token authentication
- Network isolation
- Configurable access control

✅ **Integration**
- CloudWatch (optional)
- Structured JSON logging
- AWS SNS alerts

**Next Steps:** Add Evidently AI (Section 1.3) for ML-specific monitoring (data drift, model performance)

---

## 1.3 Additional Implementation: Evidently AI (Recommended for Production)

**⚠️ Important: This is COMPLEMENTARY to Prometheus/Grafana, not a replacement!**

### Why Add Evidently AI?

Prometheus/Grafana (Section 1.2) provides **infrastructure monitoring**:
- ✅ Real-time latency and throughput metrics
- ✅ System health monitoring
- ✅ Alerting on performance issues
- ✅ Operational dashboards

Evidently AI provides **model quality monitoring**:
- ✅ Statistical data drift detection
- ✅ Model performance degradation tracking
- ✅ Data quality validation
- ✅ ML-specific metrics and reports

**Production Best Practice: Use BOTH together!**
- **Prometheus/Grafana** → Monitor system performance (latency, throughput, errors)
- **Evidently AI** → Monitor model behavior (drift, quality, predictions)

### Add Evidently if you need:
- 🎯 Production-grade drift detection with statistical rigor
- 📊 ML-specific HTML reports for stakeholders
- 🧮 Advanced statistical tests (KL divergence, Wasserstein distance)
- 🏢 Model performance tracking over time
- 📈 Automated data quality validation

### Drift Detection: Custom vs Evidently AI

| Feature | Custom (Section 1.2) | Evidently AI (This Section) |
|---------|----------------------|----------------------------|
| **Drift Detection** | Simple threshold-based | Multiple statistical methods (KS test, Jensen-Shannon, etc.) |
| **Dependencies** | ~3 packages | ~10 packages (~50MB) |
| **Learning Curve** | Quick (1-2 hours) | Moderate (3-4 hours) |
| **AWS Integration** | Native CloudWatch | Manual setup needed |
| **Reports** | Prometheus metrics | HTML dashboards + JSON/PDF reports |
| **Statistical Rigor** | Basic | Advanced (hypothesis testing, p-values) |
| **Use Case** | Quick drift indicator | Compliance, audits, detailed analysis |
| **Best For** | Lightweight monitoring | Enterprise, regulated industries |

**Recommendation:** Start with custom drift detection (Section 1.2), add Evidently AI when you need detailed reports or regulatory compliance.

### Step 1: Install Evidently

```bash
pip install evidently boto3
```

Update `requirements.txt`:
```
evidently==0.4.9
boto3==1.34.0
```

### Step 2: Create Evidently Monitoring Module

Create file: `src/evidently_monitoring.py`

```python
"""
Production-grade model monitoring using Evidently AI.
Monitors ML-specific metrics: data drift, prediction drift, model performance.
Does NOT monitor infrastructure metrics (use Prometheus for that).
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.metrics import (
    DataDriftTable,
    ColumnDriftMetric,
    DatasetDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)
import boto3

logger = logging.getLogger(__name__)


class EvidentlyMonitor:
    """Monitor model performance and data quality using Evidently."""
    
    def __init__(
        self,
        project_name: str = "flan-t5-dialogsum",
        s3_bucket: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None
    ):
        self.project_name = project_name
        self.s3_bucket = s3_bucket
        self.reference_data = reference_data or pd.DataFrame()
        
        if s3_bucket:
            self.s3_client = boto3.client('s3')
        
        # Initialize inference logs
        self.inference_logs = []
    
    def log_inference(
        self,
        dialogue: str,
        summary: str,
        reference_summary: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """
        Log inference with ML-specific features for monitoring.
        
        Args:
            dialogue: Input dialogue text
            summary: Generated summary
            reference_summary: Ground truth summary (if available) for performance tracking
            timestamp: Optional timestamp
        """
        # Extract text features for drift detection
        dialogue_features = self._extract_text_features(dialogue)
        summary_features = self._extract_text_features(summary, prefix="summary_")
        
        log_entry = {
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "dialogue": dialogue,
            "summary": summary,
            "reference_summary": reference_summary,
            
            # Input text features (for data drift detection)
            "dialogue_word_count": dialogue_features["word_count"],
            "dialogue_char_count": dialogue_features["char_count"],
            "dialogue_sentence_count": dialogue_features["sentence_count"],
            "dialogue_avg_word_length": dialogue_features["avg_word_length"],
            "dialogue_question_count": dialogue_features["question_count"],
            "dialogue_exclamation_count": dialogue_features["exclamation_count"],
            "dialogue_unique_words": dialogue_features["unique_words"],
            "dialogue_lexical_diversity": dialogue_features["lexical_diversity"],
            
            # Prediction features (for prediction drift detection)
            "summary_word_count": summary_features["word_count"],
            "summary_char_count": summary_features["char_count"],
            "summary_sentence_count": summary_features["sentence_count"],
            "summary_avg_word_length": summary_features["avg_word_length"],
            "summary_unique_words": summary_features["unique_words"],
            "summary_lexical_diversity": summary_features["lexical_diversity"],
            
            # Compression ratio (model behavior indicator)
            "compression_ratio": dialogue_features["word_count"] / max(summary_features["word_count"], 1),
        }
        
        # Add performance metrics if reference summary available
        if reference_summary:
            log_entry["rouge_1"] = self._calculate_rouge(summary, reference_summary, n=1)
            log_entry["rouge_2"] = self._calculate_rouge(summary, reference_summary, n=2)
            log_entry["rouge_l"] = self._calculate_rouge_l(summary, reference_summary)
        
        self.inference_logs.append(log_entry)
        logger.info(f"Inference logged with {len(dialogue_features)} text features")
    
    def _extract_text_features(self, text: str, prefix: str = "") -> Dict[str, float]:
        """Extract statistical features from text for drift detection."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        unique_words = len(set(words))
        lexical_diversity = unique_words / max(word_count, 1)
        
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "question_count": question_count,
            "exclamation_count": exclamation_count,
            "unique_words": unique_words,
            "lexical_diversity": lexical_diversity,
        }
    
    def _calculate_rouge(self, prediction: str, reference: str, n: int = 1) -> float:
        """Calculate ROUGE-N score (simplified implementation)."""
        try:
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            if n == 1:
                pred_ngrams = set(pred_words)
                ref_ngrams = set(ref_words)
            else:
                pred_ngrams = set(zip(*[pred_words[i:] for i in range(n)]))
                ref_ngrams = set(zip(*[ref_words[i:] for i in range(n)]))
            
            if not ref_ngrams:
                return 0.0
            
            overlap = len(pred_ngrams & ref_ngrams)
            recall = overlap / len(ref_ngrams)
            return recall
        except Exception as e:
            logger.warning(f"Error calculating ROUGE-{n}: {e}")
            return 0.0
    
    def _calculate_rouge_l(self, prediction: str, reference: str) -> float:
        """Calculate ROUGE-L score using LCS (simplified)."""
        try:
            pred_words = prediction.lower().split()
            ref_words = reference.lower().split()
            
            # Simple LCS calculation
            m, n = len(pred_words), len(ref_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if pred_words[i-1] == ref_words[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcs_length = dp[m][n]
            recall = lcs_length / max(n, 1)
            return recall
        except Exception as e:
            logger.warning(f"Error calculating ROUGE-L: {e}")
            return 0.0
    
    def convert_logs_to_dataframe(self) -> pd.DataFrame:
        """Convert inference logs to DataFrame for Evidently."""
        if not self.inference_logs:
            logger.warning("No inference logs available")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.inference_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        numerical_features: List[str] = None,
        text_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift in input text characteristics.
        Monitors changes in dialogue patterns, not infrastructure metrics.
        
        Args:
            current_data: DataFrame with current inference data
            numerical_features: Text feature names (word counts, sentence counts, etc.)
            text_columns: Raw text columns for text-specific analysis
        
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data.empty or current_data.empty:
            logger.warning("Reference or current data is empty")
            return {}
        
        # Text features for drift detection (NOT infrastructure metrics)
        if numerical_features is None:
            numerical_features = [
                'dialogue_word_count',
                'dialogue_sentence_count', 
                'dialogue_avg_word_length',
                'dialogue_lexical_diversity',
                'dialogue_question_count',
                'dialogue_unique_words',
            ]
        
        if text_columns is None:
            text_columns = ['dialogue']
        
        try:
            logger.info(f"Running DATA DRIFT detection on text features: {numerical_features}")
            
            # Create data drift report for text features
            report = Report(metrics=[
                DataDriftPreset(columns=numerical_features),
                TextOverviewPreset(column_name='dialogue') if 'dialogue' in current_data.columns else None,
            ])
            
            report.run(
                reference_data=self.reference_data[numerical_features + text_columns],
                current_data=current_data[numerical_features + text_columns]
            )
            
            # Extract metrics
            report_dict = report.as_dict()
            drift_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "drift_type": "data_drift",
                "description": "Input text characteristics changed",
                "metrics": report_dict.get('metrics', []),
                "drift_detected": self._check_drift_threshold(report_dict),
                "features_monitored": numerical_features
            }
            
            logger.info(f"Data drift detection complete: {drift_results['drift_detected']}")
            return drift_results
        
        except Exception as e:
            logger.error(f"Error in data drift detection: {e}")
            return {}
    
    def detect_prediction_drift(
        self,
        current_data: pd.DataFrame,
        prediction_features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect prediction drift - changes in generated summary characteristics.
        
        Args:
            current_data: DataFrame with current inference data
            prediction_features: Summary feature names
        
        Returns:
            Dictionary with prediction drift results
        """
        if self.reference_data.empty or current_data.empty:
            logger.warning("Reference or current data is empty")
            return {}
        
        if prediction_features is None:
            prediction_features = [
                'summary_word_count',
                'summary_sentence_count',
                'summary_lexical_diversity',
                'summary_unique_words',
                'compression_ratio',
            ]
        
        try:
            logger.info(f"Running PREDICTION DRIFT detection on summary features: {prediction_features}")
            
            report = Report(metrics=[
                DataDriftPreset(columns=prediction_features)
            ])
            
            report.run(
                reference_data=self.reference_data[prediction_features],
                current_data=current_data[prediction_features]
            )
            
            report_dict = report.as_dict()
            drift_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "drift_type": "prediction_drift",
                "description": "Generated summary characteristics changed",
                "metrics": report_dict.get('metrics', []),
                "drift_detected": self._check_drift_threshold(report_dict),
                "features_monitored": prediction_features
            }
            
            logger.info(f"Prediction drift detection complete: {drift_results['drift_detected']}")
            return drift_results
        
        except Exception as e:
            logger.error(f"Error in prediction drift detection: {e}")
            return {}
    
    def monitor_model_performance(
        self,
        current_data: pd.DataFrame,
        performance_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Monitor model performance degradation using ROUGE scores.
        Only works if reference summaries are available.
        
        Args:
            current_data: DataFrame with current inference data (must include ROUGE scores)
            performance_metrics: Metric column names
        
        Returns:
            Dictionary with performance monitoring results
        """
        if self.reference_data.empty or current_data.empty:
            logger.warning("Reference or current data is empty")
            return {}
        
        if performance_metrics is None:
            performance_metrics = ['rouge_1', 'rouge_2', 'rouge_l']
        
        # Check if performance metrics are available
        available_metrics = [m for m in performance_metrics if m in current_data.columns]
        
        if not available_metrics:
            logger.warning("No performance metrics (ROUGE scores) available. Need reference summaries.")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "drift_type": "model_performance",
                "warning": "No reference summaries available for performance tracking",
                "recommendation": "Collect ground truth summaries to enable performance monitoring"
            }
        
        try:
            logger.info(f"Running MODEL PERFORMANCE monitoring on: {available_metrics}")
            
            # Calculate performance degradation
            baseline_perf = self.reference_data[available_metrics].mean()
            current_perf = current_data[available_metrics].mean()
            
            degradation = {}
            performance_dropped = False
            
            for metric in available_metrics:
                baseline_val = baseline_perf[metric]
                current_val = current_perf[metric]
                pct_change = ((current_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                
                degradation[metric] = {
                    "baseline": float(baseline_val),
                    "current": float(current_val),
                    "change_pct": float(pct_change),
                    "degraded": pct_change < -10  # >10% drop is degradation
                }
                
                if pct_change < -10:
                    performance_dropped = True
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "drift_type": "model_performance",
                "description": "Model quality degradation detected" if performance_dropped else "Model performance stable",
                "degradation_detected": performance_dropped,
                "metrics": degradation,
                "recommendation": "Consider model retraining" if performance_dropped else "No action needed"
            }
        
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}")
            return {}
    
    def run_data_quality_tests(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Run data quality tests on ML features (NOT infrastructure metrics).
        
        Args:
            current_data: DataFrame with current inference data
            threshold: Drift threshold (0-1)
        
        Returns:
            Dictionary with test results
        """
        if self.reference_data.empty or current_data.empty:
            logger.warning("Reference or current data is empty")
            return {}
        
        try:
            logger.info("Running DATA QUALITY tests on text features...")
            
            # Test for data quality issues
            text_features = [
                'dialogue_word_count',
                'dialogue_sentence_count',
                'summary_word_count',
                'summary_sentence_count',
            ]
            
            suite = TestSuite(tests=[
                TestNumberOfDriftedColumns(
                    eq=0,  # Ideally no text feature drift
                ),
                TestShareOfDriftedColumns(
                    lt=threshold  # Less than 30% of features should drift
                ),
            ])
            
            suite.run(
                reference_data=self.reference_data[text_features],
                current_data=current_data[text_features]
            )
            
            results = suite.as_dict()
            
            # Manual data quality checks
            quality_issues = []
            
            # Check for missing or empty text
            if 'dialogue' in current_data.columns:
                empty_dialogues = current_data['dialogue'].isna().sum() + (current_data['dialogue'] == '').sum()
                if empty_dialogues > 0:
                    quality_issues.append(f"Found {empty_dialogues} empty dialogues")
            
            if 'summary' in current_data.columns:
                empty_summaries = current_data['summary'].isna().sum() + (current_data['summary'] == '').sum()
                if empty_summaries > 0:
                    quality_issues.append(f"Found {empty_summaries} empty summaries")
            
            # Check for unusually short/long texts
            if 'dialogue_word_count' in current_data.columns:
                very_short = (current_data['dialogue_word_count'] < 5).sum()
                very_long = (current_data['dialogue_word_count'] > 500).sum()
                if very_short > 0:
                    quality_issues.append(f"Found {very_short} very short dialogues (<5 words)")
                if very_long > 0:
                    quality_issues.append(f"Found {very_long} very long dialogues (>500 words)")
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "test_type": "data_quality",
                "tests_passed": len([t for t in results.get('tests', []) if t['status'] == 'success']),
                "tests_failed": len([t for t in results.get('tests', []) if t['status'] == 'fail']),
                "total_tests": len(results.get('tests', [])),
                "quality_issues": quality_issues,
                "details": results
            }
        
        except Exception as e:
            logger.error(f"Error in data quality tests: {e}")
            return {}
    
    def generate_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "drift_report.html"
    ) -> str:
        """
        Generate HTML report for drift analysis.
        
        Args:
            current_data: DataFrame with current inference data
            output_path: Path to save HTML report
        
        Returns:
            Path to generated report
        """
        if self.reference_data.empty or current_data.empty:
            logger.warning("Cannot generate report: missing data")
            return ""
        
        try:
            logger.info(f"Generating HTML report: {output_path}")
            
            report = Report(metrics=[
                DataDriftPreset()
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            report.save_html(output_path)
            logger.info(f"Report saved: {output_path}")
            
            # Upload to S3 if configured
            if self.s3_bucket:
                self._upload_to_s3(output_path)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""
    
    def _upload_to_s3(self, file_path: str):
        """Upload report to S3."""
        try:
            key = f"monitoring-reports/{datetime.utcnow().isoformat()}/{file_path}"
            self.s3_client.upload_file(file_path, self.s3_bucket, key)
            logger.info(f"Report uploaded to S3: s3://{self.s3_bucket}/{key}")
        except Exception as e:
            logger.warning(f"Failed to upload to S3: {e}")
    
    def _check_drift_threshold(self, report_dict: Dict) -> bool:
        """Check if drift exceeds threshold."""
        try:
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                if metric.get('result', {}).get('drift_detected'):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking drift threshold: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get ML monitoring summary (text features, not infrastructure)."""
        df = self.convert_logs_to_dataframe()
        
        if df.empty:
            return {}
        
        summary = {
            "total_inferences": len(df),
            
            # Input text statistics (data drift indicators)
            "avg_dialogue_words": df['dialogue_word_count'].mean() if 'dialogue_word_count' in df else None,
            "avg_dialogue_sentences": df['dialogue_sentence_count'].mean() if 'dialogue_sentence_count' in df else None,
            "avg_dialogue_lexical_diversity": df['dialogue_lexical_diversity'].mean() if 'dialogue_lexical_diversity' in df else None,
            
            # Output text statistics (prediction drift indicators)
            "avg_summary_words": df['summary_word_count'].mean() if 'summary_word_count' in df else None,
            "avg_summary_sentences": df['summary_sentence_count'].mean() if 'summary_sentence_count' in df else None,
            "avg_compression_ratio": df['compression_ratio'].mean() if 'compression_ratio' in df else None,
            
            # Model performance (if available)
            "avg_rouge_1": df['rouge_1'].mean() if 'rouge_1' in df and df['rouge_1'].notna().any() else None,
            "avg_rouge_2": df['rouge_2'].mean() if 'rouge_2' in df and df['rouge_2'].notna().any() else None,
            "avg_rouge_l": df['rouge_l'].mean() if 'rouge_l' in df and df['rouge_l'].notna().any() else None,
        }
        
        # Calculate trends
        if len(df) >= 20:
            recent_df = df.iloc[-10:]
            baseline_df = df.iloc[:10]
            
            if 'compression_ratio' in df:
                summary['compression_trend'] = (
                    "increasing" if recent_df['compression_ratio'].mean() > baseline_df['compression_ratio'].mean() 
                    else "decreasing"
                )
            
            if 'rouge_1' in df and df['rouge_1'].notna().any():
                recent_rouge = recent_df['rouge_1'].mean()
                baseline_rouge = baseline_df['rouge_1'].mean()
                summary['performance_trend'] = (
                    "improving" if recent_rouge > baseline_rouge * 1.05
                    else "degrading" if recent_rouge < baseline_rouge * 0.95
                    else "stable"
                )
        
        return {k: v for k, v in summary.items() if v is not None}
```

### Step 3: Update FastAPI for Evidently

Create file: `api/monitoring_routes.py`

```python
"""
FastAPI routes for Evidently monitoring dashboards.
"""

from fastapi import APIRouter, HTTPException
from src.evidently_monitoring import EvidentlyMonitor
import os
import pandas as pd

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Initialize monitor
monitor = EvidentlyMonitor(
    project_name="flan-t5-dialogsum",
    s3_bucket=os.getenv("S3_BUCKET")
)


@router.get("/status")
async def monitoring_status():
    """Get current monitoring status."""
    return {
        "status": "active",
        "total_inferences": len(monitor.inference_logs),
        "summary": monitor.get_summary()
    }


@router.post("/generate-report")
async def generate_report():
    """Generate and return monitoring report."""
    try:
        # Convert logs to DataFrame
        current_data = monitor.convert_logs_to_dataframe()
        
        if current_data.empty:
            raise HTTPException(status_code=400, detail="No inference data available")
        
        # Generate report
        report_path = monitor.generate_html_report(
            current_data=current_data,
            output_path="/tmp/drift_report.html"
        )
        
        if not report_path:
            raise HTTPException(status_code=500, detail="Failed to generate report")
        
        # Read and return HTML
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        return {"report": html_content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-data-drift")
async def detect_data_drift():
    """Detect drift in input text characteristics."""
    try:
        current_data = monitor.convert_logs_to_dataframe()
        
        if current_data.empty:
            raise HTTPException(status_code=400, detail="No inference data available")
        
        results = monitor.detect_data_drift(current_data)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-prediction-drift")
async def detect_prediction_drift():
    """Detect drift in generated summary characteristics."""
    try:
        current_data = monitor.convert_logs_to_dataframe()
        
        if current_data.empty:
            raise HTTPException(status_code=400, detail="No inference data available")
        
        results = monitor.detect_prediction_drift(current_data)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor-performance")
async def monitor_performance():
    """Monitor model performance (requires reference summaries)."""
    try:
        current_data = monitor.convert_logs_to_dataframe()
        
        if current_data.empty:
            raise HTTPException(status_code=400, detail="No inference data available")
        
        results = monitor.monitor_model_performance(current_data)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-quality-tests")
async def run_quality_tests():
    """Run data quality tests on ML features."""
    try:
        current_data = monitor.convert_logs_to_dataframe()
        
        if current_data.empty:
            raise HTTPException(status_code=400, detail="No inference data available")
        
        results = monitor.run_data_quality_tests(current_data)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4: Integrate with FastAPI Main

Update `api/main.py` to include Evidently:

```python
from src.evidently_monitoring import EvidentlyMonitor
from api.monitoring_routes import router as monitoring_router

# Initialize Evidently monitor
evidently_monitor = EvidentlyMonitor(
    project_name="flan-t5-dialogsum",
    s3_bucket=os.getenv("S3_BUCKET")
)

# Include monitoring routes
app.include_router(monitoring_router)

# Update summarize endpoint to log to Evidently (ML metrics only)
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Summarize dialogue with Evidently ML monitoring."""
    try:
        summary = summarize_dialogue(request.dialogue, _model, _tokenizer)
        
        # Log to Evidently with TEXT FEATURES (not infrastructure metrics)
        # Infrastructure metrics (latency, memory) are handled by Prometheus
        evidently_monitor.log_inference(
            dialogue=request.dialogue,
            summary=summary,
            reference_summary=request.reference_summary if hasattr(request, 'reference_summary') else None
        )
        
        return SummarizeResponse(summary=summary, id=str(uuid.uuid4()))
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 5: Create Monitoring Dashboard Script

Create file: `scripts/evidently_dashboard.py`

```python
"""
Generate Evidently monitoring dashboard.
"""

import argparse
import pandas as pd
from src.evidently_monitoring import EvidentlyMonitor
from datetime import datetime, timedelta


def create_baseline(num_samples: int = 100):
    """Create baseline dataset with ML features (not infrastructure metrics)."""
    print(f"📊 Creating baseline with {num_samples} TEXT FEATURES...")
    
    # Simulate baseline data with TEXT FEATURES (in production, this comes from initial model runs)
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        # Input text features (for data drift)
        'dialogue_word_count': np.random.normal(100, 20, num_samples),
        'dialogue_sentence_count': np.random.normal(8, 2, num_samples),
        'dialogue_avg_word_length': np.random.normal(4.5, 0.5, num_samples),
        'dialogue_lexical_diversity': np.random.normal(0.7, 0.1, num_samples),
        'dialogue_question_count': np.random.poisson(2, num_samples),
        'dialogue_unique_words': np.random.normal(70, 15, num_samples),
        
        # Prediction features (for prediction drift)
        'summary_word_count': np.random.normal(25, 5, num_samples),
        'summary_sentence_count': np.random.normal(2, 1, num_samples),
        'summary_lexical_diversity': np.random.normal(0.8, 0.1, num_samples),
        'summary_unique_words': np.random.normal(22, 4, num_samples),
        'compression_ratio': np.random.normal(4.0, 0.5, num_samples),
        
        # Performance metrics (if reference summaries available)
        'rouge_1': np.random.normal(0.45, 0.05, num_samples),
        'rouge_2': np.random.normal(0.25, 0.05, num_samples),
        'rouge_l': np.random.normal(0.40, 0.05, num_samples),
    })
    
    # Ensure non-negative values
    for col in baseline_data.columns:
        baseline_data[col] = baseline_data[col].clip(lower=0)
    
    return baseline_data


def main(args):
    """Generate monitoring dashboard."""
    
    monitor = EvidentlyMonitor(
        project_name="flan-t5-dialogsum",
        s3_bucket=args.s3_bucket,
        reference_data=create_baseline(args.baseline_samples)
    )
    
    # Simulate current data with TEXT FEATURES showing drift
    print("📈 Simulating current inferences with TEXT DRIFT...")
    np.random.seed(100)  # Different seed for drift
    current_data = pd.DataFrame({
        # Input features with DATA DRIFT (dialogues getting longer, more complex)
        'dialogue_word_count': np.random.normal(140, 25, 50),  # Increased from 100
        'dialogue_sentence_count': np.random.normal(11, 2, 50),  # Increased from 8
        'dialogue_avg_word_length': np.random.normal(4.8, 0.5, 50),  # Increased
        'dialogue_lexical_diversity': np.random.normal(0.75, 0.1, 50),  # Changed
        'dialogue_question_count': np.random.poisson(3, 50),  # Increased
        'dialogue_unique_words': np.random.normal(95, 18, 50),  # Increased
        
        # Prediction features with PREDICTION DRIFT (summaries getting longer)
        'summary_word_count': np.random.normal(32, 6, 50),  # Increased from 25
        'summary_sentence_count': np.random.normal(3, 1, 50),  # Increased from 2
        'summary_lexical_diversity': np.random.normal(0.78, 0.1, 50),  # Slight change
        'summary_unique_words': np.random.normal(28, 5, 50),  # Increased
        'compression_ratio': np.random.normal(4.4, 0.6, 50),  # Changed ratio
        
        # Performance metrics showing DEGRADATION
        'rouge_1': np.random.normal(0.38, 0.06, 50),  # Decreased from 0.45
        'rouge_2': np.random.normal(0.20, 0.05, 50),  # Decreased from 0.25
        'rouge_l': np.random.normal(0.33, 0.06, 50),  # Decreased from 0.40
    })
    
    # Ensure non-negative values
    for col in current_data.columns:
        current_data[col] = current_data[col].clip(lower=0)
    
    # Run DATA drift detection (input text changes)
    print("\n🔍 Running DATA DRIFT detection (input text characteristics)...")
    data_drift_results = monitor.detect_data_drift(current_data)
    print(f"  Data drift detected: {data_drift_results.get('drift_detected', False)}")
    print(f"  Features monitored: {data_drift_results.get('features_monitored', [])}")
    
    # Run PREDICTION drift detection (output text changes)
    print("\n🔍 Running PREDICTION DRIFT detection (summary characteristics)...")
    pred_drift_results = monitor.detect_prediction_drift(current_data)
    print(f"  Prediction drift detected: {pred_drift_results.get('drift_detected', False)}")
    print(f"  Features monitored: {pred_drift_results.get('features_monitored', [])}")
    
    # Monitor MODEL PERFORMANCE (ROUGE degradation)
    print("\n📉 Running MODEL PERFORMANCE monitoring (ROUGE scores)...")
    perf_results = monitor.monitor_model_performance(current_data)
    print(f"  Performance degradation: {perf_results.get('degradation_detected', False)}")
    if 'metrics' in perf_results:
        for metric, values in perf_results['metrics'].items():
            print(f"  {metric}: {values['baseline']:.3f} → {values['current']:.3f} ({values['change_pct']:+.1f}%)")
    
    # Run DATA QUALITY tests
    print("\n✅ Running DATA QUALITY tests...")
    test_results = monitor.run_data_quality_tests(current_data)
    print(f"  Tests passed: {test_results.get('tests_passed', 0)}/{test_results.get('total_tests', 0)}")
    print(f"  Quality issues: {test_results.get('quality_issues', [])}")
    
    # Generate HTML report
    print("📋 Generating HTML report...")
    report_path = monitor.generate_html_report(
        current_data=current_data,
        output_path="evidently_report.html"
    )
    print(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("MONITORING SUMMARY")
    print("="*60)
    print(monitor.get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently monitoring dashboard")
    parser.add_argument("--baseline-samples", type=int, default=100, help="Number of baseline samples")
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket for storing reports")
    
    args = parser.parse_args()
    main(args)
```

### Step 6: Environment Variables for Evidently

Update `.env.example`:

```bash
# Evidently Monitoring Configuration
S3_BUCKET=your-monitoring-bucket
EVIDENTLY_ENABLED=true
EVIDENTLY_BASELINE_SAMPLES=100
```

### Deployment Strategy

**Recommended Monitoring Stack for Production:**

```
┌─────────────────────────────────────────────────────────┐
│                    Production Setup                      │
├─────────────────────────────────────────────────────────┤
│  Infrastructure Monitoring (Section 1.2)                │
│  ✓ Prometheus - Metrics collection                      │
│  ✓ Grafana - Real-time dashboards                       │
│  ✓ CloudWatch - AWS integration & alerts                │
├─────────────────────────────────────────────────────────┤
│  Model Quality Monitoring (Section 1.3)                 │
│  ✓ Evidently AI - Drift detection & reports             │
├─────────────────────────────────────────────────────────┤
│  Result: Complete observability for ML in production    │
└─────────────────────────────────────────────────────────┘
```

**Implementation Order:**

1. **Start with basics (Section 1.2)**: Prometheus + Grafana for real-time metrics
2. **Add AWS integration (Section 1.2)**: CloudWatch for alerts and logging
3. **Enhance with ML monitoring (Section 1.3)**: Evidently AI for drift detection
4. **Iterate**: Tune thresholds based on production data

**When to Add Evidently AI:**

✅ **Add Evidently if:**
- You need detailed ML performance reports for stakeholders
- You're in a regulated industry requiring audit trails
- You need statistically rigorous drift detection
- You want automated data quality validation
- You have compliance requirements

✅ **Skip Evidently if (use only Prometheus/Grafana):**
- You're in early MVP stage
- You have very tight resource constraints
- Your use case has simple monitoring needs
- You already have custom ML monitoring tools

**Both tools are free and open-source - the choice is about complexity vs capability.**

---

### Implementation Checklist

When you're ready to implement full monitoring integration, follow these steps in order:

#### Phase 1: Prometheus + CloudWatch (Section 1.2)

1. **Install dependencies**
   ```bash
   pip install prometheus-client boto3 python-json-logger
   ```

2. **Create monitoring module**
   - Create `src/monitoring.py` (full code in Step 2 above)

3. **Update API**
   - Modify `api/main.py` to add monitoring (see Step 3 above)
   - Add `/metrics` endpoint for Prometheus

4. **Create CloudWatch setup script**
   - Create `scripts/setup_monitoring.py` (see Step 4 above)

5. **Setup Grafana (Optional)**
   - Create `docker-compose.grafana.yml`
   - Create `prometheus.yml`
   - Create `grafana/provisioning/datasources/prometheus.yml`
   - Create `grafana/provisioning/dashboards/dashboard.yml`
   - Create `grafana/dashboards/flan-t5-monitoring.json`
   - Create `grafana/provisioning/alerting/alerts.yml`

6. **Update environment variables**
   - Update `.env.example` with monitoring configuration

#### Phase 2: Evidently AI (Section 1.3)

7. **Install Evidently**
   ```bash
   pip install evidently
   ```

8. **Create Evidently monitoring module**
   - Create `src/evidently_monitoring.py` (full code in Step 2 above)

9. **Create Evidently API routes**
   - Create `api/monitoring_routes.py` (see Step 3 above)

10. **Update API to integrate Evidently**
    - Modify `api/main.py` to include monitoring routes (see Step 4 above)
    - Add inference logging to `/summarize` endpoint

11. **Create dashboard generation script**
    - Create `scripts/evidently_dashboard.py` (see Step 5 above)

12. **Update environment variables**
    - Add Evidently configuration to `.env.example`

#### Verification Steps

After implementation:

```bash
# Test Prometheus metrics
curl http://localhost:8000/metrics

# Test Evidently status
curl http://localhost:8000/monitoring/status

# Run drift detection
curl -X POST http://localhost:8000/monitoring/detect-data-drift

# Start Grafana stack (if using)
docker-compose -f docker-compose.grafana.yml up -d
```

---

## 2. Production Deployment Automation

### 2.1 Overview

Currently, the project has:
- ✅ Basic deployment scripts
- ✅ Dockerfile
- ❌ No Infrastructure as Code (Terraform/CloudFormation)
- ❌ No automated CI/CD pipeline for deployment
- ❌ No blue-green deployment strategy
- ❌ No auto-scaling configuration

### 2.2 Implementation: Terraform + GitHub Actions

#### Step 1: Create Terraform Configuration

Create file: `infrastructure/main.tf`

```hcl
# Main Terraform configuration for AWS infrastructure

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "terraform-state-bucket"
    key            = "flan-t5-dialogsum/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "flan-t5-dialogsum"
      ManagedBy   = "terraform"
    }
  }
}

# ECR Repository
resource "aws_ecr_repository" "flan_t5" {
  name                 = "${var.project_name}-${var.environment}"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "KMS"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster-${var.environment}"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Definition
resource "aws_ecs_task_definition" "flan_t5" {
  family                   = "${var.project_name}-task-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  
  container_definitions = jsonencode([{
    name      = var.project_name
    image     = "${aws_ecr_repository.flan_t5.repository_url}:latest"
    essential = true
    portMappings = [{
      containerPort = 8000
      hostPort      = 8000
      protocol      = "tcp"
    }]
    
    environment = [
      {
        name  = "MODEL_ID"
        value = var.model_id
      },
      {
        name  = "CLOUDWATCH_ENABLED"
        value = "true"
      }
    ]
    
    secrets = [
      {
        name      = "PEFT_WEIGHTS_PATH"
        valueFrom = aws_secretsmanager_secret.peft_weights.arn
      },
      {
        name      = "AWS_SECRET_ACCESS_KEY"
        valueFrom = aws_secretsmanager_secret.aws_secret_key.arn
      }
    ]
    
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.flan_t5.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])
  
  execution_role_arn = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_task_role.arn
}

# ECS Service
resource "aws_ecs_service" "flan_t5" {
  name            = "${var.project_name}-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.flan_t5.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.flan_t5.arn
    container_name   = var.project_name
    container_port   = 8000
  }
  
  depends_on = [
    aws_lb_listener.flan_t5,
    aws_iam_role_policy.ecs_task_execution_role_policy,
  ]
  
  # Enable deployment circuit breaker
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

# Auto Scaling Target
resource "aws_autoscaling_target" "ecs_target" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.flan_t5.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

# Auto Scaling Policy - CPU
resource "aws_autoscaling_policy" "ecs_policy_cpu" {
  name               = "${var.project_name}-cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_autoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_autoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_autoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# Auto Scaling Policy - Memory
resource "aws_autoscaling_policy" "ecs_policy_memory" {
  name               = "${var.project_name}-memory-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_autoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_autoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_autoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = 80.0
  }
}

# Load Balancer
resource "aws_lb" "flan_t5" {
  name               = "${var.project_name}-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
}

resource "aws_lb_target_group" "flan_t5" {
  name        = "${var.project_name}-tg-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  
  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }
}

resource "aws_lb_listener" "flan_t5" {
  load_balancer_arn = aws_lb.flan_t5.arn
  port              = 80
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.flan_t5.arn
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "flan_t5" {
  name              = "/ecs/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days
}

# IAM Roles
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "${var.project_name}-ecs-task-execution-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_role_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_task_execution_role_policy" {
  name = "${var.project_name}-ecs-task-execution-policy"
  role = aws_iam_role.ecs_task_execution_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "kms:Decrypt"
        ]
        Resource = [
          aws_secretsmanager_secret.peft_weights.arn,
          aws_secretsmanager_secret.aws_secret_key.arn
        ]
      }
    ]
  })
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name}-ecs-task-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.project_name}-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket}",
          "arn:aws:s3:::${var.s3_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# Security Groups
resource "aws_security_group" "alb" {
  name   = "${var.project_name}-alb-sg"
  vpc_id = var.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs_tasks" {
  name   = "${var.project_name}-ecs-sg"
  vpc_id = var.vpc_id
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

Create file: `infrastructure/variables.tf`

```hcl
variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "prod"
}

variable "project_name" {
  default = "flan-t5-dialogsum"
}

variable "model_id" {
  default = "google/flan-t5-base"
}

variable "vpc_id" {
  description = "VPC ID"
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "s3_bucket" {
  description = "S3 bucket for model artifacts"
}

variable "ecs_task_cpu" {
  default = 2048  # 2 CPU
}

variable "ecs_task_memory" {
  default = 4096  # 4 GB
}

variable "desired_count" {
  default = 2
}

variable "min_capacity" {
  default = 1
}

variable "max_capacity" {
  default = 5
}

variable "log_retention_days" {
  default = 30
}
```

Create file: `infrastructure/outputs.tf`

```hcl
output "load_balancer_dns" {
  value       = aws_lb.flan_t5.dns_name
  description = "DNS name of the load balancer"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.flan_t5.repository_url
  description = "ECR repository URL"
}

output "ecs_service_name" {
  value       = aws_ecs_service.flan_t5.name
  description = "ECS service name"
}

output "cloudwatch_log_group" {
  value       = aws_cloudwatch_log_group.flan_t5.name
  description = "CloudWatch log group name"
}
```

#### Step 2: Create Deployment Script

Create file: `scripts/deploy_infrastructure.sh`

```bash
#!/bin/bash

set -e

ENVIRONMENT=${1:-prod}
AWS_REGION=${2:-us-east-1}
TERRAFORM_DIR="infrastructure"

echo "🚀 Deploying FlanT5 DialogSum Infrastructure"
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"

# Initialize Terraform
echo "📦 Initializing Terraform..."
cd $TERRAFORM_DIR
terraform init -upgrade

# Plan
echo "📋 Planning deployment..."
terraform plan \
  -var="environment=$ENVIRONMENT" \
  -var="aws_region=$AWS_REGION" \
  -out=tfplan

# Apply
echo "✅ Applying infrastructure changes..."
terraform apply tfplan

# Output endpoints
echo ""
echo "🎉 Deployment complete!"
terraform output -json > "../deployment_outputs_${ENVIRONMENT}.json"
echo "Outputs saved to: deployment_outputs_${ENVIRONMENT}.json"

cd ..
```

#### Step 3: Create GitHub Actions CI/CD Pipeline

Create file: `.github/workflows/deploy.yml`

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'api/**'
      - 'Dockerfile'
      - '.github/workflows/deploy.yml'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    permissions:
      id-token: write
      contents: read
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-east-1
      
      - name: Login to ECR
        run: |
          aws ecr get-login-password --region us-east-1 | \
          docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}
      
      - name: Build Docker image
        run: |
          docker build -t ${{ secrets.ECR_REGISTRY }}/flan-t5-dialogsum:${{ github.sha }} .
          docker tag ${{ secrets.ECR_REGISTRY }}/flan-t5-dialogsum:${{ github.sha }} \
                     ${{ secrets.ECR_REGISTRY }}/flan-t5-dialogsum:latest
      
      - name: Push to ECR
        run: |
          docker push ${{ secrets.ECR_REGISTRY }}/flan-t5-dialogsum:${{ github.sha }}
          docker push ${{ secrets.ECR_REGISTRY }}/flan-t5-dialogsum:latest
      
      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster flan-t5-cluster-prod \
            --service flan-t5-service-prod \
            --force-new-deployment \
            --region us-east-1
      
      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \
            --cluster flan-t5-cluster-prod \
            --services flan-t5-service-prod \
            --region us-east-1
      
      - name: Run smoke tests
        run: |
          ENDPOINT=$(aws elbv2 describe-load-balancers \
            --query 'LoadBalancers[0].DNSName' \
            --output text)
          
          curl -X POST "http://$ENDPOINT/summarize" \
            -H "Content-Type: application/json" \
            -d '{"dialogue": "Test dialogue"}'
```

#### Step 4: Add Deployment Documentation

Create file: `docs/DEPLOYMENT_AUTOMATION.md`

```markdown
# Deployment Automation Guide

## Prerequisites

- Terraform installed (v1.3+)
- AWS CLI configured
- Docker
- GitHub repository with secrets configured

## AWS Setup

### 1. Create IAM Role for GitHub Actions

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main"
        }
      }
    }
  ]
}
EOF

# Create role
aws iam create-role \
  --role-name github-actions-flan-t5 \
  --assume-role-policy-document file://trust-policy.json

# Attach policy
aws iam attach-role-policy \
  --role-name github-actions-flan-t5 \
  --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess
```

### 2. Add GitHub Secrets

```bash
AWS_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/github-actions-flan-t5
ECR_REGISTRY=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

## Deployment Steps

### Initial Infrastructure Setup

```bash
# First time only: Create Terraform state bucket
aws s3api create-bucket --bucket terraform-state-bucket --region us-east-1
aws s3api put-bucket-versioning \
  --bucket terraform-state-bucket \
  --versioning-configuration Status=Enabled

# Create DynamoDB lock table
aws dynamodb create-table \
  --table-name terraform-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

### Deploy Infrastructure

```bash
# Create terraform.tfvars
cat > infrastructure/terraform.tfvars <<EOF
vpc_id = "vpc-xxxxx"
public_subnet_ids = ["subnet-xxxxx", "subnet-xxxxx"]
private_subnet_ids = ["subnet-xxxxx", "subnet-xxxxx"]
s3_bucket = "your-model-bucket"
EOF

# Deploy
bash scripts/deploy_infrastructure.sh prod us-east-1
```

### Automated Deployment on Push

Once GitHub Actions is configured, deployments happen automatically on push to main.

## Blue-Green Deployment

The setup supports blue-green deployments through ECS service configurations:

```bash
# Create new task definition (green)
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json

# Update service to new definition
aws ecs update-service \
  --cluster flan-t5-cluster-prod \
  --service flan-t5-service-prod \
  --task-definition flan-t5-task-prod:NEW_VERSION
```

## Monitoring Deployment

```bash
# Check service status
aws ecs describe-services \
  --cluster flan-t5-cluster-prod \
  --services flan-t5-service-prod

# View logs
aws logs tail /ecs/flan-t5-dialogsum-prod --follow
```

## Rollback

```bash
# Rollback to previous task definition
aws ecs update-service \
  --cluster flan-t5-cluster-prod \
  --service flan-t5-service-prod \
  --task-definition flan-t5-task-prod:PREVIOUS_VERSION
```
```

---

## 3. Secret Management Integration

### 3.1 Overview

Currently, the project has:
- ✅ Environment variable support
- ❌ No AWS Secrets Manager integration
- ❌ No credential rotation
- ❌ No access audit logging

### 3.2 Implementation: AWS Secrets Manager

#### Step 1: Install Dependencies

```bash
pip install python-dotenv
```

#### Step 2: Create Secret Manager Module

Create file: `src/secret_manager.py`

```python
"""
Secure secret management using AWS Secrets Manager.
"""

import os
import json
import logging
from typing import Dict, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class SecretManager:
    """Manage secrets from AWS Secrets Manager."""
    
    def __init__(self):
        self.sm_client = boto3.client('secretsmanager')
        self.cache = {}
    
    def get_secret(self, secret_name: str, use_cache: bool = True) -> Dict:
        """
        Retrieve secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret in Secrets Manager
            use_cache: Whether to cache secret values (disable for sensitive data)
        
        Returns:
            Dictionary containing secret key-value pairs
        """
        # Check cache first
        if use_cache and secret_name in self.cache:
            logger.info(f"Using cached secret: {secret_name}")
            return self.cache[secret_name]
        
        try:
            logger.info(f"Retrieving secret: {secret_name}")
            response = self.sm_client.get_secret_value(SecretId=secret_name)
            
            # Parse JSON secret
            if 'SecretString' in response:
                secret = json.loads(response['SecretString'])
            else:
                secret = response['SecretBinary']
            
            # Cache for future requests
            if use_cache:
                self.cache[secret_name] = secret
            
            logger.info(f"Successfully retrieved secret: {secret_name}")
            return secret
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Error retrieving secret {secret_name}: {error_code}")
            
            if error_code == 'ResourceNotFoundException':
                raise ValueError(f"Secret not found: {secret_name}")
            elif error_code == 'InvalidRequestException':
                raise ValueError(f"Invalid request for secret: {secret_name}")
            elif error_code == 'InvalidParameterException':
                raise ValueError(f"Invalid parameter for secret: {secret_name}")
            else:
                raise
    
    def create_secret(
        self,
        secret_name: str,
        secret_value: Dict,
        tags: Optional[list] = None
    ) -> str:
        """
        Create a new secret in AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret
            secret_value: Dictionary of key-value pairs
            tags: List of tags for the secret
        
        Returns:
            ARN of the created secret
        """
        try:
            logger.info(f"Creating secret: {secret_name}")
            response = self.sm_client.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value),
                Tags=tags or []
            )
            logger.info(f"Secret created: {secret_name}")
            return response['ARN']
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceExistsException':
                logger.warning(f"Secret already exists: {secret_name}")
                return None
            else:
                logger.error(f"Error creating secret: {e}")
                raise
    
    def update_secret(
        self,
        secret_name: str,
        secret_value: Dict
    ) -> str:
        """
        Update an existing secret.
        
        Args:
            secret_name: Name of the secret
            secret_value: Updated dictionary of key-value pairs
        
        Returns:
            ARN of the updated secret
        """
        try:
            logger.info(f"Updating secret: {secret_name}")
            response = self.sm_client.update_secret(
                SecretId=secret_name,
                SecretString=json.dumps(secret_value)
            )
            logger.info(f"Secret updated: {secret_name}")
            return response['ARN']
        
        except ClientError as e:
            logger.error(f"Error updating secret: {e}")
            raise
    
    def rotate_secret(self, secret_name: str, rotation_days: int = 30) -> str:
        """
        Enable automatic secret rotation.
        
        Args:
            secret_name: Name of the secret
            rotation_days: Days between rotations
        
        Returns:
            ARN of the secret
        """
        try:
            logger.info(f"Enabling rotation for secret: {secret_name}")
            response = self.sm_client.rotate_secret(
                SecretId=secret_name,
                RotationRules={
                    'AutomaticallyAfterDays': rotation_days
                }
            )
            logger.info(f"Secret rotation enabled for: {secret_name}")
            return response['ARN']
        
        except ClientError as e:
            logger.error(f"Error rotating secret: {e}")
            raise


class CredentialManager:
    """Manage AWS credentials securely."""
    
    def __init__(self):
        self.sm = SecretManager()
    
    def load_aws_credentials(self) -> Dict:
        """
        Load AWS credentials from Secrets Manager.
        Falls back to environment variables if secret not found.
        
        Returns:
            Dictionary with AWS credentials
        """
        # Try to load from Secrets Manager first
        try:
            secret = self.sm.get_secret('flan-t5-aws-credentials')
            os.environ['AWS_ACCESS_KEY_ID'] = secret.get('access_key_id')
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret.get('secret_access_key')
            logger.info("AWS credentials loaded from Secrets Manager")
            return secret
        
        except (ValueError, ClientError):
            logger.warning("Failed to load from Secrets Manager, falling back to environment")
            return {
                'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
            }
    
    def load_model_credentials(self) -> Dict:
        """
        Load model-specific credentials (API keys, S3 paths, etc).
        
        Returns:
            Dictionary with model credentials
        """
        try:
            secret = self.sm.get_secret('flan-t5-model-credentials')
            return secret
        except (ValueError, ClientError):
            logger.warning("Model credentials not found in Secrets Manager")
            return {
                'peft_weights_path': os.getenv('PEFT_WEIGHTS_PATH'),
                'model_id': os.getenv('MODEL_ID', 'google/flan-t5-base')
            }


def load_environment_safely():
    """
    Load environment variables from Secrets Manager instead of .env file.
    Use in production instead of python-dotenv.
    """
    sm = SecretManager()
    cred_mgr = CredentialManager()
    
    # Load AWS credentials
    cred_mgr.load_aws_credentials()
    
    # Load model credentials
    model_creds = cred_mgr.load_model_credentials()
    
    # Load API configuration
    try:
        api_config = sm.get_secret('flan-t5-api-config')
        os.environ['API_KEY'] = api_config.get('api_key', '')
        os.environ['PEFT_WEIGHTS_PATH'] = api_config.get('peft_weights_path', '')
    except (ValueError, ClientError):
        logger.warning("API config not found, using environment variables")
    
    logger.info("Environment variables loaded from Secrets Manager")
```

#### Step 3: Create Setup Script for Secrets

Create file: `scripts/setup_secrets.py`

```python
"""
Setup AWS Secrets Manager secrets for the project.
Run this once to initialize all secrets.
"""

import argparse
import json
from src.secret_manager import SecretManager

def setup_secrets(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    peft_weights_path: str,
    api_key: str
):
    """Setup all required secrets in AWS Secrets Manager."""
    
    sm = SecretManager()
    
    # AWS Credentials
    print("📝 Creating AWS credentials secret...")
    sm.create_secret(
        'flan-t5-aws-credentials',
        {
            'access_key_id': aws_access_key_id,
            'secret_access_key': aws_secret_access_key
        },
        tags=[
            {'Key': 'Environment', 'Value': 'production'},
            {'Key': 'Project', 'Value': 'flan-t5-dialogsum'}
        ]
    )
    
    # Model Credentials
    print("📝 Creating model credentials secret...")
    sm.create_secret(
        'flan-t5-model-credentials',
        {
            'peft_weights_path': peft_weights_path,
            'model_id': 'google/flan-t5-base'
        },
        tags=[
            {'Key': 'Environment', 'Value': 'production'},
            {'Key': 'Project', 'Value': 'flan-t5-dialogsum'}
        ]
    )
    
    # API Configuration
    print("📝 Creating API configuration secret...")
    sm.create_secret(
        'flan-t5-api-config',
        {
            'api_key': api_key,
            'peft_weights_path': peft_weights_path
        },
        tags=[
            {'Key': 'Environment', 'Value': 'production'},
            {'Key': 'Project', 'Value': 'flan-t5-dialogsum'}
        ]
    )
    
    # Enable rotation for sensitive secrets
    print("🔄 Enabling rotation for AWS credentials...")
    sm.rotate_secret('flan-t5-aws-credentials', rotation_days=90)
    
    print("✅ All secrets created and configured!")
    print("\nNext steps:")
    print("1. Update IAM role policies to allow access to these secrets")
    print("2. Update infrastructure/main.tf with secret ARNs")
    print("3. Deploy infrastructure with Terraform")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup AWS Secrets Manager secrets")
    parser.add_argument("--aws-key-id", required=True, help="AWS access key ID")
    parser.add_argument("--aws-secret", required=True, help="AWS secret access key")
    parser.add_argument("--peft-path", required=True, help="Path to PEFT weights (local or S3)")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    
    args = parser.parse_args()
    
    setup_secrets(
        aws_access_key_id=args.aws_key_id,
        aws_secret_access_key=args.aws_secret,
        peft_weights_path=args.peft_path,
        api_key=args.api_key
    )
```

#### Step 4: Update API to Use Secrets Manager

Update `api/main.py`:

```python
from src.secret_manager import load_environment_safely

# Load secrets at startup
load_environment_safely()

# Then use environment variables as normal
MODEL_ID = os.getenv("MODEL_ID", "google/flan-t5-base")
PEFT_WEIGHTS_PATH = os.getenv("PEFT_WEIGHTS_PATH")
API_KEY = os.getenv("API_KEY")
```

#### Step 5: Create Secrets Documentation

Create file: `docs/SECRET_MANAGEMENT.md`

```markdown
# Secret Management Guide

## Overview

Secrets (AWS credentials, API keys, model paths) are stored in AWS Secrets Manager instead of `.env` files.

## Setup

### 1. Create Secrets

```bash
python scripts/setup_secrets.py \
  --aws-key-id YOUR_KEY \
  --aws-secret YOUR_SECRET \
  --peft-path s3://bucket/path \
  --api-key YOUR_API_KEY
```

### 2. Grant ECS Task Access

Update IAM policy in `infrastructure/main.tf` (already included).

### 3. Update Environment

In Terraform variables or task definition, reference the secrets:

```hcl
secrets = [
  {
    name      = "AWS_SECRET_ACCESS_KEY"
    valueFrom = "arn:aws:secretsmanager:region:account:secret:flan-t5-aws-credentials"
  }
]
```

## Best Practices

- 🔐 Never commit credentials to git
- 🔄 Rotate credentials every 90 days
- 📋 Audit secret access in CloudTrail
- 🏷️ Tag secrets for easy management
- 🛡️ Use IAM roles instead of long-term credentials

## Rotating Secrets

```bash
# Manual rotation
python -c "from src.secret_manager import SecretManager; \
           sm = SecretManager(); \
           sm.update_secret('flan-t5-aws-credentials', {'access_key_id': 'NEW_KEY', ...})"

# Or configure automatic rotation in AWS console
```

## Accessing Secrets in Code

```python
from src.secret_manager import SecretManager

sm = SecretManager()
credentials = sm.get_secret('flan-t5-aws-credentials')
api_key = credentials['access_key_id']
```
```

---

## Summary Checklist

### Model Monitoring
- [ ] Install monitoring dependencies
- [ ] Create `src/monitoring.py`
- [ ] Update `api/main.py` with monitor integration
- [ ] Setup CloudWatch dashboard
- [ ] Configure SNS alerts
- [ ] Update `.env.example` with monitoring variables

### Production Deployment Automation
- [ ] Create Terraform configuration (`infrastructure/`)
- [ ] Create deployment script (`scripts/deploy_infrastructure.sh`)
- [ ] Setup GitHub Actions pipeline (`.github/workflows/deploy.yml`)
- [ ] Create Terraform state bucket in S3
- [ ] Configure AWS IAM for GitHub Actions
- [ ] Add GitHub secrets

### Secret Management
- [ ] Install dotenv package
- [ ] Create `src/secret_manager.py`
- [ ] Create `scripts/setup_secrets.py`
- [ ] Setup secrets in AWS Secrets Manager
- [ ] Update `api/main.py` to use SecretManager
- [ ] Create documentation

---

## Testing Implementation

### Test Monitoring
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Person A: Hi there! Person B: Hello!"}'

# Check metrics
curl http://localhost:8000/metrics
```

### Test Secret Management
```bash
python -c "from src.secret_manager import SecretManager; \
           sm = SecretManager(); \
           print(sm.get_secret('flan-t5-aws-credentials'))"
```

### Test Deployment Automation
```bash
cd infrastructure
terraform plan -var-file=dev.tfvars
```

---

**After implementing these features, update the README.md section from "❌ No X" to "✅ X Implemented"**
