# Docker Deployment Guide

Complete guide for deploying the FLAN-T5 DialogSum API using Docker, from local development to production.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [File Structure](#file-structure)
4. [Development Setup](#development-setup)
5. [Production Deployment](#production-deployment)
6. [Security Features](#security-features)
7. [Common Commands](#common-commands)
8. [Monitoring (Optional)](#monitoring-optional)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### API Only (No Monitoring)

```powershell
# 1. Setup environment
Copy-Item .env.example .env
# Edit .env with your settings

# 2. Start the API
docker-compose up -d

# 3. Test
curl http://localhost:8000/health

# Access: http://localhost:8000
```

### API + Monitoring Stack

```powershell
# Start everything (API + Prometheus + Grafana + AlertManager)
make docker-dev

# Access:
# API:        http://localhost:8000
# Grafana:    http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

---

## Architecture Overview

### Container Communication

```
┌─────────────────────────────────────────────────────────────┐
│            Docker Network: "flan-t5-network"                 │
│                                                               │
│  ┌──────────────┐                                            │
│  │   FastAPI    │ ← Your application                         │
│  │   :8000      │   - Model inference                        │
│  │              │   - Exposes /metrics endpoint              │
│  └──────┬───────┘                                            │
│         │                                                     │
│         │ (Optional monitoring - only if enabled)            │
│         ↓                                                     │
│  ┌──────────────┐                                            │
│  │ Prometheus   │ ← Scrapes http://api:8000/metrics          │
│  │   :9090      │   - Stores time-series data                │
│  └──────┬───────┘                                            │
│         │                                                     │
│         ├─→ ┌────────────┐                                   │
│         │   │  Grafana   │ ← Dashboards                      │
│         │   │   :3000    │                                   │
│         │   └────────────┘                                   │
│         │                                                     │
│         └─→ ┌──────────────┐                                 │
│             │ AlertManager │ ← Alerts to Slack/Email         │
│             │    :9093     │                                 │
│             └──────────────┘                                 │
└───────────────────────────────────────────────────────────────┘
         │              │              │
    Port 8000      Port 3000      Port 9090
    (Public)       (Dashboard)    (Metrics)
```

### Multi-File Architecture (Professional)

The project uses multiple Docker Compose files for separation of concerns:

| File | Purpose | When Loaded |
|------|---------|-------------|
| `docker-compose.yml` | **Core API service** | Always |
| `docker-compose.monitoring.yml` | Monitoring stack (Prometheus, Grafana) | Optional - only when monitoring needed |
| `docker-compose.override.yml` | Development conveniences (auto-loaded) | Automatically in dev |
| `docker-compose.prod.yml` | Production security hardening | Only in production |

**Why this approach?**
- ✅ Deploy API without monitoring (monitoring can be external like Datadog)
- ✅ Different teams manage different stacks
- ✅ Environment-specific configurations
- ✅ Follows Docker Compose best practices

---

## File Structure

```
project/
├── docker-compose.yml              # Base - API service
├── docker-compose.monitoring.yml   # Monitoring stack (optional)
├── docker-compose.override.yml     # Dev settings (auto-loaded)
├── docker-compose.prod.yml         # Production hardening
├── Dockerfile                      # API image definition
│
├── .env                            # Environment variables (gitignored)
├── .env.example                    # Template
│
├── Makefile                        # Simplified commands
│
├── monitoring/                     # Monitoring configs
│   ├── prometheus.yml              # Prometheus configuration
│   ├── prometheus-alerts.yml       # Alert rules
│   ├── alertmanager.yml            # Alert routing
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   └── dashboards/
│       └── dashboards/
│           └── flan-t5-monitoring.json
│
└── docs/
    └── DOCKER_DEPLOYMENT_GUIDE.md  # This file
```

---

## Development Setup

### 1. Configure Environment

```powershell
# Copy template
Copy-Item .env.example .env

# Generate API key for metrics
$key = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
Write-Host "Generated API Key: $key"

# Edit .env and set:
# METRICS_API_KEY=<generated-key>
# METRICS_REQUIRE_AUTH=false  # For easier local testing
```

### 2. Start Services

**Option A: API Only**
```powershell
docker-compose up -d

# Or using Makefile
make docker-dev-api-only
```

**Option B: API + Monitoring**
```powershell
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Or using Makefile
make docker-dev
```

### 3. Verify

```powershell
# Check all services
docker-compose ps

# View logs
docker-compose logs -f

# Test API
curl http://localhost:8000/health

# Generate test inference
Invoke-RestMethod -Uri http://localhost:8000/summarize `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"dialogue": "John: Hey! How are you?\nMary: Great, thanks!"}'
```

### 4. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

---

## Production Deployment

### 1. Security Checklist

Before deploying to production:

- [ ] Strong passwords in `.env` (never commit to Git)
- [ ] `METRICS_REQUIRE_AUTH=true`
- [ ] Strong `METRICS_API_KEY` (32+ characters)
- [ ] Change Grafana admin password from default
- [ ] HTTPS reverse proxy configured (nginx/traefik)
- [ ] Firewall rules configured
- [ ] Regular security updates scheduled
- [ ] Backup strategy for volumes
- [ ] Monitoring alerts configured

### 2. Start Production Stack

```powershell
# Ensure .env has production values
notepad .env

# Start with production hardening
docker-compose -f docker-compose.yml `
               -f docker-compose.monitoring.yml `
               -f docker-compose.prod.yml `
               up -d

# Or using Makefile
make docker-prod
```

### 3. Verify Deployment

```powershell
# Check health
make docker-health

# Test API (should work)
curl http://localhost:8000/health

# Verify metrics require auth
curl http://localhost:8000/metrics  # Should return 403

# Test with authentication
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/metrics
```

### 4. Production Access

In production, services should NOT be exposed directly:

```
Internet
    ↓
HTTPS (443) → Nginx/Traefik (Reverse Proxy)
    ↓
    ├─→ http://api:8000         (Main API)
    └─→ http://grafana:3000     (Dashboards)

Prometheus/AlertManager: Internal only (no public access)
```

#### Example nginx configuration:

```nginx
# /etc/nginx/sites-available/flan-t5
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 443 ssl http2;
    server_name grafana.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Security Features

### What's Implemented (Production Mode)

```yaml
# From docker-compose.prod.yml

security_opt:
  - no-new-privileges:true  # Prevents privilege escalation

read_only: true             # Immutable filesystem (prevents tampering)

user: "65534:65534"         # Run as non-root (nobody user)

cap_drop:                   # Drop all Linux capabilities
  - ALL

ports:
  - "127.0.0.1:8000:8000"  # Only localhost (reverse proxy required)

deploy:
  resources:
    limits:
      cpus: '4.0'           # Resource limits (prevents DoS)
      memory: 8G

logging:
  driver: "json-file"
  options:
    max-size: "10m"         # Log rotation
    max-file: "3"
```

### Security Comparison

| Feature | Development | Production |
|---------|-------------|------------|
| **User** | root | nobody (65534) |
| **Filesystem** | Read-write | Read-only |
| **Capabilities** | All | None (dropped) |
| **Ports** | Public (0.0.0.0) | Localhost only |
| **Auth** | Optional | Required |
| **Secrets** | Relaxed | Strict validation |
| **Logs** | Unlimited | Size-limited rotation |

---

## Common Commands

### Using Makefile (Recommended)

```bash
# Development
make docker-dev              # Start API + monitoring
make docker-dev-api-only     # Start only API
make docker-logs             # View logs
make docker-health           # Check status

# Production
make docker-prod             # Start with security hardening
make docker-prod-down        # Stop production

# Utilities
make docker-build            # Rebuild API image
make docker-shell            # Shell into API container
make docker-stats            # Resource usage
make docker-clean            # Remove everything (with confirmation)
```

### Manual Docker Compose Commands

```powershell
# Start API only
docker-compose up -d

# Start API + Monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Start Production (security hardened)
docker-compose -f docker-compose.yml `
               -f docker-compose.monitoring.yml `
               -f docker-compose.prod.yml `
               up -d

# Stop services
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v

# View logs
docker-compose logs -f
docker-compose logs -f api

# Rebuild API
docker-compose build api
docker-compose up -d api

# Execute command in container
docker-compose exec api bash
docker-compose exec api python --version

# Check status
docker-compose ps

# Resource usage
docker stats
```

### Container Management

```powershell
# Restart specific service
docker-compose restart api

# Update after code changes
docker-compose build api
docker-compose up -d api

# Shell into running container
docker-compose exec api bash

# View container inspect
docker inspect flan-t5-api

# Check logs (last 100 lines)
docker-compose logs --tail=100 api
```

---

## Monitoring (Optional)

### When to Enable Monitoring

Enable monitoring when you need:
- 📊 Real-time performance metrics (latency, throughput)
- 🚨 Automated alerts (high latency, errors, downtime)
- 📈 Historical trend analysis
- 🔍 Debugging performance issues

### Enable Monitoring

```powershell
# Start with monitoring
make docker-dev

# Or manually
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### Configure Alerts

Edit `monitoring/alertmanager.yml` to add your notification channels:

```yaml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

receivers:
  - name: 'slack-critical'
    slack_configs:
      - channel: '#alerts'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

Restart AlertManager:
```powershell
docker-compose restart alertmanager
```

### Monitoring Stack Details

See [PRODUCTION_IMPLEMENTATION_GUIDE.md](PRODUCTION_IMPLEMENTATION_GUIDE.md) Section 1.2 for complete monitoring setup including:
- Prometheus configuration
- Grafana dashboards
- Alert rules
- Security setup

---

## Troubleshooting

### Container won't start

```powershell
# Check logs for errors
docker-compose logs api

# Check disk space
docker system df

# Clean up
docker system prune -a

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### API shows as unhealthy

```powershell
# Check health endpoint directly
curl http://localhost:8000/health

# View container logs
docker-compose logs api

# Check if model is loading
docker-compose exec api ls -lh /root/.cache/huggingface

# Memory issues? Check limits
docker stats
```

### Prometheus shows API as "DOWN"

```powershell
# Verify API is running
docker-compose ps api

# Check API metrics endpoint
curl http://localhost:8000/metrics

# Verify authentication (if enabled)
curl -H "Authorization: Bearer YOUR_KEY" http://localhost:8000/metrics

# Check Prometheus can reach API from inside network
docker-compose exec prometheus wget -O- http://api:8000/metrics
```

### Port already in use

```powershell
# Find what's using the port
netstat -ano | findstr :8000

# Kill the process (if safe)
Stop-Process -Id <PID> -Force

# Or change port in docker-compose.override.yml:
# ports:
#   - "8001:8000"  # Use different host port
```

### Volume permissions issues

```powershell
# Linux/Mac: Fix volume permissions
docker-compose exec api chown -R 65534:65534 /app/logs

# Or run as root in development (not recommended for production)
# In docker-compose.override.yml:
# user: "0:0"
```

### Out of memory

```powershell
# Check memory usage
docker stats

# Increase limits in docker-compose.override.yml:
# deploy:
#   resources:
#     limits:
#       memory: 16G  # Increase limit
```

### Logs filling disk

```powershell
# Check log sizes
docker system df -v

# Configure rotation in docker-compose.yml:
# logging:
#   driver: "json-file"
#   options:
#     max-size: "10m"
#     max-file: "3"

# Manual cleanup
docker system prune -a
```

---

## Data Persistence

### Volumes

All data is stored in Docker named volumes:

- `model-cache` - HuggingFace model cache (API)
- `prometheus-data` - Metrics database (Prometheus)
- `alertmanager-data` - Alert state (AlertManager)
- `grafana-data` - Dashboards and settings (Grafana)

### Backup Data

```powershell
# Create backups directory
mkdir backups

# Backup Prometheus metrics
docker run --rm `
  -v flant5-dialogsum-sagemaker_prometheus-data:/data `
  -v ${PWD}/backups:/backup `
  alpine tar czf /backup/prometheus-$(Get-Date -Format "yyyyMMdd").tar.gz /data

# Backup Grafana dashboards
docker run --rm `
  -v flant5-dialogsum-sagemaker_grafana-data:/data `
  -v ${PWD}/backups:/backup `
  alpine tar czf /backup/grafana-$(Get-Date -Format "yyyyMMdd").tar.gz /data
```

### Restore Data

```powershell
# Restore Prometheus
docker run --rm `
  -v flant5-dialogsum-sagemaker_prometheus-data:/data `
  -v ${PWD}/backups:/backup `
  alpine tar xzf /backup/prometheus-20260208.tar.gz -C /

# Restore Grafana
docker run --rm `
  -v flant5-dialogsum-sagemaker_grafana-data:/data `
  -v ${PWD}/backups:/backup `
  alpine tar xzf /backup/grafana-20260208.tar.gz -C /
```

---

## Environment Variables Reference

See `.env.example` for complete list. Key variables:

```bash
# Model Configuration
MODEL_ID=google/flan-t5-base
PEFT_WEIGHTS_PATH=                    # Optional LoRA weights

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_REQUIRE_AUTH=true             # MUST be true in production
METRICS_API_KEY=your-secret-key       # Generate: openssl rand -hex 32

# Application
LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR
WORKERS=4                             # Number of worker processes

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me      # CHANGE IN PRODUCTION

# CloudWatch (optional)
CLOUDWATCH_ENABLED=false
AWS_REGION=us-east-1
```

---

## Deployment Workflow

```
┌─────────────┐
│  Develop    │  ← Local: docker-compose up -d
│  Locally    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Commit to  │  ← Push code to Git
│  Git        │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  CI/CD      │  ← GitHub Actions, Jenkins, etc.
│  Pipeline   │    - Run tests
└──────┬──────┘    - Build Docker image
       │           - Push to registry
       ↓
┌─────────────┐
│  Deploy to  │  ← Pull image, docker-compose up
│  Production │    with docker-compose.prod.yml
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Monitor &  │  ← Grafana dashboards
│  Alert      │    AlertManager notifications
└─────────────┘
```

---

## Best Practices

### Development
- ✅ Use `docker-compose.override.yml` for local customizations (auto-loaded)
- ✅ Mount code volumes for hot reload during development
- ✅ Use `make docker-logs` to watch logs during development
- ✅ Keep `.env` out of Git (already in `.gitignore`)

### Production
- ✅ Always use `docker-compose.prod.yml` for security hardening
- ✅ Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- ✅ Set up HTTPS reverse proxy (never expose HTTP directly)
- ✅ Configure log aggregation (ELK, Loki, CloudWatch)
- ✅ Regular backups of volumes
- ✅ Monitor resource usage and set appropriate limits
- ✅ Use vulnerability scanning (`docker scan`)
- ✅ Keep base images updated

### Monitoring
- ✅ Set realistic alert thresholds based on your SLA
- ✅ Configure multiple notification channels
- ✅ Test alerts before relying on them
- ✅ Document what each alert means and how to respond
- ✅ Review and adjust alert rules monthly

---

## FAQ

**Q: Do I need monitoring for local development?**
A: No. Start with just `docker-compose up -d` (API only). Add monitoring when you need it.

**Q: Why multiple Docker Compose files?**
A: Separation of concerns. API is independent. Monitoring is optional. Production has different security needs.

**Q: Can I deploy this to AWS/Azure/GCP?**
A: Yes! Works on EC2, Azure VMs, GCP Compute. For managed services, consider ECS/AKS/GKE.

**Q: How do I scale to multiple API instances?**
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      replicas: 3  # Run 3 instances
```

**Q: How do I add GPU support?**
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Q: What if I want to use external monitoring (Datadog, New Relic)?**
A: Perfect! Just don't load `docker-compose.monitoring.yml`. Use only the base `docker-compose.yml`.

---

## Next Steps

1. ✅ Review this guide
2. → Set up `.env` with your configuration
3. → Test locally: `make docker-dev-api-only`
4. → Add monitoring when needed: `make docker-dev`
5. → Read [PRODUCTION_IMPLEMENTATION_GUIDE.md](PRODUCTION_IMPLEMENTATION_GUIDE.md) for detailed monitoring setup
6. → Deploy to production: `make docker-prod`

For detailed monitoring configuration, alert rules, and Grafana dashboards, see the full monitoring guide in Section 1.2 of the Production Implementation Guide.
