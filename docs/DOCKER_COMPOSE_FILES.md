# Docker Compose Files Explained

## Why Multiple Files?

Your project uses the **Docker Compose Multi-File Pattern** - an industry-standard approach for managing different environments.

---

## The Three Files

### 1️⃣ **docker-compose.yml** (Base - Always Used)

**Purpose:** Core API service definition

```yaml
services:
  api:
    build: .
    image: flan-t5-api
    environment:
      - MODEL_ID=...
      - PROMETHEUS_ENABLED=...
    volumes:
      - model-cache:/root/.cache/huggingface
    healthcheck:
      test: curl -f http://localhost:8000/health
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    networks:
      - app-network
```

**Contains:**
- Service definition (what container to run)
- Environment variables
- Basic health checks
- Resource limits
- Volume mounts

**Used:** Always (foundation for both dev and prod)

---

### 2️⃣ **docker-compose.override.yml** (Dev - Auto-Loaded)

**Purpose:** Development conveniences

```yaml
services:
  api:
    ports:
      - "8000:8000"  # Expose port for local access
    
    environment:
      - LOG_LEVEL=DEBUG  # Verbose logging
      - METRICS_REQUIRE_AUTH=false  # Easy testing
    
    deploy:
      resources:
        limits:
          cpus: '4.0'  # Use more CPU locally
          memory: 8G
```

**Contains:**
- Port mappings (access from host)
- Debug settings
- Relaxed security
- Higher resource limits (you have more resources locally)

**Used:** Automatically loaded when you run `docker-compose up -d`

**Magic:** Docker Compose **automatically** loads `docker-compose.override.yml` if it exists - no `-f` flag needed!

---

### 3️⃣ **docker-compose.prod.yml** (Production - Manual)

**Purpose:** Security hardening and production settings

```yaml
services:
  api:
    ports:
      - "127.0.0.1:8000:8000"  # Only localhost (reverse proxy required)
    
    environment:
      - LOG_LEVEL=WARNING  # Less verbose
      - METRICS_REQUIRE_AUTH=true  # Security enabled
    
    security_opt:
      - no-new-privileges:true  # Can't escalate privileges
    read_only: true  # Immutable filesystem
    user: "65534:65534"  # Run as non-root
    cap_drop:
      - ALL  # Drop all Linux capabilities
    
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'  # Guaranteed resources
          memory: 4G
    
    logging:
      driver: "json-file"
      options:
        max-size: "10m"  # Log rotation
        max-file: "3"
```

**Contains:**
- Security hardening (non-root, read-only, no capabilities)
- Strict resource limits
- Log rotation
- Production-only settings

**Used:** Explicitly loaded with `-f` flag for production deployments

---

## How They Work Together

### **Development** (Automatic)

```powershell
docker-compose up -d
```

**Loads:**
1. `docker-compose.yml` (base)
2. `docker-compose.override.yml` (auto-loaded)

**Result:**
- ✅ API accessible at http://localhost:8000
- ✅ Debug logging enabled
- ✅ No authentication burden
- ✅ Easy local development

---

### **Production** (Explicit)

```powershell
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

**Loads:**
1. `docker-compose.yml` (base)
2. `docker-compose.prod.yml` (replaces override settings)

**Result:**
- 🔒 Security hardened (non-root, read-only)
- 🔒 No public ports (reverse proxy required)
- 🔒 Authentication enabled
- 🔒 Resource limits enforced
- 🔒 Log rotation enabled

---

## File Loading Priority

Docker Compose **merges** files in order, with later files overriding earlier ones:

```
Development:
  docker-compose.yml (base)
  ↓ merged with ↓
  docker-compose.override.yml (dev settings)
  = Development configuration

Production:
  docker-compose.yml (base)
  ↓ merged with ↓
  docker-compose.prod.yml (production settings)
  = Production configuration
```

**Example - Port Mapping:**

| File | Port Setting | Result |
|------|--------------|--------|
| `docker-compose.yml` | No ports defined | Ports not exposed |
| `docker-compose.override.yml` | `"8000:8000"` | **Dev:** Exposed to all interfaces |
| `docker-compose.prod.yml` | `"127.0.0.1:8000:8000"` | **Prod:** Only localhost |

---

## Benefits of Multi-File

### ✅ **Separation of Concerns**
- Base config doesn't change between environments
- Environment-specific settings isolated
- Easier to review security changes

### ✅ **No Code Duplication**
- Shared settings in base file
- Only differences in override files
- DRY principle

### ✅ **Safer Production**
- Can't accidentally deploy with dev settings
- Explicit production configuration
- Security settings documented

### ✅ **Easy Local Development**
- Auto-loaded override = convenience
- No commands to remember
- Just `docker-compose up -d`

---

## Common Patterns

### **View Merged Configuration**

```powershell
# See what actually runs in dev
docker-compose config

# See what would run in production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml config
```

### **Override Specific Settings Locally**

Create `docker-compose.local.yml` (gitignored):
```yaml
services:
  api:
    volumes:
      - ./api:/app/api  # Mount code for hot reload
```

Then:
```powershell
docker-compose -f docker-compose.yml -f docker-compose.local.yml up -d
```

---

## Alternative: Single File Approach

If you prefer ONE file, you could use environment variables:

```yaml
# docker-compose.yml (single file)
services:
  api:
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - METRICS_REQUIRE_AUTH=${METRICS_REQUIRE_AUTH:-true}
```

Then:
```bash
# .env.dev
API_PORT=8000
LOG_LEVEL=DEBUG
METRICS_REQUIRE_AUTH=false

# .env.prod
API_PORT=127.0.0.1:8000
LOG_LEVEL=WARNING
METRICS_REQUIRE_AUTH=true
```

**BUT:** Can't do security hardening (read_only, user, etc.) via env vars - would need manual editing.

---

## Your Current Setup

```
✅ docker-compose.yml           - Base API (used always)
✅ docker-compose.override.yml  - Dev settings (auto-loaded)
✅ docker-compose.prod.yml      - Production hardening (explicit)

Commands:
  Development: docker-compose up -d
  Production:  docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
  Or:          make docker-prod
```

This is a **professional, industry-standard** setup used by companies like:
- Google (GCP Cloud Run)
- AWS (ECS configuration patterns)
- Microsoft (Azure Container Apps)

---

## Quick Reference

| Scenario | Command | Files Loaded |
|----------|---------|--------------|
| Local dev | `docker-compose up -d` | base + override (auto) |
| Production | `make docker-prod` | base + prod |
| View config | `docker-compose config` | Shows merged dev config |
| Stop dev | `docker-compose down` | - |
| Stop prod | `make docker-prod-down` | - |

---

**TL;DR:** You have 3 files because it's a professional pattern:
- One base definition (what to run)
- One for dev convenience (auto-loaded)
- One for production security (explicit)

This is **normal and recommended** for real-world projects!
