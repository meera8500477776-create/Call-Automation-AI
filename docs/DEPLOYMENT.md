# Deployment Guide

This guide covers deploying the Config-Driven Voice Agent Platform using Docker and Kubernetes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Environment Configuration](#environment-configuration)
- [Secrets Management](#secrets-management)
- [Auto-Scaling Configuration](#auto-scaling-configuration)
- [Monitoring and Observability](#monitoring-and-observability)

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Kubernetes 1.24+ (for production deployment)
- kubectl configured with cluster access
- KEDA 2.0+ installed in Kubernetes cluster (for auto-scaling)
- GPU nodes with NVIDIA drivers (for local LLM inference)

## Docker Deployment

### Local Development

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Configure environment variables:**
   Edit `.env` and add your API keys:
   ```bash
   DEEPGRAM_API_KEY=your_deepgram_key
   OPENAI_API_KEY=your_openai_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   TWILIO_ACCOUNT_SID=your_twilio_sid
   TWILIO_AUTH_TOKEN=your_twilio_token
   ```

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - API: http://localhost:8000
   - Health check: http://localhost:8000/health

### Production Docker Build

1. **Build production image:**
   ```bash
   docker build -t voice-agent:latest .
   ```

2. **Run with production settings:**
   ```bash
   docker run -d \
     --name voice-agent \
     -p 8000:8000 \
     -e ENVIRONMENT=production \
     -e DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY \
     -e OPENAI_API_KEY=$OPENAI_API_KEY \
     -e ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
     -e TWILIO_ACCOUNT_SID=$TWILIO_ACCOUNT_SID \
     -e TWILIO_AUTH_TOKEN=$TWILIO_AUTH_TOKEN \
     -v $(pwd)/config:/app/config:ro \
     voice-agent:latest
   ```

## Kubernetes Deployment

### Step 1: Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### Step 2: Configure Secrets

**Option A: Using kubectl (for development)**

```bash
kubectl create secret generic voice-agent-secrets \
  --namespace=voice-agent \
  --from-literal=deepgram_api_key=YOUR_KEY \
  --from-literal=openai_api_key=YOUR_KEY \
  --from-literal=elevenlabs_api_key=YOUR_KEY \
  --from-literal=twilio_account_sid=YOUR_SID \
  --from-literal=twilio_auth_token=YOUR_TOKEN \
  --from-literal=fhir_client_id=YOUR_CLIENT_ID \
  --from-literal=fhir_client_secret=YOUR_SECRET \
  --from-literal=dms_api_key=YOUR_KEY
```

**Option B: Using External Secrets Operator (recommended for production)**

Install External Secrets Operator and configure with your secret management system (AWS Secrets Manager, HashiCorp Vault, etc.).

### Step 3: Create ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml
```

### Step 4: Create Persistent Volume Claim (for GPU nodes)

```bash
kubectl apply -f k8s/pvc.yaml
```

### Step 5: Deploy Application

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Step 6: Verify Deployment

```bash
# Check pod status
kubectl get pods -n voice-agent

# Check service
kubectl get svc -n voice-agent

# View logs
kubectl logs -n voice-agent -l app=voice-agent --tail=100
```

### Step 7: Configure Auto-Scaling (KEDA)

1. **Install KEDA:**
   ```bash
   kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml
   ```

2. **Deploy ScaledObjects:**
   ```bash
   kubectl apply -f k8s/keda-scaledobject.yaml
   ```

3. **Verify KEDA configuration:**
   ```bash
   kubectl get scaledobject -n voice-agent
   ```

## Environment Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production`, `staging`, `development` |
| `LOG_LEVEL` | Logging level | `INFO`, `DEBUG`, `WARNING` |
| `DEEPGRAM_API_KEY` | Deepgram STT API key | `abc123...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ELEVENLABS_API_KEY` | ElevenLabs TTS API key | `xyz789...` |
| `TWILIO_ACCOUNT_SID` | Twilio account SID | `AC...` |
| `TWILIO_AUTH_TOKEN` | Twilio auth token | `...` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CONCURRENT_CALLS` | Maximum concurrent calls | `1000` |
| `LATENCY_THRESHOLD_MS` | Latency alert threshold | `1000` |
| `ENABLE_OBSERVABILITY` | Enable observability platform | `true` |

## Secrets Management

### Best Practices

1. **Never commit secrets to version control**
   - Use `.gitignore` to exclude `.env` files
   - Use placeholder values in `k8s/secret.yaml`

2. **Use external secret management in production**
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault
   - Google Secret Manager

3. **Rotate secrets regularly**
   - Set up automated rotation policies
   - Update secrets in Kubernetes without downtime

4. **Use RBAC to restrict secret access**
   ```bash
   kubectl create role secret-reader \
     --verb=get,list \
     --resource=secrets \
     --namespace=voice-agent
   ```

### Example: AWS Secrets Manager Integration

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets
  namespace: voice-agent
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: voice-agent-sa
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: voice-agent-secrets
  namespace: voice-agent
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: voice-agent-secrets
  data:
  - secretKey: deepgram_api_key
    remoteRef:
      key: voice-agent/deepgram
  - secretKey: openai_api_key
    remoteRef:
      key: voice-agent/openai
```

## Auto-Scaling Configuration

### KEDA Scaling Triggers

The platform uses KEDA for auto-scaling based on multiple metrics:

#### 1. Concurrent Calls Scaling

Scales based on active voice sessions:
- **Threshold**: 50 concurrent calls per GPU pod
- **Min replicas**: 1
- **Max replicas**: 10

#### 2. GPU Utilization Scaling

Scales based on GPU usage:
- **Threshold**: 70% GPU utilization
- **Scale up**: When utilization > 80%
- **Scale down**: When utilization < 30%

#### 3. Request Queue Depth

Scales based on pending requests:
- **Threshold**: 100 requests in queue
- **Cooldown**: 5 minutes

### Customizing Scaling Policies

Edit `k8s/keda-scaledobject.yaml`:

```yaml
spec:
  minReplicaCount: 2  # Increase minimum replicas
  maxReplicaCount: 20  # Increase maximum replicas
  pollingInterval: 30  # Check metrics every 30 seconds
  cooldownPeriod: 600  # Wait 10 minutes before scaling down
```

### Cost Optimization

The `ScalingController` provides cost optimization recommendations:

```python
from voice_agent.deployment import ScalingController, LoadMetrics, ScalingPolicy

controller = ScalingController(policy)
metrics = controller.get_current_load()

# Get cost optimization recommendation
strategy = controller.optimize_cost(
    metrics=metrics,
    cloud_cost_per_call=0.05,
    local_cost_per_call=0.02
)
# Returns: "cloud", "local", or "hybrid"
```

## Monitoring and Observability

### Health Checks

- **Liveness probe**: `/health` - Checks if application is running
- **Readiness probe**: `/ready` - Checks if application can accept traffic

### Metrics Collection

The platform exposes Prometheus metrics:

```yaml
# Add Prometheus annotations to deployment
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

### Key Metrics

- `voice_agent_active_sessions` - Current active calls
- `voice_agent_end_to_end_latency_ms` - End-to-end latency
- `voice_agent_request_queue_depth` - Pending requests
- `nvidia_gpu_duty_cycle` - GPU utilization

### Logging

Logs are structured JSON for easy parsing:

```bash
# View logs in Kubernetes
kubectl logs -n voice-agent -l app=voice-agent -f

# Filter by log level
kubectl logs -n voice-agent -l app=voice-agent | grep '"level":"ERROR"'
```

### Distributed Tracing

Configure LangSmith or Helicone for LLM tracing:

```bash
# Add to environment variables
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=voice-agent-prod
```

## Troubleshooting

### Common Issues

**1. Pods not starting**
```bash
kubectl describe pod -n voice-agent <pod-name>
kubectl logs -n voice-agent <pod-name>
```

**2. High latency**
- Check GPU utilization: `kubectl top nodes`
- Review KEDA scaling: `kubectl get scaledobject -n voice-agent`
- Check network latency to external APIs

**3. Secret access errors**
```bash
# Verify secret exists
kubectl get secret voice-agent-secrets -n voice-agent

# Check RBAC permissions
kubectl auth can-i get secrets --namespace=voice-agent
```

**4. GPU not available**
```bash
# Check GPU nodes
kubectl get nodes -l gpu=true

# Verify NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia
```

## Production Checklist

- [ ] Secrets stored in external secret manager
- [ ] RBAC policies configured
- [ ] Resource limits set on all pods
- [ ] Auto-scaling configured and tested
- [ ] Monitoring and alerting set up
- [ ] Backup strategy for persistent volumes
- [ ] Disaster recovery plan documented
- [ ] Load testing completed
- [ ] Security scanning performed
- [ ] Compliance requirements verified (HIPAA, etc.)

## Support

For issues or questions:
- Check logs: `kubectl logs -n voice-agent -l app=voice-agent`
- Review metrics: Access Prometheus/Grafana dashboards
- Contact: support@example.com
