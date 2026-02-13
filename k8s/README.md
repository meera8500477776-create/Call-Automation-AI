# Kubernetes Deployment Manifests

This directory contains Kubernetes manifests for deploying the Config-Driven Voice Agent Platform.

## Quick Start

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Create secrets (update with your values first!)
kubectl apply -f secret.yaml

# 3. Create config map
kubectl apply -f configmap.yaml

# 4. Create persistent volume claim (for GPU nodes)
kubectl apply -f pvc.yaml

# 5. Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# 6. Configure auto-scaling (requires KEDA)
kubectl apply -f keda-scaledobject.yaml
```

## Files

- `namespace.yaml` - Creates the voice-agent namespace
- `secret.yaml` - Stores API keys and credentials (update before deploying!)
- `configmap.yaml` - Application configuration
- `deployment.yaml` - Main application and GPU deployments
- `service.yaml` - LoadBalancer and ClusterIP services
- `pvc.yaml` - Persistent volume claim for model cache
- `keda-scaledobject.yaml` - KEDA auto-scaling configuration

## Important Notes

1. **Update secrets before deploying!** The `secret.yaml` file contains placeholder values.
2. **GPU nodes required** for local LLM inference (voice-agent-gpu deployment)
3. **KEDA must be installed** for auto-scaling to work
4. **Prometheus required** for KEDA metrics-based scaling

## Customization

### Resource Limits

Edit `deployment.yaml` to adjust CPU/memory/GPU limits:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### Scaling Configuration

Edit `keda-scaledobject.yaml` to adjust scaling thresholds:

```yaml
spec:
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      threshold: "50"  # Adjust this value
```

## Verification

```bash
# Check all resources
kubectl get all -n voice-agent

# Check pod logs
kubectl logs -n voice-agent -l app=voice-agent --tail=100

# Check scaling status
kubectl get scaledobject -n voice-agent
```

For detailed deployment instructions, see [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).
