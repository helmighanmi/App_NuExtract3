# Deployment assets

| Path | Purpose |
| --- | --- |
| `vllm/serve.sh` | Bare-metal / container entrypoint with production vLLM flags |
| `kubernetes/vllm.yaml` | GPU model server, PVC, probes, Service |
| `kubernetes/app.yaml` | Stateless API (HPA) + Streamlit |
| `kubernetes/kustomization.yaml` | `kubectl apply -k deploy/kubernetes` |

See [../docs/deployment.md](../docs/deployment.md) for sizing, MTP, quantization,
cloud options and a production checklist.
