#!/bin/bash
set -e

# Configuration
ENVIRONMENT=${1:-prod}
REGION=${2:-us-west-2}
CLUSTER_NAME="ray-llm-cluster-${ENVIRONMENT}"
NAMESPACE="ray-${ENVIRONMENT}"

echo "Deploying Ray LLM cluster to ${ENVIRONMENT} environment in ${REGION}..."

# Install required tools
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting."; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "terraform is required but not installed. Aborting."; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI is required but not installed. Aborting."; exit 1; }

# Initialize and apply Terraform infrastructure
echo "Provisioning infrastructure with Terraform..."
cd ../terraform
terraform init
terraform workspace select ${ENVIRONMENT} || terraform workspace new ${ENVIRONMENT}
terraform apply -var="environment=${ENVIRONMENT}" -var="aws_region=${REGION}" -auto-approve

# Get EKS cluster credentials
echo "Configuring kubectl for the EKS cluster..."
aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${REGION}

# Create namespace if it doesn't exist
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Create Hugging Face token secret
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
  echo "Warning: HUGGING_FACE_HUB_TOKEN environment variable not set. Using placeholder value."
  echo "For production, ensure this token is properly set."
  HUGGING_FACE_HUB_TOKEN="placeholder-token"
fi

kubectl create secret generic hf-token \
  --from-literal=token=${HUGGING_FACE_HUB_TOKEN} \
  --namespace=${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

# Create persistent volume claim for model cache
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ray-model-cache-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: gp2
EOF

# Apply Kubernetes manifests
echo "Deploying Ray cluster..."
kubectl apply -f ../kubernetes/ray-cluster.yaml -n ${NAMESPACE}

# Wait for Ray cluster to be ready
echo "Waiting for Ray cluster to be ready..."
kubectl wait --for=condition=Established --timeout=5m raycluster/${CLUSTER_NAME} -n ${NAMESPACE}

# Deploy Ray Serve LLM application
echo "Deploying Ray Serve LLM application..."
kubectl apply -f ../kubernetes/ray-serve-llm.yaml -n ${NAMESPACE}

# Set up monitoring
echo "Setting up monitoring..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f ../monitoring/prometheus-config.yaml

# Deploy Prometheus and Grafana using Helm
echo "Deploying Prometheus and Grafana..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set server.configMapOverrideName=prometheus-config

helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
helm upgrade --install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=true \
  --set persistence.size=10Gi

# Get access information
echo ""
echo "Deployment completed successfully!"
echo ""
echo "Ray Dashboard URL:"
echo "  kubectl port-forward svc/${CLUSTER_NAME}-head-svc 8265:8265 -n ${NAMESPACE}"
echo "  Then visit: http://localhost:8265"
echo ""
echo "Ray Serve LLM API:"
echo "  kubectl port-forward svc/ray-serve-llm-${ENVIRONMENT}-llm-app-svc 8000:8000 -n ${NAMESPACE}"
echo "  Then use: http://localhost:8000 as your OpenAI-compatible API endpoint"
echo ""
echo "Grafana Dashboard:"
echo "  kubectl port-forward svc/grafana 3000:3000 -n monitoring"
echo "  Then visit: http://localhost:3000"
echo "  Username: admin, Password: $(kubectl get secret grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode)"
echo "" 