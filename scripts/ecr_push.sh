#!/bin/bash
set -euo pipefail
export HOME=/home/valentin
export PATH="/home/valentin/.local/bin:/home/valentin/.pulumi/bin:/usr/local/bin:/usr/bin:/bin"

ECR="564175165906.dkr.ecr.us-east-1.amazonaws.com"

# Configure Docker to push one layer at a time (reduces TLS timeout issues)
mkdir -p /home/valentin/.docker
cat > /home/valentin/.docker/daemon.json 2>/dev/null << 'EOF' || true
{"max-concurrent-uploads": 1}
EOF

echo "=== Logging into ECR ==="
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ECR"

echo "=== Pushing function image (1 layer at a time) ==="
DOCKER_CONFIG=/home/valentin/.docker docker push --concurrency 1 "$ECR/cross-bench-function:latest" 2>&1 || \
docker push "$ECR/cross-bench-function:latest" 2>&1

echo "=== Pushing orchestrator image ==="
DOCKER_CONFIG=/home/valentin/.docker docker push --concurrency 1 "$ECR/cross-bench-orchestrator:latest" 2>&1 || \
docker push "$ECR/cross-bench-orchestrator:latest" 2>&1

echo "ALL_PUSHED"
