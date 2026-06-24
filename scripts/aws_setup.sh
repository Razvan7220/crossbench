#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# aws_setup.sh — Recreate the full cross-bench AWS infrastructure
#
# Prerequisites:
#   - AWS CLI configured (aws sts get-caller-identity works)
#   - Pulumi CLI installed (~/.pulumi/bin/pulumi)
#   - Python venv at .venv with pulumi_aws installed
#   - PULUMI_CONFIG_PASSPHRASE set
#
# What this creates:
#   - VPC (10.0.0.0/16) with 2 public + 2 private subnets
#   - ECS cluster (EC2-backed, c5.xlarge, ASG 0-1)
#   - RDS Postgres 16 (db.t4g.medium, private subnet)
#   - 4 ECR repos (orchestrator, function, bigcodebench, terminal)
#   - 2 S3 buckets (artifacts, datasets)
#   - CodeBuild project for CI/CD
#   - All IAM roles, security groups, log groups
#
# Usage:
#   export PULUMI_CONFIG_PASSPHRASE="benchplatform2026"
#   bash scripts/aws_setup.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
export HOME=/home/valentin
export PATH="/home/valentin/.local/bin:/home/valentin/.pulumi/bin:/usr/local/bin:/usr/bin:/bin"

REGION="us-east-1"
AWS_ACCOUNT="564175165906"

cd /home/valentin/cross_bench

echo "=== Step 1: Pulumi up (VPC, ECS, RDS, ECR, S3, IAM) ==="
pulumi up --yes --skip-preview

echo ""
echo "=== Step 2: Build and push Docker images ==="
ECR_URL="${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"

# Build orchestrator
docker build -t cross-bench-orchestrator:latest .
docker tag cross-bench-orchestrator:latest "${ECR_URL}/cross-bench-orchestrator:latest"
docker push "${ECR_URL}/cross-bench-orchestrator:latest"

# Build function sandbox
docker build -t cross-bench-function:latest \
    -f src/cross_bench/execution/docker_templates/base.Dockerfile .
docker tag cross-bench-function:latest "${ECR_URL}/cross-bench-function:latest"
docker push "${ECR_URL}/cross-bench-function:latest"

echo ""
echo "=== Step 3: Create CodeBuild project ==="
# Check if it already exists
if ! aws codebuild batch-get-projects --names cross-bench-build --region "$REGION" \
    --query 'projects[0].name' --output text 2>/dev/null | grep -q cross-bench-build; then

    # Create S3 source bucket if needed
    BUCKET="codebuild-cross-bench-source-${AWS_ACCOUNT}"
    aws s3 mb "s3://${BUCKET}" --region "$REGION" 2>/dev/null || true

    # Create CodeBuild service role
    bash scripts/setup_codebuild.sh
fi

echo ""
echo "=== Step 4: Verify ==="
pulumi stack output --json
echo ""
echo "Infrastructure is ready. To run a benchmark:"
echo "  bash scripts/run_demo.sh"
echo ""
echo "To tear down:"
echo "  bash scripts/aws_teardown.sh"
