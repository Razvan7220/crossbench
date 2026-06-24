#!/usr/bin/env bash
# =============================================================================
# deploy_aws.sh — Deploy CodCoBench to AWS and run a HumanEval demo
#
# Prerequisites:
#   - AWS CLI configured (aws sts get-caller-identity works)
#   - Pulumi CLI installed (curl -fsSL https://get.pulumi.com | sh)
#   - Docker running locally (for image builds)
#   - OPENAI_API_KEY set in environment
#
# Usage:
#   ./scripts/deploy_aws.sh               # full deploy + run demo
#   ./scripts/deploy_aws.sh infra          # deploy infrastructure only
#   ./scripts/deploy_aws.sh images         # build & push images only
#   ./scripts/deploy_aws.sh demo           # launch HumanEval run only
#   ./scripts/deploy_aws.sh teardown       # destroy all resources
# =============================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

AWS_REGION="${AWS_REGION:-us-east-1}"
STACK="${PULUMI_STACK:-dev}"
TASK_LIMIT="${TASK_LIMIT:-10}"

# ── Colours ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}▸ $*${NC}"; }
ok()    { echo -e "${GREEN}✓ $*${NC}"; }
fail()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────
preflight() {
    command -v aws    >/dev/null || fail "AWS CLI not found"
    command -v pulumi >/dev/null || fail "Pulumi CLI not found"
    command -v docker >/dev/null || fail "Docker not found"
    aws sts get-caller-identity >/dev/null 2>&1 || fail "AWS credentials not configured"
    [ -n "${OPENAI_API_KEY:-}" ] || fail "OPENAI_API_KEY not set"
    ok "Pre-flight checks passed"
}

# ── Step 1: Pulumi infrastructure ──────────────────────────────────
deploy_infra() {
    info "Initialising Pulumi stack '${STACK}'..."
    pulumi stack select "$STACK" 2>/dev/null || pulumi stack init "$STACK"

    info "Setting Pulumi config..."
    pulumi config set cross-bench:environment "$STACK"
    pulumi config set --secret cross-bench:db_password "BenchPlatf0rm_$(openssl rand -hex 8)"
    pulumi config set --secret cross-bench:openai_api_key "$OPENAI_API_KEY"

    info "Deploying infrastructure (this takes ~5 min)..."
    pulumi up --yes

    ok "Infrastructure deployed"
}

# ── Step 2: Build & push Docker images ─────────────────────────────
push_images() {
    AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    ECR_URL="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

    info "Authenticating to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" \
        | docker login --username AWS --password-stdin "$ECR_URL"

    info "Building base image..."
    bash scripts/build_docker_images.sh base

    info "Building sandbox images..."
    bash scripts/build_docker_images.sh function

    info "Pushing sandbox images to ECR..."
    docker tag  cross-bench-base:latest     "$ECR_URL/cross-bench-function:base"
    docker tag  cross-bench-function:latest  "$ECR_URL/cross-bench-function:latest"
    docker push "$ECR_URL/cross-bench-function:latest"

    info "Building orchestrator image..."
    docker build -t cross-bench-orchestrator:latest -f Dockerfile .

    info "Pushing orchestrator image to ECR..."
    docker tag  cross-bench-orchestrator:latest "$ECR_URL/cross-bench-orchestrator:latest"
    docker push "$ECR_URL/cross-bench-orchestrator:latest"

    ok "All images pushed to ECR"
}

# ── Step 3: Launch HumanEval demo ──────────────────────────────────
run_demo() {
    CLUSTER=$(pulumi stack output ecs_cluster)
    TASK_DEF=$(pulumi stack output task_def_arn)

    info "Launching full HumanEval benchmark (164 tasks) on cluster ${CLUSTER}..."
    TASK_ARN=$(aws ecs run-task \
        --cluster "$CLUSTER" \
        --task-definition "$TASK_DEF" \
        --count 1 \
        --overrides "{
            \"containerOverrides\": [{
                \"name\": \"orchestrator\",
                \"command\": [
                    \"run\",
                    \"--profile\", \"online_no_memory\",
                    \"--benchmark\", \"humaneval\",
                    \"--mode\", \"single_shot\"
                ]
            }]
        }" \
        --query 'tasks[0].taskArn' --output text)

    ok "Task launched: ${TASK_ARN}"
    info "Tailing CloudWatch logs (Ctrl-C to stop)..."

    LOG_GROUP=$(pulumi stack output --json | python3 -c "
import sys, json
# The log group name follows the pattern cross-bench-{env}-logs
print('/aws/ecs/cross-bench-${STACK}' if True else '')
" 2>/dev/null || echo "")

    # Wait a few seconds for logs to appear
    sleep 5
    aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" \
        --query 'tasks[0].lastStatus' --output text

    info "Monitor with:"
    echo "  aws ecs describe-tasks --cluster $CLUSTER --tasks $TASK_ARN"
    echo "  aws logs tail /cross-bench-${STACK}-logs --follow"
}

# ── Teardown ───────────────────────────────────────────────────────
teardown() {
    info "Destroying all AWS resources..."
    pulumi destroy --yes
    ok "All resources destroyed"
}

# ── Main ───────────────────────────────────────────────────────────
case "${1:-all}" in
    infra)    preflight; deploy_infra ;;
    images)   preflight; push_images ;;
    demo)     run_demo ;;
    teardown) teardown ;;
    all)      preflight; deploy_infra; push_images; run_demo ;;
    *)        echo "Usage: $0 {all|infra|images|demo|teardown}"; exit 1 ;;
esac
