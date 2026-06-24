#!/bin/bash
# Kick off the CodeBuild project that builds + pushes orchestrator
# images. The project is provisioned by Pulumi and pulls source from
# GitHub Enterprise (``devotion-labs.ghe.com/research/CrossBench``);
# there's nothing to package locally — just push your commits to GHE
# first, then run this.
#
# Source branch defaults to ``main``. Override with a positional arg:
#
#     ./scripts/codebuild_rebuild.sh          # main
#     ./scripts/codebuild_rebuild.sh feature/foo
#
# PROJECT / REGION can be overridden via env if you aren't running from
# the Pulumi stack that owns the project.
set -euo pipefail
export HOME=/home/valentin
export PATH="/home/valentin/.pulumi/bin:/home/valentin/.local/bin:/usr/local/bin:/usr/bin:/bin"

REGION="${REGION:-us-east-1}"
SOURCE_VERSION="${1:-main}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${PROJECT:-}" ]; then
    echo "=== Resolving project name from Pulumi outputs ==="
    pushd "$REPO_ROOT/src/cross_bench/infra/pulumi" > /dev/null
    PROJECT="$(pulumi stack output --show-secrets --json 2>/dev/null \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("codebuild_project",""))')"
    popd > /dev/null
fi

if [ -z "${PROJECT:-}" ]; then
    echo "ERROR: could not resolve PROJECT from Pulumi outputs. Set it explicitly:"
    echo "  PROJECT=cross-bench-dev-build $0"
    exit 2
fi

echo "  PROJECT=$PROJECT"
echo "  SOURCE_VERSION=$SOURCE_VERSION"

echo "=== Starting CodeBuild ==="
BUILD_ID=$(aws codebuild start-build \
    --project-name "$PROJECT" \
    --source-version "$SOURCE_VERSION" \
    --region "$REGION" \
    --query 'build.id' --output text)
echo "Build: $BUILD_ID"

while true; do
    STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
        --query 'builds[0].buildStatus' --output text)
    PHASE=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
        --query 'builds[0].currentPhase' --output text)
    echo "  $(date +%H:%M:%S) | $STATUS | $PHASE"
    [ "$STATUS" != "IN_PROGRESS" ] && break
    sleep 15
done

if [ "$STATUS" = "SUCCEEDED" ]; then
    echo "=== BUILD SUCCEEDED — images updated in ECR ==="
else
    echo "=== BUILD FAILED: $STATUS ==="
    aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
        --query 'builds[0].logs.deepLink' --output text
    exit 1
fi
