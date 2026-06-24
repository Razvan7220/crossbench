#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# aws_teardown.sh — Tear down all cross-bench AWS infrastructure
#
# This destroys everything created by aws_setup.sh / Pulumi:
#   - ECS cluster, ASG, launch template, capacity provider
#   - RDS instance (skip_final_snapshot=true, no snapshot created)
#   - VPC, subnets, security groups, IGW, route tables
#   - ECR repos (images deleted)
#   - S3 buckets (artifacts, datasets)
#   - IAM roles, instance profiles, policies
#   - CloudWatch log group, SSM parameter
#
# Also cleans up non-Pulumi resources:
#   - CodeBuild project
#   - CodeBuild S3 source bucket
#
# Usage:
#   export PULUMI_CONFIG_PASSPHRASE="benchplatform2026"
#   bash scripts/aws_teardown.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
export HOME=/home/valentin
export PATH="/home/valentin/.local/bin:/home/valentin/.pulumi/bin:/usr/local/bin:/usr/bin:/bin"

REGION="us-east-1"
AWS_ACCOUNT="564175165906"

cd /home/valentin/cross_bench

echo "=== Step 1: Ensure ASG is at 0 (no running EC2) ==="
ASG_NAME=$(aws autoscaling describe-auto-scaling-groups --region "$REGION" \
    --query "AutoScalingGroups[?contains(AutoScalingGroupName,'cross-bench')].AutoScalingGroupName" \
    --output text 2>/dev/null || echo "")
if [ -n "$ASG_NAME" ]; then
    echo "  Scaling $ASG_NAME to 0..."
    aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG_NAME" \
        --desired-capacity 0 --region "$REGION" 2>/dev/null || true
    sleep 5
fi

echo ""
echo "=== Step 2: Empty S3 buckets (required before Pulumi destroy) ==="
for BUCKET in cross-bench-dev-artifacts cross-bench-dev-datasets; do
    FULL=$(aws s3api list-buckets --query "Buckets[?starts_with(Name,'${BUCKET}')].Name" --output text 2>/dev/null || echo "")
    if [ -n "$FULL" ] && [ "$FULL" != "None" ]; then
        echo "  Emptying $FULL..."
        aws s3 rm "s3://${FULL}" --recursive --region "$REGION" 2>/dev/null || true
        # Also delete version markers for versioned buckets
        aws s3api list-object-versions --bucket "$FULL" --region "$REGION" \
            --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' --output json 2>/dev/null | \
            python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('Objects'):
    print(json.dumps(data))
" | xargs -r -I{} aws s3api delete-objects --bucket "$FULL" --delete '{}' --region "$REGION" 2>/dev/null || true
        # Delete markers too
        aws s3api list-object-versions --bucket "$FULL" --region "$REGION" \
            --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' --output json 2>/dev/null | \
            python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('Objects'):
    print(json.dumps(data))
" | xargs -r -I{} aws s3api delete-objects --bucket "$FULL" --delete '{}' --region "$REGION" 2>/dev/null || true
    fi
done

echo ""
echo "=== Step 3: Delete ECR images (required before Pulumi destroy) ==="
for REPO in cross-bench-orchestrator cross-bench-function cross-bench-bigcodebench cross-bench-terminal; do
    IMAGES=$(aws ecr list-images --repository-name "$REPO" --region "$REGION" \
        --query 'imageIds[*]' --output json 2>/dev/null || echo "[]")
    if [ "$IMAGES" != "[]" ] && [ -n "$IMAGES" ]; then
        echo "  Deleting images in $REPO..."
        aws ecr batch-delete-image --repository-name "$REPO" --image-ids "$IMAGES" \
            --region "$REGION" 2>/dev/null || true
    fi
done

echo ""
echo "=== Step 4: Pulumi destroy (all managed resources) ==="
pulumi destroy --yes --skip-preview

echo ""
echo "=== Step 5: Clean up non-Pulumi resources ==="

# CodeBuild project
echo "  Deleting CodeBuild project..."
aws codebuild delete-project --name cross-bench-build --region "$REGION" 2>/dev/null || true

# CodeBuild source bucket
CB_BUCKET="codebuild-cross-bench-source-${AWS_ACCOUNT}"
if aws s3api head-bucket --bucket "$CB_BUCKET" --region "$REGION" 2>/dev/null; then
    echo "  Emptying and deleting CodeBuild source bucket..."
    aws s3 rm "s3://${CB_BUCKET}" --recursive --region "$REGION" 2>/dev/null || true
    aws s3api delete-bucket --bucket "$CB_BUCKET" --region "$REGION" 2>/dev/null || true
fi

# CodeBuild IAM role (if created by setup_codebuild.sh)
echo "  Cleaning up CodeBuild IAM role..."
for POLICY_ARN in \
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser" \
    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess" \
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"; do
    aws iam detach-role-policy --role-name codebuild-cross-bench-role \
        --policy-arn "$POLICY_ARN" 2>/dev/null || true
done
aws iam delete-role --role-name codebuild-cross-bench-role 2>/dev/null || true

echo ""
echo "=== Step 6: Verification ==="
echo ""
echo "Checking for remaining cross-bench resources..."

ERRORS=0

# Check EC2
EC2=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=*cross-bench*" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
    --query 'Reservations[*].Instances[*].InstanceId' --output text 2>/dev/null || echo "")
if [ -n "$EC2" ] && [ "$EC2" != "None" ]; then
    echo "  WARNING: EC2 instances still exist: $EC2"
    ERRORS=$((ERRORS + 1))
else
    echo "  EC2: clean"
fi

# Check ECS
ECS=$(aws ecs describe-clusters --clusters cross-bench-dev-cluster-8d6dec0 --region "$REGION" \
    --query 'clusters[?status!=`INACTIVE`].clusterName' --output text 2>/dev/null || echo "")
if [ -n "$ECS" ] && [ "$ECS" != "None" ]; then
    echo "  WARNING: ECS cluster still active: $ECS"
    ERRORS=$((ERRORS + 1))
else
    echo "  ECS: clean"
fi

# Check RDS
RDS=$(aws rds describe-db-instances --region "$REGION" \
    --query "DBInstances[?contains(DBInstanceIdentifier,'cross-bench')].DBInstanceIdentifier" \
    --output text 2>/dev/null || echo "")
if [ -n "$RDS" ] && [ "$RDS" != "None" ]; then
    echo "  WARNING: RDS instance still exists: $RDS"
    ERRORS=$((ERRORS + 1))
else
    echo "  RDS: clean"
fi

# Check ECR
ECR=$(aws ecr describe-repositories --region "$REGION" \
    --query "repositories[?starts_with(repositoryName,'cross-bench')].repositoryName" \
    --output text 2>/dev/null || echo "")
if [ -n "$ECR" ] && [ "$ECR" != "None" ]; then
    echo "  WARNING: ECR repos still exist: $ECR"
    ERRORS=$((ERRORS + 1))
else
    echo "  ECR: clean"
fi

# Check S3
S3=$(aws s3api list-buckets \
    --query "Buckets[?starts_with(Name,'cross-bench') || starts_with(Name,'codebuild-cross-bench')].Name" \
    --output text 2>/dev/null || echo "")
if [ -n "$S3" ] && [ "$S3" != "None" ]; then
    echo "  WARNING: S3 buckets still exist: $S3"
    ERRORS=$((ERRORS + 1))
else
    echo "  S3: clean"
fi

# Check NAT Gateways (common cost leak)
NAT=$(aws ec2 describe-nat-gateways --region "$REGION" \
    --filter "Name=tag:Name,Values=*cross-bench*" "Name=state,Values=available" \
    --query 'NatGateways[*].NatGatewayId' --output text 2>/dev/null || echo "")
if [ -n "$NAT" ] && [ "$NAT" != "None" ]; then
    echo "  WARNING: NAT gateways still exist: $NAT"
    ERRORS=$((ERRORS + 1))
else
    echo "  NAT Gateways: clean"
fi

# Check CodeBuild
CB=$(aws codebuild batch-get-projects --names cross-bench-build --region "$REGION" \
    --query 'projects[0].name' --output text 2>/dev/null || echo "")
if [ -n "$CB" ] && [ "$CB" != "None" ]; then
    echo "  WARNING: CodeBuild project still exists: $CB"
    ERRORS=$((ERRORS + 1))
else
    echo "  CodeBuild: clean"
fi

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "All cross-bench resources destroyed successfully."
    echo ""
    echo "To recreate: bash scripts/aws_setup.sh"
else
    echo "WARNING: $ERRORS resource(s) may still exist. Check above."
fi
