#!/bin/bash
set -euo pipefail
export HOME=/home/valentin
export PATH="/home/valentin/.local/bin:/home/valentin/.pulumi/bin:/usr/local/bin:/usr/bin:/bin"
export PULUMI_CONFIG_PASSPHRASE="benchplatform2026"

REGION="us-east-1"
CLUSTER="cross-bench-dev-cluster-ebd51df"
TASK_DEF="cross-bench-dev-orchestrator"
ASG="cross-bench-dev-asg-d1290e6"
AWS_ACCOUNT="564175165906"
ECR_URL="${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
SANDBOX_IMAGE="${ECR_URL}/cross-bench-function:latest"

# Ensure EC2 instance is running
INSTANCE_COUNT=$(aws ecs list-container-instances --cluster "$CLUSTER" --region "$REGION" \
    --query 'length(containerInstanceArns)' --output text 2>/dev/null || echo 0)

if [ "$INSTANCE_COUNT" -eq 0 ]; then
    echo "=== Scaling ASG to 1 — waiting for EC2 to register ==="
    aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" \
        --desired-capacity 1 --region "$REGION"

    for i in $(seq 1 30); do
        sleep 15
        COUNT=$(aws ecs list-container-instances --cluster "$CLUSTER" --region "$REGION" \
            --query 'length(containerInstanceArns)' --output text 2>/dev/null || echo 0)
        echo "  $(date +%H:%M:%S) | Registered instances: $COUNT"
        [ "$COUNT" -gt 0 ] && break
    done
fi

# Pre-pull the sandbox image on the EC2 host via SSM (if needed)
# The ECS agent already authenticates with ECR, but the Docker daemon
# on the host needs the function image available for sibling containers.
echo "=== Ensuring sandbox image is on EC2 host ==="
INSTANCE_ID=$(aws ecs list-container-instances --cluster "$CLUSTER" --region "$REGION" \
    --query 'containerInstanceArns[0]' --output text | xargs -I{} \
    aws ecs describe-container-instances --cluster "$CLUSTER" --container-instances {} \
    --region "$REGION" --query 'containerInstances[0].ec2InstanceId' --output text)

aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[
        'aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URL',
        'docker pull $SANDBOX_IMAGE'
    ]" \
    --region "$REGION" \
    --query 'Command.CommandId' --output text

# Give SSM a moment to execute
sleep 15

# Accept parameters (defaults to HumanEval + gpt-4o baseline)
PROFILE="${1:-online_no_memory}"
BENCHMARK="${2:-humaneval}"

echo "=== Launching $BENCHMARK benchmark (profile: $PROFILE) ==="
TASK_ARN=$(aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$TASK_DEF" \
    --launch-type EC2 \
    --count 1 \
    --overrides '{
        "containerOverrides": [{
            "name": "orchestrator",
            "command": [
                "run",
                "--profile", "'"$PROFILE"'",
                "--benchmark", "'"$BENCHMARK"'",
                "--mode", "single_shot"
            ],
            "environment": [
                {"name": "BENCH_SANDBOX_IMAGE", "value": "'"$SANDBOX_IMAGE"'"}
            ]
        }]
    }' \
    --region "$REGION" \
    --query 'tasks[0].taskArn' --output text)

echo "Task ARN: $TASK_ARN"
TASK_ID=$(basename "$TASK_ARN")

echo ""
echo "=== Polling task status (every 30s) ==="
while true; do
    STATUS=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" \
        --region "$REGION" --query 'tasks[0].lastStatus' --output text 2>/dev/null || echo "UNKNOWN")
    echo "  $(date +%H:%M:%S) | $STATUS"
    case "$STATUS" in
        STOPPED|DEPROVISIONING) break ;;
    esac
    sleep 30
done

EXIT_CODE=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" \
    --region "$REGION" --query 'tasks[0].containers[0].exitCode' --output text 2>/dev/null || echo "?")
STOP_REASON=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" \
    --region "$REGION" --query 'tasks[0].stoppedReason' --output text 2>/dev/null || echo "?")

echo ""
echo "=== Task finished ==="
echo "  Exit code:   $EXIT_CODE"
echo "  Stop reason: $STOP_REASON"
