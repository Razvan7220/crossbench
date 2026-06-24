#!/bin/bash
export PATH="/home/valentin/.local/bin:$PATH"
CLUSTER="cross-bench-dev-cluster-ebd51df"
REGION="us-east-1"

for i in $(seq 1 24); do
    sleep 15
    COUNT=$(aws ecs list-container-instances --cluster "$CLUSTER" --region "$REGION" \
        --query 'length(containerInstanceArns)' --output text 2>/dev/null || echo 0)
    echo "$(date +%H:%M:%S) | Registered instances: $COUNT"
    if [ "$COUNT" -ge 2 ]; then
        echo "=== 2 instances ready ==="
        exit 0
    fi
done
echo "ERROR: Timed out waiting for 2 instances"
exit 1
