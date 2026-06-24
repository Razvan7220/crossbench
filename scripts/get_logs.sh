#!/bin/bash
export HOME=/home/valentin
export PATH="/home/valentin/.local/bin:/usr/bin:/bin"

LOG_GROUP="cross-bench-dev-logs-ec7ea6b"
REGION="us-east-1"

STREAM=$(aws logs describe-log-streams \
    --log-group-name "$LOG_GROUP" \
    --region "$REGION" \
    --order-by LastEventTime \
    --descending \
    --query 'logStreams[0].logStreamName' \
    --output text)

echo "Log stream: $STREAM"
echo "---"

aws logs get-log-events \
    --log-group-name "$LOG_GROUP" \
    --log-stream-name "$STREAM" \
    --region "$REGION" \
    --query 'events[*].message' \
    --output text
