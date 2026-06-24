#!/usr/bin/env bash
#
# Two-cell SWE-bench Pro smoke on AWS ECS.
#
# Cells: no_memory / mem0v2 (episodic).
# Tasks: 10-task stratified set (4 py / 3 go / 2 js / 1 ts, seed=42).
# Total ECS tasks: 20 (2 schemas × 10 tasks × 1 task per shard).
# Wall-clock: ~30-45 min after the orchestrator build SUCCEEDS.
# Cost: ~$10-18 (Anthropic Haiku 4.5 + ECS).
#
# What it does:
#   1. Polls the named CodeBuild build until SUCCEEDED.
#   2. Fires two dispatch_swebench.py invocations in parallel.
#   3. Each dispatch fan-outs 10 ECS tasks, polls until STOPPED,
#      auto-merges into a per-cell summary.json under
#      s3://<bucket>/runs/swebench_pro/<cell>/summary.json
#   4. Prints headline metrics for each cell when both finish.
#
# Usage:
#   SHA=4e76ec2 BUILD_ID=cross-bench-dev-build:38b5fbea-... \
#     bash scripts/smoke_pro_3cells.sh
#
# Overrides:
#   SHA               commit SHA (default: current HEAD short)
#   BUILD_ID          CodeBuild build ID to wait on (default: skip wait)
#   S3_BUCKET         artifacts bucket (default: cross-bench-dev-artifacts-3684f16)

set -euo pipefail

SHA="${SHA:-$(git rev-parse HEAD | cut -c1-7)}"
BUILD_ID="${BUILD_ID:-}"
S3_BUCKET="${S3_BUCKET:-cross-bench-dev-artifacts-3684f16}"
TS=$(date -u +%Y%m%d-%H%M%S)

echo "=== SWE-bench Pro smoke ==="
echo "  SHA       : $SHA"
echo "  Build ID  : ${BUILD_ID:-(none — skipping wait)}"
echo "  S3 bucket : $S3_BUCKET"
echo "  Timestamp : $TS"
echo

# --- 1. Wait for build SUCCEEDED ----------------------------------------
if [ -n "$BUILD_ID" ]; then
  echo "Waiting for CodeBuild SUCCEEDED…"
  until [ "$(aws codebuild batch-get-builds --ids "$BUILD_ID" \
            --region us-east-1 \
            --query 'builds[0].buildStatus' --output text)" = "SUCCEEDED" ]; do
    STATUS=$(aws codebuild batch-get-builds --ids "$BUILD_ID" \
             --region us-east-1 \
             --query 'builds[0].buildStatus' --output text)
    if [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "STOPPED" ] || [ "$STATUS" = "FAULT" ]; then
      echo "ERROR: build $BUILD_ID exited with status $STATUS"
      exit 1
    fi
    echo "  status=$STATUS  (sleep 30)"
    sleep 30
  done
  echo "  build SUCCEEDED"
  echo
fi

# --- 2. Shared task IDs --------------------------------------------------
TASK_IDS=(
  "instance_internetarchive__openlibrary-f343c08f89c772f7ba6c0246f384b9e6c3dc0add-v08d8e8889ec945ab821fb156c04c7d2e2810debb"
  "instance_ansible__ansible-984216f52e76b904e5b0fa0fb956ab4f1e0a7751-v1055803c3a812189a1133297f7f5468579283f86"
  "instance_qutebrowser__qutebrowser-cc360cd4a34a126274c7b51f3b63afbaf3e05a02-v5fc38aaf22415ab0b70567368332beee7955b367"
  "instance_internetarchive__openlibrary-25858f9f0c165df25742acf8309ce909773f0cdd-v13642507b4fc1f8d234172bf8129942da2c2ca26"
  "instance_flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a"
  "instance_gravitational__teleport-2be514d3c33b0ae9188e11ac9975485c853d98bb-vce94f93ad1030e3136852817f2423c1b3ac37bc4"
  "instance_navidrome__navidrome-1e96b858a91c640fe64e84c5e5ad8cc0954ea38d"
  "instance_protonmail__webclients-c6f65d205c401350a226bb005f42fac1754b0b5b"
  "instance_NodeBB__NodeBB-51d8f3b195bddb13a13ddc0de110722774d9bb1b-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e"
  "instance_tutao__tutanota-f373ac3808deefce8183dad8d16729839cc330c1-v2939aa9f4356f0dc9f523ee5ce19d09e08ab979b"
)
TID_ARGS=()
for t in "${TASK_IDS[@]}"; do TID_ARGS+=(--task-id "$t"); done

# --- 3. Dispatch 2 cells in parallel ------------------------------------
NOMEM_RUN="pro-smoke-nomem-$TS"
MEM0_RUN="pro-smoke-mem0v2-$TS"

echo "Dispatching 2 cells in parallel:"
echo "  - $NOMEM_RUN"
echo "  - $MEM0_RUN"
echo

LOG=/tmp/pro-smoke-dispatch-$TS
mkdir -p "$LOG"

python scripts/dispatch_swebench.py --sha "$SHA" --variant pro \
  --profiles online,haiku45,no_memory \
  "${TID_ARGS[@]}" --tasks-per-shard 1 \
  --run-id "$NOMEM_RUN" > "$LOG/nomem.log" 2>&1 &
PID_NOMEM=$!

python scripts/dispatch_swebench.py --sha "$SHA" --variant pro \
  --profiles online,haiku45,mem0v2_embedded,mem0_no_infer \
  "${TID_ARGS[@]}" --tasks-per-shard 1 \
  --run-id "$MEM0_RUN" > "$LOG/mem0.log" 2>&1 &
PID_MEM0=$!

echo "Dispatch PIDs: nomem=$PID_NOMEM mem0=$PID_MEM0"
echo "Logs: $LOG/{nomem,mem0}.log"
echo "Waiting for both to complete…"

wait $PID_NOMEM && echo "nomem: dispatch returned 0" || echo "nomem: dispatch FAILED"
wait $PID_MEM0  && echo "mem0:  dispatch returned 0" || echo "mem0:  dispatch FAILED"

echo
echo "=== Headline metrics ==="
for CELL in "$NOMEM_RUN" "$MEM0_RUN"; do
  echo
  echo "--- $CELL ---"
  aws s3 cp "s3://$S3_BUCKET/runs/swebench_pro/$CELL/summary.json" - 2>/dev/null \
    | jq '.aggregate_metrics | {pass_rate, avg_resolved, total_tasks, passed, failed, avg_steps_taken}' \
    || echo "  (summary not found — check dispatch log at $LOG/)"
done

echo
echo "All artifacts: s3://$S3_BUCKET/runs/swebench_pro/"
