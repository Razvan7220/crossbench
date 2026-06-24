#!/usr/bin/env bash
# Diagnose where a NoPrediction task failed in the pipeline.
#
# Usage: scripts/diag_nopred.sh <task_id> <parent_run_id>
#
# Prints the CloudWatch log timeline and classifies the failure as one of:
#   - "task.start MISSING"      → orchestrator never picked it up
#   - "image pull failed"        → docker pull errored (check sandbox.prefetch_summary errors)
#   - "seal.invoke MISSING"      → image pulled, but sweagent never launched
#   - "sweagent-internal"        → all stages fired, sweagent itself returned with no .pred

set -euo pipefail
TID="${1:?usage: $0 <task_id> <parent_run_id>}"
PARENT="${2:?usage: $0 <task_id> <parent_run_id>}"

echo "============================================="
echo "  TASK:   $TID"
echo "  PARENT: $PARENT"
echo "============================================="

# Single-filter CloudWatch — much faster than dual-pattern. Filter by task_id
# only; we'll grep for parent run_id locally.
LINES=$(aws logs filter-log-events \
  --log-group-name cross-bench-dev-logs-81b05c7 \
  --filter-pattern "\"$TID\"" \
  --start-time $(date -u -d '14 days ago' +%s)000 \
  --region us-east-1 \
  --query 'events[*].message' --output text 2>/dev/null \
  | tr '\t' '\n' | grep "$PARENT" || true)

if [ -z "$LINES" ]; then
  echo "  no log events found — try widening window or check the parent run_id"
  exit 1
fi

echo
echo "=== timeline (5 key markers) ==="
echo "$LINES" | grep -E "task\.start|sandbox\.image_pulled|sandbox\.prefetch_summary|seal\.invoke|task\.complete" | head -10

echo
echo "=== classification ==="
has_start=$(echo "$LINES" | grep -c "task\.start" || true)
has_pull=$(echo "$LINES" | grep -cE "sandbox\.image_pulled" || true)
has_summary=$(echo "$LINES" | grep -c "sandbox\.prefetch_summary" || true)
has_invoke=$(echo "$LINES" | grep -c "seal\.invoke" || true)
has_complete=$(echo "$LINES" | grep -c "task\.complete" || true)

prefetch_errors=$(echo "$LINES" | grep "sandbox\.prefetch_summary" | head -1 | grep -oE "errors=[0-9]+" | grep -oE "[0-9]+" || echo "0")

[ "$has_start" -gt 0 ]    && echo "  ✓ task.start"               || echo "  ✗ task.start MISSING (engine never picked it up)"
[ "$has_pull" -gt 0 ]     && echo "  ✓ sandbox.image_pulled"     || echo "  ✗ sandbox.image_pulled MISSING (image pull failed/skipped)"
[ "$has_summary" -gt 0 ]  && echo "  ✓ sandbox.prefetch_summary (errors=$prefetch_errors)" \
                          || echo "  ✗ sandbox.prefetch_summary MISSING"
[ "$has_invoke" -gt 0 ]   && echo "  ✓ seal.invoke (sweagent started)" \
                          || echo "  ✗ seal.invoke MISSING (sweagent never started)"
[ "$has_complete" -gt 0 ] && echo "  ✓ task.complete"            || echo "  ✗ task.complete MISSING (run dropped)"

echo
echo "=== verdict ==="
if [ "$prefetch_errors" -gt 0 ]; then
  echo "  IMAGE PULL FAILURE — docker pull failed. Check Docker Hub auth/rate limit or ECR mirror status."
elif [ "$has_pull" -gt 0 ] && [ "$has_invoke" -eq 0 ]; then
  echo "  ORCHESTRATOR FAILURE BETWEEN PULL AND SWEAGENT — image came down but SEAL never launched."
elif [ "$has_pull" -gt 0 ] && [ "$has_invoke" -gt 0 ]; then
  echo "  SWEAGENT-INTERNAL FAILURE — all stages fired, sweagent ran but didn't produce a .pred."
  echo "  Likely causes:"
  echo "    - swerex container bring-up timed out (slow ~170s death)"
  echo "    - per-instance container ENTRYPOINT exited immediately"
  echo "    - sweagent instance validation rejected the task ID"
elif [ "$has_start" -eq 0 ]; then
  echo "  ENGINE NEVER STARTED THIS TASK — orchestrator may have crashed early."
fi
