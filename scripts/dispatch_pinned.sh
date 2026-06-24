#!/usr/bin/env bash
# Race-proof ECS dispatch — pins the orchestrator image to a specific
# commit SHA tag rather than ``:latest``.
#
# WHY THIS EXISTS
#   ``:latest`` is the ECS task-definition default. The buildspec pushes
#   both ``:latest`` and ``:<short-sha>``, but multiple concurrent builds
#   all overwrite ``:latest`` — whichever finishes last wins. A dispatch
#   that uses the default task-def (``:latest``) reads whatever image
#   happens to be there at task-pull time, which may NOT be the build
#   you just triggered.
#
#   This script registers a new task-definition revision pinned to the
#   short SHA tag, then runs the task against that revision. The pin is
#   immutable so the race is gone.
#
# USAGE
#   scripts/dispatch_pinned.sh <commit-sha-7-or-longer> <run-id> -- <cross-bench CLI args>
#
# EXAMPLES
#   # 50-task BM25 smoke, pinned to commit d13732c:
#   scripts/dispatch_pinned.sh d13732c swebench-mem0v2-bm25-n50-$(date -u +%Y%m%d-%H%M%S) -- \
#     run -p online -p mem0v2_embedded -p mem0_no_infer \
#         -p swebench_retrieval_bm25 -b swebench -n 50
#
#   # Full-Verified Oracle ceiling, pinned to commit deadbee:
#   scripts/dispatch_pinned.sh deadbee swebench-oracle-full-$(date -u +%Y%m%d-%H%M%S) -- \
#     run -p online -p mem0v2_embedded -p mem0_no_infer \
#         -p swebench_retrieval_oracle -b swebench
#
# REQUIREMENTS
#   - The SHA tag must exist in ECR (push happens automatically per build
#     since the buildspec change of 2026-05-21).
#   - AWS CLI configured for us-east-1.
#   - Python 3.x available as ``python`` (used for JSON manipulation —
#     no ``jq`` dependency, so this works on Windows Git Bash too).

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 <commit-sha> <run-id> -- <cross-bench CLI args...>" >&2
  exit 2
fi

SHA="$1"
RUN_ID="$2"
SEP="$3"
if [ "$SEP" != "--" ]; then
  echo "expected '--' separator before CLI args; got '$SEP'" >&2
  exit 2
fi
shift 3
# Remaining args are the cross-bench CLI invocation (``run -p ... -b ...``).
CLI_ARGS=("$@")

REGION="${AWS_REGION:-us-east-1}"
ECR_URL="${ECR_URL:-564175165906.dkr.ecr.us-east-1.amazonaws.com}"
CLUSTER="${CLUSTER:-cross-bench-dev-cluster-a2d8750}"
CAPACITY_PROVIDER="${CAPACITY_PROVIDER:-cross-bench-dev-cp-f3ceab1}"
TASK_DEF_FAMILY="${TASK_DEF_FAMILY:-cross-bench-dev-orchestrator}"

SHORT_SHA="${SHA:0:7}"
PINNED_IMAGE="${ECR_URL}/cross-bench-orchestrator:${SHORT_SHA}"

echo "==> Verifying SHA tag exists in ECR: $PINNED_IMAGE"
if ! aws ecr describe-images \
      --repository-name cross-bench-orchestrator \
      --region "$REGION" \
      --image-ids "imageTag=${SHORT_SHA}" \
      --query 'imageDetails[0].imageDigest' \
      --output text >/dev/null 2>&1; then
  echo "ERROR: tag :${SHORT_SHA} not found in ECR." >&2
  echo "       Did the build that pushed this SHA succeed? Trigger one first:" >&2
  echo "       aws codebuild start-build --project-name cross-bench-dev-build \\" >&2
  echo "           --source-version $SHA --region $REGION" >&2
  exit 3
fi

# Temp scratch files; cleaned up at end (and on any error via trap).
# IMPORTANT: scratch lives in the REPO ROOT, not /tmp. ``mktemp -d -t ...``
# returns ``/tmp/cb-dispatch-XXXX`` on Git Bash for Windows — but the AWS
# CLI is the Windows binary and can't read Unix-style ``/tmp/`` paths
# through ``file:///tmp/...``. Using a worktree-local path works on both
# Linux and Windows because both AWS CLIs see the same filesystem.
SCRATCH_DIR=$(mktemp -d "./.cb-dispatch.XXXXXX")
trap 'rm -rf "$SCRATCH_DIR"' EXIT
TD_RAW="$SCRATCH_DIR/td-raw.json"
TD_NEW="$SCRATCH_DIR/td-new.json"
OVERRIDES="$SCRATCH_DIR/overrides.json"

echo "==> Fetching current task-def family for image-override"
aws ecs describe-task-definition \
  --task-definition "$TASK_DEF_FAMILY" \
  --region "$REGION" \
  --query 'taskDefinition' \
  --output json > "$TD_RAW"

# Build the new task-def JSON: swap image to the pinned SHA tag,
# strip fields that ``register-task-definition`` rejects (it wants the
# original input shape, not the materialized read-shape).
PINNED_IMAGE="$PINNED_IMAGE" python - "$TD_RAW" "$TD_NEW" <<'PY'
import json, os, sys

td_raw, td_new = sys.argv[1], sys.argv[2]
pinned_image = os.environ["PINNED_IMAGE"]

with open(td_raw) as f:
    d = json.load(f)

# Override the orchestrator container's image. The container we want is
# named ``orchestrator``; defensive even if the schema changes order.
for c in d.get("containerDefinitions", []):
    if c.get("name") == "orchestrator":
        c["image"] = pinned_image
        break
else:
    raise SystemExit("ERROR: container 'orchestrator' not found in task-def")

# ``register-task-definition`` rejects materialized read-only fields.
for k in (
    "taskDefinitionArn", "revision", "status", "requiresAttributes",
    "compatibilities", "registeredAt", "registeredBy",
):
    d.pop(k, None)

with open(td_new, "w") as f:
    json.dump(d, f)
PY

echo "==> Registering pinned task-def revision (image=$PINNED_IMAGE)"
NEW_REV=$(aws ecs register-task-definition \
  --region "$REGION" \
  --cli-input-json "file://$TD_NEW" \
  --query 'taskDefinition.revision' \
  --output text)
PINNED_TASK_DEF="${TASK_DEF_FAMILY}:${NEW_REV}"
echo "==> Registered: $PINNED_TASK_DEF"

# Build run-task overrides JSON. CLI_ARGS comes in via env so the Python
# heredoc sees the array intact (heredoc + bash array expansion is awkward
# across shells; this avoids the issue).
echo "==> Dispatching with run_id=$RUN_ID"
RUN_ID="$RUN_ID" CLI_ARGS_JOINED="$(printf '%s\n' "${CLI_ARGS[@]}")" \
  python - "$OVERRIDES" <<'PY'
import json, os, sys

out = sys.argv[1]
run_id = os.environ["RUN_ID"]
cli_args = [a for a in os.environ["CLI_ARGS_JOINED"].split("\n") if a]

overrides = {
    "containerOverrides": [
        {
            "name": "orchestrator",
            "command": cli_args + ["--run-id", run_id],
        }
    ]
}
with open(out, "w") as f:
    json.dump(overrides, f)
PY

aws ecs run-task \
  --cluster "$CLUSTER" \
  --task-definition "$PINNED_TASK_DEF" \
  --capacity-provider-strategy "capacityProvider=${CAPACITY_PROVIDER},weight=1" \
  --region "$REGION" \
  --overrides "file://$OVERRIDES" \
  --query 'tasks[0].{taskArn:taskArn,lastStatus:lastStatus,desiredStatus:desiredStatus}' \
  --output json

echo
echo "==> Dispatched. Artifacts will appear under:"
echo "    s3://cross-bench-dev-artifacts-3684f16/${RUN_ID}/"
