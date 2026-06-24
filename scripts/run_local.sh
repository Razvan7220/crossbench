#!/usr/bin/env bash
# =============================================================================
# run_local.sh — Run benchmarks locally (no AWS required)
#
# Prerequisites:
#   - Python 3.12+ with venv
#   - Docker running (for sandbox execution)
#   - OPENAI_API_KEY set (for online profiles) OR Ollama running (for local profiles)
#
# Usage:
#   ./scripts/run_local.sh                          # HumanEval, 10 tasks, online_no_memory
#   ./scripts/run_local.sh --limit 164              # Full HumanEval (all tasks)
#   ./scripts/run_local.sh --profile local          # Use local Ollama model
#   ./scripts/run_local.sh --benchmark mbpp         # Different benchmark
#   ./scripts/run_local.sh --profile online_claude  # Use Claude model
# =============================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ── Defaults ───────────────────────────────────────────────────────
PROFILE="${PROFILE:-online_no_memory}"
BENCHMARK="${BENCHMARK:-humaneval}"
LIMIT="${LIMIT:-10}"
MODE="${MODE:-single_shot}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-./artifacts_local}"

# ── Parse CLI args (override defaults) ─────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile|-p)   PROFILE="$2";    shift 2 ;;
        --benchmark|-b) BENCHMARK="$2";  shift 2 ;;
        --limit|-n)     LIMIT="$2";      shift 2 ;;
        --mode)         MODE="$2";       shift 2 ;;
        --artifacts)    ARTIFACTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Colours ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}▸ $*${NC}"; }
ok()    { echo -e "${GREEN}✓ $*${NC}"; }
fail()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ── Pre-flight ─────────────────────────────────────────────────────
info "Pre-flight checks..."

# Docker
docker info >/dev/null 2>&1 || fail "Docker is not running"

# Virtual environment
if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install / update package
if ! python -c "import cross_bench" 2>/dev/null; then
    info "Installing cross-bench..."
    pip install -e ".[llm]" -q
fi

# Profile-specific checks
case "$PROFILE" in
    local|local_no_memory)
        docker ps --filter "name=ollama" --format '{{.Names}}' | grep -q ollama \
            || fail "Ollama container not running. Start with: docker compose --profile ollama up -d"
        ;;
    online|online_no_memory)
        [ -n "${OPENAI_API_KEY:-}" ] || fail "OPENAI_API_KEY not set"
        ;;
    online_claude)
        [ -n "${ANTHROPIC_API_KEY:-}" ] || fail "ANTHROPIC_API_KEY not set"
        ;;
esac

ok "Pre-flight passed"

# ── Build sandbox base image ───────────────────────────────────────
if ! docker image inspect cross-bench-base:latest >/dev/null 2>&1; then
    info "Building sandbox base image..."
    bash scripts/build_docker_images.sh base
fi

# ── Run benchmark ──────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  Profile:    $PROFILE"
echo "  Benchmark:  $BENCHMARK"
echo "  Limit:      $LIMIT tasks"
echo "  Mode:       $MODE"
echo "  Artifacts:  $ARTIFACTS_DIR"
echo "────────────────────────────────────────────────────────────────"
echo ""

mkdir -p "$ARTIFACTS_DIR"

LIMIT_FLAG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    LIMIT_FLAG="--limit $LIMIT"
fi

cross-bench run \
    --profile "$PROFILE" \
    --benchmark "$BENCHMARK" \
    --mode "$MODE" \
    --artifacts-dir "$ARTIFACTS_DIR" \
    $LIMIT_FLAG

ok "Benchmark complete! Results in $ARTIFACTS_DIR"
