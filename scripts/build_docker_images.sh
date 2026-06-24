#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="${SCRIPT_DIR}/../src/cross_bench/execution/docker_templates"

BENCHMARK="${1:-all}"

build_image() {
    local name="$1"
    local dockerfile="$2"
    echo "Building ${name}..."
    docker build -t "${name}:latest" -f "${dockerfile}" "${TEMPLATES_DIR}"
}

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "base" ]]; then
    build_image "cross-bench-base" "${TEMPLATES_DIR}/base.Dockerfile"
fi

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "function" || "$BENCHMARK" == "humaneval" || "$BENCHMARK" == "mbpp" ]]; then
    build_image "cross-bench-function" "${TEMPLATES_DIR}/function_level.Dockerfile"
fi

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "bigcodebench" ]]; then
    build_image "cross-bench-bigcodebench" "${TEMPLATES_DIR}/bigcodebench.Dockerfile"
fi

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "terminal" || "$BENCHMARK" == "terminalbench" ]]; then
    build_image "cross-bench-terminal" "${TEMPLATES_DIR}/terminal.Dockerfile"
fi

echo "Done."
