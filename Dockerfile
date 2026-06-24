# syntax=docker/dockerfile:1.7
#
# Cache strategy:
#   - System apt-get layer is its own cached step (rarely changes).
#   - Dependency install (Stage 1) only invalidates when ``pyproject.toml``
#     changes. ``src/`` is stubbed so hatchling can resolve the package without
#     the real code. A typical source-only edit therefore skips the slow pip
#     resolve+install entirely.
#   - BuildKit ``--mount=type=cache`` keeps pip's download cache across
#     builds, so even when Stage 1 IS invalidated, packages don't re-download
#     from PyPI (resolution still runs but installs are 2-3x faster).
#
# Effect: a "I edited a Python file" rebuild lands at ~30s instead of
# ~15-20min. A "pyproject extras changed" rebuild lands at ~5-8min instead
# of ~15-20min.
#
# Requires the build host to have BuildKit enabled (we set
# DOCKER_BUILDKIT=1 in buildspec.yml).
FROM public.ecr.aws/docker/library/python:3.12-slim

# System deps + Docker CLI (client only — daemon runs on the EC2 host)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential ca-certificates gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Stage 1: dependency install ----
# Copy ONLY pyproject.toml + stub the cross_bench package so hatchling can
# build a (functionally empty) wheel. Pip then resolves and installs every
# optional extra. This layer is cached by Docker as long as pyproject.toml
# doesn't change — meaning a typical source-only edit doesn't re-trigger
# the full install.
#
# Install every runtime extra the orchestrator might exercise:
#   llm       — OpenAI + Anthropic SDKs (every profile uses these)
#   embedding — sentence-transformers (Mastra / Hindsight / Graphiti embedded)
#   memory    — umbrella: mem0, zep, letta, memos, evermemos client SDKs
#   benchmark — umbrella: every benchmark adapter's third-party dependency
#
# A missing extra is silent at image-build time but fails at evaluate()
# time for that adapter — see tests/test_packaging.py for the lock that
# prevents the umbrella from drifting out of sync with pyproject.toml.
COPY pyproject.toml constraints.txt ./
# COPY the vendored submodules so the ``seal`` extra can resolve the
# ``sweagent @ file:./third_party/swe_agent`` direct-URL reference at
# pip-install time. buildspec.yml's ``git submodule update --init
# --recursive`` populates these directories on the CodeBuild host
# before this COPY runs; local ``docker build`` requires the same
# step from the operator (workflow.md SEAL setup recipe).
#
# We COPY ``third_party/`` BEFORE the Stage 1 pip install for two
# reasons: (1) the pip install line below now references the ``seal``
# extra, which can't resolve without the directory present; (2) the
# Docker layer cache key for the pip install line includes the
# ``third_party/`` content hash — bumping a submodule SHA correctly
# invalidates the install layer and re-runs ``pip install``.
COPY third_party/ third_party/

RUN mkdir -p src/cross_bench && \
    printf '"""stub for the dependency-only install layer."""\n' \
        > src/cross_bench/__init__.py
# Use the pip-tools / uv-generated constraints.txt to skip the resolver's
# backtracking entirely — every transitive is pinned at the locally-tested
# version. Without this, pip explores ~80+ ``anthropic`` candidates trying
# to satisfy peer constraints across the heavy ML stack, timing out
# CodeBuild's 30-min wall clock. See ``docs/regenerate-lock.md`` (or
# workflow.md) for how to refresh the lock when bumping a dep.
#
# ``seal`` adds ~50 MB of sweagent deps (LiteLLM, swerex, datasets) to
# the image. Required for ``-b swebench_pro_seal`` dispatches; harmless
# for non-SEAL runs (extra dep weight only).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c constraints.txt ".[llm,embedding,memory,benchmark,seal]"

# ---- Patch swerex docker.py ----
# Seven substitutions to make swerex's standalone-Python build + remote
# runtime work for SWE-bench Pro from a CONTAINERIZED orchestrator (ECS
# task / docker-out-of-docker setup). The script lives at
# ``scripts/patch_swerex.py`` for readability; each substitution
# carries inline rationale. Summary:
#
#   1. bullseye builder pin — GLIBC compat across Pro task images.
#   2. truthy check on ``python_standalone_dir`` — empty-string disables
#      the build (preserves upstream's escape hatch).
#   3. ``--index-url https://pypi.org/simple/`` on swe-rex install —
#      bypasses Pro images' broken devpi pip.conf.
#   4. ``--entrypoint=`` on ``docker run`` — Pro images set
#      ENTRYPOINT=["/bin/bash"] which would wrap swerex's
#      ``/bin/sh -c …`` into a broken concatenation.
#   5. libssl.so.1.1 bundle — Pro Bookworm images ship libssl3 only;
#      the standalone Python (bullseye builder) wants libssl1.1.
#   6. ``--network=container:<orchestrator>`` — when
#      ``/.dockerenv`` is present, share the orchestrator's network
#      namespace so ``127.0.0.1:8000`` reaches the spawned sibling
#      container directly. Without this, the orchestrator's probe at
#      ``http://127.0.0.1:<host-port>`` hits its own loopback (the
#      ``-p`` port mapping lands on the EC2 host's loopback, not the
#      orchestrator container's). Empirically: silent 180s startup
#      timeout → 60min hang on the first NodeBB Pro task of the
#      ``seal-haiku45-bedrock-postmerge`` run.
#   7. Kill the Popen target BEFORE reading stdout/stderr on the
#      startup-timeout error path. Without this, ``.read()`` blocks on
#      the still-running ``docker run`` until our outer subprocess
#      timeout (60min) fires — converting a 180s startup-fail-fast
#      into a 60min hang. Independent of #6 but surfaced by the same
#      run, so co-shipped.
#
# Test (idempotent): ``python scripts/patch_swerex.py``. Each
# substitution short-circuits if its sentinel is already in the file.
COPY scripts/patch_swerex.py /tmp/patch_swerex.py
RUN python3 /tmp/patch_swerex.py

# ---- Stage 2: real source + package overlay ----
# Replace the stub source with the actual code, then re-install just
# cross-bench (--no-deps, since every dep is already in the cached layer
# above). This second install completes in seconds because pip skips
# dependency resolution.
COPY src/ src/
COPY configs/ configs/


RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

ENV BENCH_CONFIGS_DIR=/app/configs

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["cross-bench"]
CMD ["run"]
