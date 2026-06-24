#!/usr/bin/env python3
"""Mirror SWE-bench Pro per-instance images from Docker Hub to ECR.

Invoked by ``buildspec.mirror.pro.yml`` inside the
``cross-bench-dev-mirror-swebench-pro-images`` CodeBuild project.

For each instance in the ``ScaleAI/SWE-bench_Pro`` HF dataset:

    docker pull jefzda/sweap-images:<dockerhub_tag>
    docker tag  <pulled> ${ECR_REPO_URI}:<dockerhub_tag>
    docker push ${ECR_REPO_URI}:<dockerhub_tag>
    docker rmi  <pulled> <tagged>   # free CodeBuild local disk

Idempotent: skip when ``aws ecr describe-images --image-ids
imageTag=<dockerhub_tag>`` already lists the tag.

Unlike Verified mirroring, no naming translation is needed — Scale AI's
``dockerhub_tag`` field on each row is already a usable Docker tag.
We reuse the same string as the ECR tag so the adapter's
``get_sandbox_image`` (``jefzda/sweap-images:<dockerhub_tag>``) maps
cleanly to ``<ecr>:<dockerhub_tag>`` via the resolver.

See ``deviations.md`` S21 for the Pro variant deviation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

_PRO_DOCKERHUB_REPO = "jefzda/sweap-images"


def _ecr_tag_exists(*, repo_name: str, region: str, tag: str) -> bool:
    """True iff ``aws ecr describe-images`` lists this tag."""
    res = subprocess.run(
        [
            "aws", "ecr", "describe-images",
            "--repository-name", repo_name,
            "--region", region,
            "--image-ids", f"imageTag={tag}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def _run(cmd: list[str], *, check: bool = True) -> int:
    """Run a command; stream stdout/stderr through to the build log."""
    print(f"  + {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd)
    if check and res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {res.returncode}")
    return res.returncode


def _mirror_one(
    dockerhub_tag: str,
    *,
    ecr_repo: str,
    ecr_repo_name: str,
    region: str,
) -> tuple[str, str]:
    """Pull + retag + push one instance. Returns ``(tag, status)``."""
    upstream = f"docker.io/{_PRO_DOCKERHUB_REPO}:{dockerhub_tag}"
    ecr_tag = dockerhub_tag  # identity — Pro tags are already ECR-safe
    ecr_image = f"{ecr_repo}:{ecr_tag}"

    if _ecr_tag_exists(repo_name=ecr_repo_name, region=region, tag=ecr_tag):
        print(f"[{dockerhub_tag}] already in ECR — skipping", flush=True)
        return dockerhub_tag, "already_in_ecr"

    print(f"[{dockerhub_tag}] mirroring {upstream} -> {ecr_image}", flush=True)
    t0 = time.monotonic()
    try:
        _run(["docker", "pull", upstream])
        _run(["docker", "tag", upstream, ecr_image])
        _run(["docker", "push", ecr_image])
    except RuntimeError as e:
        print(f"[{dockerhub_tag}] ERROR: {e}", flush=True, file=sys.stderr)
        # Best-effort cleanup so we don't leave the bad image consuming disk.
        _run(["docker", "rmi", "-f", upstream, ecr_image], check=False)
        return dockerhub_tag, f"error:{e}"

    elapsed = time.monotonic() - t0
    print(f"[{dockerhub_tag}] OK in {elapsed:.1f}s", flush=True)
    # Free disk between iterations — CodeBuild local disk fills fast.
    _run(["docker", "rmi", upstream, ecr_image], check=False)
    return dockerhub_tag, "mirrored"


def _load_dockerhub_tags(limit: int, offset: int) -> list[str]:
    """Load Pro instance dockerhub_tag values from the HF dataset, sliced."""
    from datasets import load_dataset  # local import — heavy

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    total = len(ds)
    start = max(0, offset)
    end = total if limit <= 0 else min(total, start + limit)
    print(
        f"Loaded ScaleAI/SWE-bench_Pro (test split, total={total}); "
        f"slicing [{start}:{end}]",
        flush=True,
    )
    tags: list[str] = []
    for i in range(start, end):
        tag = (ds[i].get("dockerhub_tag") or "").strip()
        if tag:
            tags.append(tag)
        else:
            print(
                f"[warn] row {i} (instance_id={ds[i].get('instance_id')}) "
                f"has empty dockerhub_tag — skipping",
                flush=True,
            )
    return tags


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ecr-repo", required=True,
        help="Full ECR repo URI (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/swebench-pro)",
    )
    p.add_argument(
        "--ecr-repo-name", default="swebench-pro",
        help="ECR repository name (last path segment of --ecr-repo)",
    )
    p.add_argument("--region", required=True)
    p.add_argument(
        "--limit", type=int, default=0,
        help="Max instances to mirror; 0 means all-remaining-after-offset.",
    )
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    tags = _load_dockerhub_tags(args.limit, args.offset)
    print(f"Will mirror {len(tags)} instance(s).\n", flush=True)

    statuses: list[tuple[str, str]] = []
    t0 = time.monotonic()
    for i, tag in enumerate(tags, 1):
        print(f"\n--- [{i}/{len(tags)}] {tag} ---", flush=True)
        statuses.append(_mirror_one(
            tag,
            ecr_repo=args.ecr_repo,
            ecr_repo_name=args.ecr_repo_name,
            region=args.region,
        ))

    elapsed = time.monotonic() - t0
    mirrored = sum(1 for _, s in statuses if s == "mirrored")
    skipped = sum(1 for _, s in statuses if s == "already_in_ecr")
    errors = sum(1 for _, s in statuses if s.startswith("error:"))
    print(
        f"\n=== Summary === total={len(statuses)} "
        f"mirrored={mirrored} skipped={skipped} errors={errors} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
