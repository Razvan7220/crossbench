#!/usr/bin/env python3
"""Mirror SWE-bench per-instance images from Docker Hub to ECR.

Invoked by ``buildspec.mirror.yml`` inside the
``cross-bench-dev-mirror-swebench-images`` CodeBuild project.

For each instance:

    docker pull docker.io/swebench/sweb.eval.x86_64.<id_1776>:latest
    docker tag  <pulled> ${ECR_REPO_URI}:<id_with_double_underscore>
    docker push ${ECR_REPO_URI}:<id_with_double_underscore>
    docker rmi  <pulled> <tagged>  (free local disk)

Idempotent: skip when ``aws ecr describe-images --image-ids
imageTag=<id>`` already lists the tag (covered by ``ecr:DescribeImages``
IAM permission granted to the build role).

The Docker Hub tag uses the ``_1776_`` separator (Princeton's
convention); the ECR tag uses ``__`` so the adapter's
``instance_image_key`` form is the lookup key without translation.

See ``deviations.md`` S4 and ``workflow.md`` "AWS > Mirroring
SWE-bench images".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _instance_id_to_upstream_tag(instance_id: str) -> str:
    """``astropy__astropy-12907`` -> ``astropy_1776_astropy-12907`` (lower)."""
    return instance_id.lower().replace("__", "_1776_")


def _ecr_tag_exists(*, region: str, tag: str) -> bool:
    """True iff ``aws ecr describe-images`` lists this tag."""
    res = subprocess.run(
        [
            "aws", "ecr", "describe-images",
            "--repository-name", "swebench",
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
    instance_id: str,
    *,
    ecr_repo: str,
    region: str,
) -> tuple[str, str]:
    """Pull + retag + push one instance. Returns ``(instance_id, status)``."""
    upstream = (
        f"docker.io/swebench/sweb.eval.x86_64."
        f"{_instance_id_to_upstream_tag(instance_id)}:latest"
    )
    ecr_tag = instance_id.lower()  # mirror what the adapter constructs
    ecr_image = f"{ecr_repo}:{ecr_tag}"

    if _ecr_tag_exists(region=region, tag=ecr_tag):
        print(f"[{instance_id}] already in ECR — skipping", flush=True)
        return instance_id, "already_in_ecr"

    print(f"[{instance_id}] mirroring {upstream} -> {ecr_image}", flush=True)
    t0 = time.monotonic()
    try:
        _run(["docker", "pull", upstream])
        _run(["docker", "tag", upstream, ecr_image])
        _run(["docker", "push", ecr_image])
    except RuntimeError as e:
        print(f"[{instance_id}] ERROR: {e}", flush=True, file=sys.stderr)
        # Best-effort cleanup so we don't leave the bad image consuming disk
        _run(["docker", "rmi", "-f", upstream, ecr_image], check=False)
        return instance_id, f"error:{e}"

    elapsed = time.monotonic() - t0
    print(f"[{instance_id}] OK in {elapsed:.1f}s", flush=True)
    # Free disk between iterations — CodeBuild local disk fills fast.
    _run(["docker", "rmi", upstream, ecr_image], check=False)
    return instance_id, "mirrored"


def _load_instance_ids(variant: str, limit: int, offset: int) -> list[str]:
    """Load instance IDs from the SWE-bench dataset, sliced by limit+offset."""
    from datasets import load_dataset  # local import — heavy

    dataset_name = {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "lite": "princeton-nlp/SWE-bench_Lite",
        "full": "princeton-nlp/SWE-bench",
    }[variant]
    ds = load_dataset(dataset_name, split="test")
    total = len(ds)
    start = max(0, offset)
    end = total if limit <= 0 else min(total, start + limit)
    print(
        f"Loaded {dataset_name} (test split, total={total}); "
        f"slicing [{start}:{end}]",
        flush=True,
    )
    return [ds[i]["instance_id"] for i in range(start, end)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ecr-repo", required=True,
                   help="Full ECR repo URI (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/swebench)")
    p.add_argument("--region", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="Max instances to mirror; 0 means all-remaining-after-offset.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--variant", default="verified",
                   choices=["verified", "lite", "full"])
    args = p.parse_args()

    instance_ids = _load_instance_ids(args.variant, args.limit, args.offset)
    print(f"Will mirror {len(instance_ids)} instance(s).\n", flush=True)

    statuses: list[tuple[str, str]] = []
    t0 = time.monotonic()
    for i, iid in enumerate(instance_ids, 1):
        print(f"\n--- [{i}/{len(instance_ids)}] {iid} ---", flush=True)
        statuses.append(_mirror_one(iid, ecr_repo=args.ecr_repo, region=args.region))

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
