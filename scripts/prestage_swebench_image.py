#!/usr/bin/env python3
"""Pre-stage SWE-bench per-instance Docker images for local smoke runs.

The CrossBench SWE-bench adapter consumes the upstream
``swebench.harness.test_spec.make_test_spec(...)`` convention:

    instance_image_key = "sweb.eval.x86_64.<instance_id_lower>:latest"

where ``<instance_id_lower>`` uses the double-underscore separator
(e.g. ``astropy__astropy-12907``). Princeton publishes the matching
images on Docker Hub under a different separator (``_1776_``):

    docker.io/swebench/sweb.eval.x86_64.<id_1776>:latest

So a local smoke needs to:

    1. ``docker pull`` from Docker Hub using the ``_1776_`` form.
    2. ``docker tag`` it under the bare ``__`` form so the adapter's
       sandbox launcher finds it locally and skips the pull (which
       would 404 against the default ``library/`` namespace).

This script does both, idempotently. Pass instance IDs on the
command line; pass ``--from-dataset`` to pre-stage the whole
``princeton-nlp/SWE-bench_Verified`` test split (~500 images,
~2 TB of disk --usually NOT what you want locally).

Examples
--------
    # Single instance (the canonical first-row smoke target)
    python scripts/prestage_swebench_image.py astropy__astropy-12907

    # The first N of Verified
    python scripts/prestage_swebench_image.py --from-dataset --limit 5

    # Show what would happen without pulling
    python scripts/prestage_swebench_image.py astropy__astropy-12907 --dry-run

Notes
-----
* Docker daemon must be reachable (``docker info``).
* Anonymous Docker Hub pulls are rate-limited (100 / 6 h). For
  many images use ``docker login`` with a Docker Hub PAT first.
* Each image is ~3-5 GB. Plan disk accordingly --Docker Desktop
  defaults to a 64 GB WSL2 vhdx on Windows.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def _instance_id_to_upstream_tag(instance_id: str) -> str:
    """``astropy__astropy-12907`` -> ``astropy_1776_astropy-12907`` (lower).

    Mirrors Princeton's publishing convention. The base instance_id is
    lowercased to match the adapter's ``instance_image_key`` and
    upstream's ``make_test_spec`` byte-for-byte.
    """
    return instance_id.lower().replace("__", "_1776_")


def _upstream_image(instance_id: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{_instance_id_to_upstream_tag(instance_id)}:latest"


def _local_image(instance_id: str) -> str:
    return f"sweb.eval.x86_64.{instance_id.lower()}:latest"


def _docker_image_exists(image: str) -> bool:
    """``docker image inspect`` returns 0 iff the image is present locally."""
    res = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return res.returncode == 0


def _stage_one(instance_id: str, *, dry_run: bool = False) -> bool:
    """Pull + retag for one instance. Returns True if local image is ready."""
    local = _local_image(instance_id)
    upstream = _upstream_image(instance_id)

    if _docker_image_exists(local):
        print(f"[{instance_id}] already local as {local} --skipping")
        return True

    print(f"[{instance_id}] pulling {upstream}")
    if dry_run:
        print(f"[{instance_id}] DRY RUN --would tag as {local}")
        return False
    pull = subprocess.run(["docker", "pull", upstream])
    if pull.returncode != 0:
        print(f"[{instance_id}] ERROR: docker pull failed (exit {pull.returncode})", file=sys.stderr)
        return False

    print(f"[{instance_id}] tagging as {local}")
    tag = subprocess.run(["docker", "tag", upstream, local])
    if tag.returncode != 0:
        print(f"[{instance_id}] ERROR: docker tag failed (exit {tag.returncode})", file=sys.stderr)
        return False

    print(f"[{instance_id}] OK")
    return True


def _load_verified_instance_ids(limit: int | None) -> list[str]:
    """Read the first ``limit`` instance IDs from SWE-bench Verified.

    Loaded via ``datasets.load_dataset`` so the result matches what
    the CrossBench adapter sees byte-for-byte (same dataset, same
    row ordering).
    """
    from datasets import load_dataset  # local import --datasets is heavy

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    sliced = ds if limit is None else ds.select(range(min(limit, len(ds))))
    return [row["instance_id"] for row in sliced]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "instance_ids", nargs="*",
        help="Instance IDs to stage (e.g. astropy__astropy-12907). "
             "Ignored when --from-dataset is set.",
    )
    p.add_argument(
        "--from-dataset", action="store_true",
        help="Load instance IDs from princeton-nlp/SWE-bench_Verified.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of instances when --from-dataset is set.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without pulling.",
    )
    args = p.parse_args()

    if shutil.which("docker") is None:
        print("ERROR: docker CLI not found on PATH", file=sys.stderr)
        return 2

    if args.from_dataset:
        if args.instance_ids:
            print("WARN: ignoring positional instance_ids when --from-dataset is set",
                  file=sys.stderr)
        instance_ids = _load_verified_instance_ids(args.limit)
    elif args.instance_ids:
        instance_ids = args.instance_ids
    else:
        p.print_help()
        return 2

    print(f"Pre-staging {len(instance_ids)} image(s):")
    ok = 0
    for iid in instance_ids:
        if _stage_one(iid, dry_run=args.dry_run):
            ok += 1
    print(f"\nDone --{ok}/{len(instance_ids)} ready locally.")
    return 0 if ok == len(instance_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
