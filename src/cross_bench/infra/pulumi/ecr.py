"""ECR repositories for sandbox Docker images."""

import pulumi_aws as aws

SANDBOX_IMAGES = [
    "cross-bench-function",
    "cross-bench-bigcodebench",
    "cross-bench-terminal",
    "cross-bench-orchestrator",
]

# Name of the single repo that mirrors SWE-bench per-instance images.
# All 500 (Verified) or ~2.3k (full) instances live as tags inside this
# one repo — `<acct>.dkr.ecr.<region>.amazonaws.com/swebench:<instance_id>`.
# See ``deviations.md`` S4 (image prefetch) and the mirror CodeBuild
# project in ``codebuild.py``.
SWEBENCH_MIRROR_REPO = "swebench"

# Name of the single repo that mirrors SWE-bench Pro per-instance images.
# Pro uses Scale AI's ``jefzda/sweap-images:<dockerhub_tag>`` naming on
# Docker Hub; the mirror reuses ``dockerhub_tag`` as the ECR tag so the
# resolver translation is straight identity. See ``deviations.md`` S21.
SWEBENCH_PRO_MIRROR_REPO = "swebench-pro"


def create_ecr_repos(project_name: str) -> dict[str, aws.ecr.Repository]:
    repos = {}
    for image_name in SANDBOX_IMAGES:
        repo = aws.ecr.Repository(
            f"{project_name}-{image_name}",
            name=image_name,
            image_tag_mutability="MUTABLE",
            image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
                scan_on_push=True,
            ),
        )

        # Lifecycle policy: keep last 10 images
        aws.ecr.LifecyclePolicy(
            f"{project_name}-{image_name}-lifecycle",
            repository=repo.name,
            policy="""{
                "rules": [
                    {
                        "rulePriority": 1,
                        "description": "Keep last 10 images",
                        "selection": {
                            "tagStatus": "any",
                            "countType": "imageCountMoreThan",
                            "countNumber": 10
                        },
                        "action": {"type": "expire"}
                    }
                ]
            }""",
        )
        repos[image_name] = repo

    return repos


def create_build_cache_repo(project_name: str) -> aws.ecr.Repository:
    """ECR repo backing the BuildKit registry cache for the orchestrator image.

    BuildKit's ``--cache-to type=registry,mode=max`` pushes each layer as
    its own blob into this repo and tags the manifest list as
    ``:buildcache``. The next build pulls the cache via
    ``--cache-from type=registry,ref=...`` — granular per-layer, so only
    layers whose Dockerfile inputs changed need to re-run.

    Why this exists: CodeBuild's built-in ``LOCAL_DOCKER_LAYER_CACHE`` is
    per-instance only. AWS rotates idle CodeBuild instances after ~15-30
    min, so the local cache survives only inside tight bursts. The
    registry cache persists across instances → consistent warm builds.

    Lifecycle: untagged blobs (intermediate layer objects from
    superseded cache pushes) expire after 7 days. The ``:buildcache``
    tag itself is overwritten on every successful build so we always
    point at the freshest cache state.
    """
    repo = aws.ecr.Repository(
        f"{project_name}-orchestrator-cache",
        name="cross-bench-orchestrator-cache",
        image_tag_mutability="MUTABLE",
        image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
            scan_on_push=False,  # cache layers — scanning is meaningless
        ),
    )
    aws.ecr.LifecyclePolicy(
        f"{project_name}-orchestrator-cache-lifecycle",
        repository=repo.name,
        policy="""{
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Expire untagged layer blobs after 7 days",
                    "selection": {
                        "tagStatus": "untagged",
                        "countType": "sinceImagePushed",
                        "countUnit": "days",
                        "countNumber": 7
                    },
                    "action": {"type": "expire"}
                }
            ]
        }""",
    )
    return repo


def create_swebench_mirror_repo(project_name: str) -> aws.ecr.Repository:
    """ECR repo that holds mirrored SWE-bench per-instance images.

    Schema: one tag per upstream instance, named after the adapter's
    ``instance_image_key`` form (``__`` separator, lower-case). E.g.
    upstream ``swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest``
    lands here as ``<repo>:astropy__astropy-12907``.

    Lifecycle policy: NO expiry. SWE-bench instances are immutable
    (each maps to a specific commit-pinned repo state); once mirrored
    we want them to stay until the leaderboard column itself is
    retired. The 500-tag Verified set sits at ~2.5 TB.
    """
    return aws.ecr.Repository(
        f"{project_name}-{SWEBENCH_MIRROR_REPO}",
        name=SWEBENCH_MIRROR_REPO,
        image_tag_mutability="IMMUTABLE",
        image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
            scan_on_push=False,  # mirroring upstream, not building — scan is noise
        ),
    )


def create_swebench_pro_mirror_repo(project_name: str) -> aws.ecr.Repository:
    """ECR repo that holds mirrored SWE-bench Pro per-instance images.

    Schema: one tag per upstream instance, named after the row's
    ``dockerhub_tag`` field (already ECR-safe — Scale AI publishes them
    that way on Docker Hub). E.g. upstream
    ``jefzda/sweap-images:flipt-io.flipt-2eac0df47b5ecc...`` lands here as
    ``<repo>:flipt-io.flipt-2eac0df47b5ecc...``.

    Lifecycle policy: NO expiry. Pro instances are immutable (each maps
    to a specific commit-pinned repo state). The 731-tag Pro set sits at
    ~2-4 TB.
    """
    return aws.ecr.Repository(
        f"{project_name}-{SWEBENCH_PRO_MIRROR_REPO}",
        name=SWEBENCH_PRO_MIRROR_REPO,
        image_tag_mutability="IMMUTABLE",
        image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
            scan_on_push=False,  # mirroring upstream, not building
        ),
    )
