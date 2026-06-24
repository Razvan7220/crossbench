"""Pulumi IaC entry point — deploys the cross-bench stack to AWS."""

import pulumi

from cross_bench.infra.pulumi.codebuild import (
    create_codebuild,
    create_swebench_mirror_codebuild,
    create_swebench_pro_mirror_codebuild,
)
from cross_bench.infra.pulumi.ecr import (
    create_build_cache_repo,
    create_ecr_repos,
    create_swebench_mirror_repo,
    create_swebench_pro_mirror_repo,
)
from cross_bench.infra.pulumi.ecs import create_ecs_cluster
from cross_bench.infra.pulumi.rds import create_database
from cross_bench.infra.pulumi.s3 import create_s3_buckets
from cross_bench.infra.pulumi.vpc import create_vpc

config = pulumi.Config()
env = config.get("environment") or "dev"
project_name = f"cross-bench-{env}"

# VPC
vpc, subnets, security_groups = create_vpc(project_name)

# S3
artifacts_bucket, datasets_bucket = create_s3_buckets(project_name)

# RDS
db_instance, db_url = create_database(project_name, vpc, subnets, security_groups)

# ECR
ecr_repos = create_ecr_repos(project_name)

# SWE-bench mirror repo (single repo, one tag per instance). See
# `ecr.py::create_swebench_mirror_repo` and `deviations.md` S4.
swebench_mirror_repo = create_swebench_mirror_repo(project_name)

# SWE-bench Pro mirror repo (sibling — Scale AI's harder benchmark,
# 731 tasks, multi-language). See `deviations.md` S21.
swebench_pro_mirror_repo = create_swebench_pro_mirror_repo(project_name)

# Orchestrator BuildKit cache repo. CodeBuild's ``--cache-from`` /
# ``--cache-to type=registry`` writes per-layer cache blobs into this
# repo, giving us warm builds across CodeBuild instance rotations
# (which the per-instance ``LOCAL_DOCKER_LAYER_CACHE`` couldn't).
# See ``ecr.py::create_build_cache_repo`` and workflow.md "Build cache".
orchestrator_cache_repo = create_build_cache_repo(project_name)

# CodeBuild (builds + pushes images into the ECR repos above)
codebuild_project = create_codebuild(project_name, ecr_repos)

# CodeBuild project that one-shots the upstream SWE-bench image mirror.
# Triggered manually via `aws codebuild start-build`; see
# `workflow.md` "AWS > Mirroring SWE-bench images".
swebench_mirror_project = create_swebench_mirror_codebuild(
    project_name, swebench_mirror_repo
)

# Same shape for SWE-bench Pro — different source repo + buildspec.
# See `workflow.md` "AWS > Mirroring Pro images".
swebench_pro_mirror_project = create_swebench_pro_mirror_codebuild(
    project_name, swebench_pro_mirror_repo
)

# ECS
ecs_cluster, task_def = create_ecs_cluster(
    project_name,
    vpc=vpc,
    subnets=subnets,
    security_groups=security_groups,
    db_url=db_url,
    artifacts_bucket=artifacts_bucket,
    ecr_repos=ecr_repos,
    swebench_mirror_repo=swebench_mirror_repo,
)

# Exports
pulumi.export("vpc_id", vpc.id)
pulumi.export("artifacts_bucket", artifacts_bucket.bucket)
pulumi.export("datasets_bucket", datasets_bucket.bucket)
pulumi.export("db_endpoint", db_instance.endpoint)
pulumi.export("ecs_cluster", ecs_cluster.name)
pulumi.export("task_def_arn", task_def.arn)
pulumi.export("ecr_repos", {name: repo.repository_url for name, repo in ecr_repos.items()})
pulumi.export("swebench_mirror_repo", swebench_mirror_repo.repository_url)
pulumi.export("swebench_pro_mirror_repo", swebench_pro_mirror_repo.repository_url)
pulumi.export("orchestrator_cache_repo", orchestrator_cache_repo.repository_url)
pulumi.export("public_subnet_ids", [s.id for s in subnets["public"]])
pulumi.export("ecs_security_group_id", security_groups["ecs"].id)
pulumi.export("codebuild_project", codebuild_project.name)
pulumi.export("swebench_mirror_codebuild_project", swebench_mirror_project.name)
pulumi.export(
    "swebench_pro_mirror_codebuild_project", swebench_pro_mirror_project.name
)
