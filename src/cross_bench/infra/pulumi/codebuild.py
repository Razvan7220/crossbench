"""CodeBuild project that builds Docker images and pushes them to ECR.

Source = GitHub Enterprise (``devotion-labs.ghe.com/research/CrossBench``).
Auth is a Personal Access Token managed in-band as a Pulumi
``aws.codebuild.SourceCredential`` resource. Set the token once on the
stack with::

    pulumi config set --secret ghe_pat <PAT>
    pulumi up

The PAT is encrypted in stack state, registered with CodeBuild on
``pulumi up``, and CodeBuild looks it up automatically on build start.
Rotation = ``pulumi config set --secret ghe_pat <new>`` + ``pulumi up``.

Why CodeBuild rather than local ``docker build`` + ``docker push``:

* Developers on Windows hit Defender / ASR blocks on docker-cli
  execution; running the build inside AWS sidesteps that entirely.
* The orchestrator image bundles ``[llm,embedding,memory-mem0]``
  extras — that's hundreds of MB of wheels. Building in-cloud keeps
  large pushes off the developer's uplink.
"""

from __future__ import annotations

import json

import pulumi
import pulumi_aws as aws

GHE_REPO_URL = "https://devotion-labs.ghe.com/research/CrossBench.git"


def create_codebuild(
    project_name: str,
    ecr_repos: dict[str, aws.ecr.Repository],
) -> aws.codebuild.Project:
    # GHE Personal Access Token, registered with CodeBuild as a
    # SourceCredential. Account+region scoped — only one of these per
    # (server_type, region), so creating it here is also the stack-level
    # auth surface for any future CodeBuild project against the same GHE.
    ghe_pat = pulumi.Config().require_secret("ghe_pat")
    ghe_credential = aws.codebuild.SourceCredential(
        f"{project_name}-ghe-credential",
        server_type="GITHUB_ENTERPRISE",
        auth_type="PERSONAL_ACCESS_TOKEN",
        token=ghe_pat,
    )

    role = aws.iam.Role(
        f"{project_name}-codebuild-role",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "codebuild.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        }),
    )

    region = aws.get_region().name
    account_id = aws.get_caller_identity().account_id
    cb_project_name = f"{project_name}-build"

    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr-public:GetAuthorizationToken",
                    "sts:GetServiceBearerToken",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:PutImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                ],
                "Resource": f"arn:aws:ecr:{region}:{account_id}:repository/cross-bench-*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/{cb_project_name}:*",
            },
        ],
    })

    aws.iam.RolePolicy(
        f"{project_name}-codebuild-policy",
        role=role.id,
        policy=policy,
    )

    project = aws.codebuild.Project(
        f"{project_name}-build",
        name=cb_project_name,
        description="Builds cross-bench Docker images and pushes to ECR.",
        service_role=role.arn,
        build_timeout=30,
        source=aws.codebuild.ProjectSourceArgs(
            type="GITHUB_ENTERPRISE",
            location=GHE_REPO_URL,
            # buildspec.yml lives at the repo root, CodeBuild auto-detects it.
            git_clone_depth=1,
            git_submodules_config=aws.codebuild.ProjectSourceGitSubmodulesConfigArgs(
                # The vendored SEAL submodules (third_party/swe_agent,
                # third_party/swebench_pro) are public github.com repos.
                # The buildspec's pre_build step runs
                # `git submodule update --init --recursive` explicitly and
                # hermetically, so we keep CodeBuild's native submodule
                # fetch off here.
                fetch_submodules=False,
            ),
        ),
        # Default branch/ref for manual start-build invocations. Callers
        # can override per-build via ``--source-version``.
        source_version="main",
        artifacts=aws.codebuild.ProjectArtifactsArgs(type="NO_ARTIFACTS"),
        environment=aws.codebuild.ProjectEnvironmentArgs(
            type="LINUX_CONTAINER",
            image="aws/codebuild/amazonlinux2-x86_64-standard:5.0",
            compute_type="BUILD_GENERAL1_MEDIUM",
            # privileged_mode is required for docker-in-docker builds.
            privileged_mode=True,
        ),
        # LOCAL cache persists Docker layers and source on the CodeBuild
        # instance's disk across builds. Without it every build re-runs
        # the apt-get + ``pip install -c constraints.txt ...`` layers
        # from scratch (~8-10 min per build); with it, Python-only edits
        # rebuild in ~30 sec because Stage 1 hits the layer cache. The
        # cache is best-effort — CodeBuild rotates instances over time,
        # so a cache-cold build is still possible; that's tolerable.
        # LOCAL_SOURCE_CACHE also avoids re-cloning GHE on each build.
        cache=aws.codebuild.ProjectCacheArgs(
            type="LOCAL",
            modes=["LOCAL_DOCKER_LAYER_CACHE", "LOCAL_SOURCE_CACHE"],
        ),
        # Explicit dependency on ECR repos (the build script pushes into
        # them) and on the GHE credential (the build clones from GHE) —
        # both must exist before the first build fires.
        opts=pulumi.ResourceOptions(
            depends_on=[*ecr_repos.values(), ghe_credential],
        ),
    )

    return project


def create_swebench_mirror_codebuild(
    project_name: str,
    swebench_repo: aws.ecr.Repository,
) -> aws.codebuild.Project:
    """One-shot CodeBuild that mirrors SWE-bench per-instance images.

    Triggered manually:

        aws codebuild start-build \\
            --project-name cross-bench-dev-mirror-swebench-images \\
            --region us-east-1

    The build reads upstream SWE-bench Verified instance IDs, then for
    each:

        docker pull swebench/sweb.eval.x86_64.<id_1776>:latest
        docker tag <pulled> <ecr_uri>:<id_with_double_underscore>
        docker push
        docker rmi (free disk — CodeBuild local disk is finite)

    Buildspec lives at ``buildspec.mirror.yml`` and invokes
    ``scripts/mirror_swebench_images.py`` for the per-instance loop.

    Compute sizing: BUILD_GENERAL1_LARGE has ~150 GB local disk, vs
    MEDIUM's ~50 GB. Each upstream image is ~5 GB compressed +
    ~12 GB uncompressed during pull; we ``docker rmi`` aggressively
    between iterations but want headroom for parallel layer pulls.

    Build timeout: 8 h ceiling (must finish within
    ``480`` minutes). Realistic anonymous-pull wall-clock for the
    500-instance Verified set is ~25-30 h (Docker Hub anon rate-limit
    is 100 pulls / 6 h); to fit within the timeout, the script
    chunks via the ``LIMIT`` and ``OFFSET`` env vars so multiple
    sequential builds can mirror the full set.
    """
    ghe_pat = pulumi.Config().require_secret("ghe_pat")
    # ``ghe_credential`` is created in ``create_codebuild`` and is
    # account+region-scoped — only one per (server_type, region). The
    # second CodeBuild project reuses the same credential automatically;
    # no separate ``SourceCredential`` resource is needed here.

    role = aws.iam.Role(
        f"{project_name}-mirror-codebuild-role",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "codebuild.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        }),
    )

    region = aws.get_region().name
    account_id = aws.get_caller_identity().account_id
    cb_project_name = f"{project_name}-mirror-swebench-images"

    # ``swebench_repo.arn`` is a Pulumi Output and cannot go through
    # plain ``json.dumps``. ``pulumi.Output.json_dumps`` resolves every
    # Output inside the structure first, then serializes. Under
    # pulumi-aws 7.x (where this previously "worked" implicitly under
    # 6.x via some stringification path) the strict-typed Output API
    # is enforced.
    policy = pulumi.Output.json_dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:DescribeImages",
                    "ecr:PutImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                ],
                "Resource": swebench_repo.arn,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/{cb_project_name}:*",
            },
        ],
    })

    aws.iam.RolePolicy(
        f"{project_name}-mirror-codebuild-policy",
        role=role.id,
        policy=policy,
    )

    return aws.codebuild.Project(
        f"{project_name}-mirror-build",
        name=cb_project_name,
        description=(
            "Mirrors SWE-bench Verified per-instance images from Docker "
            "Hub (swebench/sweb.eval.x86_64.<id_1776>:latest) to ECR "
            "(swebench:<id>). One-shot; manually triggered. Override "
            "LIMIT and OFFSET env vars on start-build to chunk."
        ),
        service_role=role.arn,
        # 8 h ceiling. Anonymous Docker Hub rate-limit is 100/6h; expect
        # multiple invocations to cover the full 500-instance set.
        build_timeout=480,
        source=aws.codebuild.ProjectSourceArgs(
            type="GITHUB_ENTERPRISE",
            location=GHE_REPO_URL,
            buildspec="buildspec.mirror.yml",
            git_clone_depth=1,
        ),
        source_version="main",
        artifacts=aws.codebuild.ProjectArtifactsArgs(type="NO_ARTIFACTS"),
        environment=aws.codebuild.ProjectEnvironmentArgs(
            type="LINUX_CONTAINER",
            image="aws/codebuild/amazonlinux2-x86_64-standard:5.0",
            # LARGE for the ~150 GB local disk (vs MEDIUM's ~50 GB).
            # The mirror script docker-rmis after each push, but we want
            # headroom for parallel layer pulls.
            compute_type="BUILD_GENERAL1_LARGE",
            privileged_mode=True,
            environment_variables=[
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="ECR_REPO_URI",
                    value=swebench_repo.repository_url,
                ),
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="AWS_REGION",
                    value=region,
                ),
                # Default range — caller can override on start-build to
                # chunk across multiple invocations. LIMIT=0 means "all".
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="LIMIT",
                    value="0",
                ),
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="OFFSET",
                    value="0",
                ),
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="SWEBENCH_VARIANT",
                    value="verified",
                ),
            ],
        ),
        opts=pulumi.ResourceOptions(
            depends_on=[swebench_repo],
        ),
    )


def create_swebench_pro_mirror_codebuild(
    project_name: str,
    swebench_pro_repo: aws.ecr.Repository,
) -> aws.codebuild.Project:
    """One-shot CodeBuild that mirrors SWE-bench Pro per-instance images.

    Sibling of :func:`create_swebench_mirror_codebuild`; same shape but
    targeting Scale AI's Pro image set (``jefzda/sweap-images:<tag>`` on
    Docker Hub → ``<ecr>/swebench-pro:<tag>``). Driven by
    ``buildspec.mirror.pro.yml`` and ``scripts/mirror_swebench_pro_images.py``.

    Triggered manually:

        aws codebuild start-build \\
            --project-name cross-bench-dev-mirror-swebench-pro-images \\
            --region us-east-1

    Compute / timeout sizing matches the Verified mirror — same image
    size envelope (~3-5 GB compressed, ~10-15 GB uncompressed) and
    same Docker Hub anon-pull rate-limit ceiling. The 731-instance Pro
    set typically needs 2-3 chunked invocations to clear under
    anonymous pulls; a Docker Hub PAT cuts that to one.

    See ``deviations.md`` S21 and ``workflow.md`` "Mirroring Pro images".
    """
    # ``ghe_credential`` is already created by the first mirror builder
    # (Verified). Account+region-scoped SourceCredentials are reused
    # automatically; no separate resource is needed here.

    role = aws.iam.Role(
        f"{project_name}-pro-mirror-codebuild-role",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "codebuild.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        }),
    )

    region = aws.get_region().name
    account_id = aws.get_caller_identity().account_id
    cb_project_name = f"{project_name}-mirror-swebench-pro-images"

    policy = pulumi.Output.json_dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:GetAuthorizationToken",
                ],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:DescribeImages",
                    "ecr:PutImage",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                ],
                "Resource": swebench_pro_repo.arn,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/{cb_project_name}:*",
            },
        ],
    })

    aws.iam.RolePolicy(
        f"{project_name}-pro-mirror-codebuild-policy",
        role=role.id,
        policy=policy,
    )

    return aws.codebuild.Project(
        f"{project_name}-pro-mirror-build",
        name=cb_project_name,
        description=(
            "Mirrors SWE-bench Pro per-instance images from Docker Hub "
            "(jefzda/sweap-images:<tag>) to ECR (swebench-pro:<tag>). "
            "One-shot; manually triggered. Override LIMIT and OFFSET "
            "env vars on start-build to chunk under Docker Hub anon-"
            "pull rate limits."
        ),
        service_role=role.arn,
        build_timeout=480,
        source=aws.codebuild.ProjectSourceArgs(
            type="GITHUB_ENTERPRISE",
            location=GHE_REPO_URL,
            buildspec="buildspec.mirror.pro.yml",
            git_clone_depth=1,
        ),
        source_version="main",
        artifacts=aws.codebuild.ProjectArtifactsArgs(type="NO_ARTIFACTS"),
        environment=aws.codebuild.ProjectEnvironmentArgs(
            type="LINUX_CONTAINER",
            image="aws/codebuild/amazonlinux2-x86_64-standard:5.0",
            compute_type="BUILD_GENERAL1_LARGE",
            privileged_mode=True,
            environment_variables=[
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="ECR_REPO_URI",
                    value=swebench_pro_repo.repository_url,
                ),
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="AWS_REGION",
                    value=region,
                ),
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="LIMIT",
                    value="0",
                ),
                aws.codebuild.ProjectEnvironmentEnvironmentVariableArgs(
                    name="OFFSET",
                    value="0",
                ),
            ],
        ),
        opts=pulumi.ResourceOptions(
            depends_on=[swebench_pro_repo],
        ),
    )
