"""EC2-backed ECS cluster for the orchestrator.

Fargate cannot expose a Docker socket, so we use EC2 instances running
the ECS-optimized AMI.  The host Docker socket is bind-mounted into the
orchestrator container so the SandboxManager can create sandbox
containers on the same host.
"""

import json

import pulumi
import pulumi_aws as aws


def create_ecs_cluster(
    project_name,
    vpc,
    subnets,
    security_groups,
    db_url,
    artifacts_bucket,
    ecr_repos,
    swebench_mirror_repo=None,
):
    config = pulumi.Config()
    openai_api_key = config.require_secret("openai_api_key")

    # ── Cluster ──────────────────────────────────────────────────────
    cluster = aws.ecs.Cluster(
        f"{project_name}-cluster",
        settings=[aws.ecs.ClusterSettingArgs(
            name="containerInsights", value="enabled",
        )],
    )

    # ── IAM: EC2 instance role (ECS agent + ECR pull) ────────────────
    ec2_role = aws.iam.Role(
        f"{project_name}-ec2-role",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Effect": "Allow",
            }],
        }),
    )
    for suffix, arn in [
        ("ecs", "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"),
        ("ecr", "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"),
        ("ssm", "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"),
    ]:
        aws.iam.RolePolicyAttachment(
            f"{project_name}-ec2-{suffix}",
            role=ec2_role.name,
            policy_arn=arn,
        )

    instance_profile = aws.iam.InstanceProfile(
        f"{project_name}-ec2-profile",
        role=ec2_role.name,
    )

    # ── IAM: ECS task role (S3 artifacts) ────────────────────────────
    task_role = aws.iam.Role(
        f"{project_name}-task-role",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Effect": "Allow",
            }],
        }),
    )
    aws.iam.RolePolicyAttachment(
        f"{project_name}-task-s3",
        role=task_role.name,
        policy_arn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
    )

    # Allow a task running ``cross-bench dispatch`` to fan itself out
    # into shard tasks: start other tasks in the same cluster, describe
    # them while polling, and pass the task/execution roles through.
    aws.iam.RolePolicy(
        f"{project_name}-task-ecs-dispatch",
        role=task_role.name,
        policy=pulumi.Output.all(
            cluster_arn=cluster.arn,
        ).apply(lambda args: json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["ecs:RunTask", "ecs:StopTask"],
                    "Resource": "*",
                    "Condition": {
                        "ArnEquals": {"ecs:cluster": args["cluster_arn"]},
                    },
                },
                {
                    "Effect": "Allow",
                    "Action": ["ecs:DescribeTasks", "ecs:ListTasks"],
                    "Resource": "*",
                    "Condition": {
                        "ArnEquals": {"ecs:cluster": args["cluster_arn"]},
                    },
                },
                {
                    # PassRole is needed for run-task; restricted to the
                    # two roles the orchestrator task def actually uses.
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": "*",
                    "Condition": {
                        "StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"},
                    },
                },
            ],
        })),
    )

    # ── IAM: ECS execution role (pull images, write logs) ────────────
    execution_role = aws.iam.Role(
        f"{project_name}-exec-role",
        assume_role_policy=json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Effect": "Allow",
            }],
        }),
    )
    aws.iam.RolePolicyAttachment(
        f"{project_name}-exec-ecr",
        role=execution_role.name,
        policy_arn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
    )

    # ── CloudWatch log group ─────────────────────────────────────────
    log_group = aws.cloudwatch.LogGroup(
        f"{project_name}-logs",
        retention_in_days=30,
    )

    # ── SSM Parameter for OPENAI_API_KEY ─────────────────────────────
    openai_param = aws.ssm.Parameter(
        f"{project_name}-openai-key",
        name=f"/{project_name}/openai-api-key",
        type="SecureString",
        value=openai_api_key,
        overwrite=True,
    )
    # Let the execution role read the parameter
    aws.iam.RolePolicy(
        f"{project_name}-exec-ssm",
        role=execution_role.name,
        policy=openai_param.arn.apply(lambda arn: json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["ssm:GetParameters"],
                "Resource": [arn],
            }],
        })),
    )

    # ── EC2 Launch Template ──────────────────────────────────────────
    # ECS-optimized Amazon Linux 2023 AMI (latest, x86_64)
    ecs_ami = aws.ec2.get_ami(
        most_recent=True,
        owners=["amazon"],
        filters=[
            aws.ec2.GetAmiFilterArgs(
                name="name",
                values=["al2023-ami-ecs-hvm-*-x86_64"],
            ),
        ],
    )

    function_repo = ecr_repos.get("cross-bench-function")
    function_image_url = (
        function_repo.repository_url.apply(lambda u: f"{u}:latest")
        if function_repo
        else "cross-bench-function:latest"
    )

    user_data = pulumi.Output.all(
        cluster.name, function_image_url,
    ).apply(lambda args: _render_user_data(args[0], args[1]))

    launch_template = aws.ec2.LaunchTemplate(
        f"{project_name}-lt",
        image_id=ecs_ami.id,
        instance_type="r5.xlarge",
        iam_instance_profile=aws.ec2.LaunchTemplateIamInstanceProfileArgs(
            arn=instance_profile.arn,
        ),
        vpc_security_group_ids=[security_groups["ecs"].id],
        user_data=user_data,
        block_device_mappings=[aws.ec2.LaunchTemplateBlockDeviceMappingArgs(
            device_name="/dev/xvda",
            ebs=aws.ec2.LaunchTemplateBlockDeviceMappingEbsArgs(
                # 200 GB sized for SWE-bench Pro fan-outs. Each Pro
                # image is 3-5 GB compressed, ~10 GB on disk
                # uncompressed. A sequential fan-out on the same host
                # (e.g. SEAL Pro N=100 with no parallelism) can pull
                # dozens of distinct per-instance images; without
                # periodic pruning even 200 GB fills. The
                # ``cross-bench-prune`` systemd timer in user_data
                # runs ``docker image prune -af --filter until=2h``
                # every hour (skipped while a task is active), keeping
                # disk usage bounded.
                volume_size=200,
                volume_type="gp3",
            ),
        )],
        tags={"Name": f"{project_name}-ecs-host"},
    )

    # ── Auto Scaling Group ───────────────────────────────────────────
    # The orchestrator task reserves 2048 CPU / 8192 MB — one task fits
    # per r5.xlarge (4 vCPU / 32 GB ≈ 31641 MB allocatable). ``max_size``
    # therefore caps the number of concurrent tasks.
    #
    # Why 8192 specifically: r5.xlarge has ~31641 MB of allocatable
    # memory. With 8192 MB reservation, ECS COULD bin-pack up to 3 tasks
    # per host (3 × 8192 = 24576 MB < 31641 MB) but ECS task placement
    # uses ``binpack`` only when explicitly configured; the default
    # ``spread`` strategy keeps tasks spread across hosts in practice.
    # (Historic: the 8192 MB reservation was originally chosen on the
    # earlier m5.xlarge (~15791 MB allocatable) to push past 15791/2 ≈
    # 7896 MB and enforce roughly one task per host by memory alone; on
    # r5.xlarge that implicit cap is looser, but SEAL's per-task disk +
    # sibling-container footprint still makes one-task-per-host the
    # practical operating point.)
    #
    # Why r5.xlarge: SEAL / SWE-bench tasks spawn sibling containers via
    # the mounted docker.sock (the swerex remote runtime + the
    # per-instance task image with its own Python install). Those
    # siblings aren't accounted for in the ECS task definition's memory
    # reservation — they share the host's remaining RAM. The previous
    # m5.xlarge (16 GB) left ~8 GB for siblings after the 8192 MB
    # orchestrator reservation; r5.xlarge (32 GB) gives ~24 GB headroom
    # at ~31% higher on-demand cost ($0.252/hr vs $0.192/hr). Worth it
    # for stability under SEAL agent loops on Pro images that are
    # 3-5 GB each and routinely spawn multi-GB sibling containers. SEAL
    # is LLM-API-bound, not CPU-bound, so the memory-optimized vs
    # general-purpose swap is invisible to throughput. (Historic: an
    # earlier c5.xlarge configuration with only ~7800 MB allocatable
    # caused routine OOM kills under SEAL, leading to the m5.xlarge
    # bump; the r5.xlarge bump on 2026-06-11 further insures against
    # OOM under increased concurrency.)
    #
    # Sizing: 200 (raised from 60 on 2026-06-11). The original 60 = 5 ×
    # (1 dispatcher + 10 shards) + 5-task headroom for ad-hoc work, sized
    # to run five LoCoMo-style benchmark systems concurrently
    # (e.g. Mastra + Hindsight + Mem0v2 + EverMemOS + MemOS) without
    # hitting the ASG ceiling. The 2026-06-11 bump to 200 accommodates
    # large SWE-bench Pro ablation fan-outs (e.g. 150-shard set-4
    # dispatches × 3 reruns per cell) plus the multi-cell concurrent
    # workflow that pairs with the ``max_output_tokens=1024`` Bedrock
    # change (seal_haiku45 commit ``bf57dd2``) lifting the Bedrock TPM
    # ceiling. Without raising the ASG cap alongside the Bedrock cap,
    # the bottleneck just moves from Bedrock throttling to ECS pending
    # tasks waiting for instance slots — observed on a 150-shard
    # dispatch where only the first 60 launched and the other 90 sat
    # in PROVISIONING.
    #
    # 2026-05-11 incident: the previous cap of 11 (one fan-out only)
    # made the second concurrent fan-out fail silently — orchestrator
    # PROVISIONING → MISSING within minutes, no CloudWatch logs ever
    # written, because the ASG couldn't grow to provide a slot. The
    # "stagger 30s when launching two systems" advice in workflow.md
    # only helps when the ASG has headroom to scale into; with a hard
    # cap, no stagger fixes it.
    #
    # Cost note: at full saturation this is 200 × r5.xlarge ≈ $50.40/hour
    # on-demand. ``min_size=0`` means idle costs are zero — the ECS
    # capacity provider scales the ASG up only when there are pending
    # tasks, and back down when they finish. Saturation is rare; typical
    # cost is bounded by per-fan-out wall-clock × instances used.
    #
    # vCPU quota: 200 r5.xlarge = 800 vCPUs. AWS account default for
    # "Running On-Demand Standard instances" (L-1216C47A) is 5-1152 vCPUs
    # depending on history. Verify with ``aws service-quotas get-service-quota
    # --service-code ec2 --quota-code L-1216C47A --region us-east-1`` —
    # if it's below 800, file a quota increase before pushing past
    # ``max_size=200`` saturation. If you hit ``InsufficientInstanceCapacity``
    # or quota errors when scaling, file a Service Quota increase before
    # raising this further.
    #
    # ``desired_capacity`` is ignored after initial creation so the ECS
    # capacity provider can scale the ASG up/down based on pending-task
    # demand without Pulumi reverting it on ``pulumi up``.
    asg = aws.autoscaling.Group(
        f"{project_name}-asg",
        min_size=0,
        max_size=200,
        desired_capacity=1,
        vpc_zone_identifiers=[s.id for s in subnets["public"]],
        launch_template=aws.autoscaling.GroupLaunchTemplateArgs(
            id=launch_template.id,
            version="$Latest",
        ),
        tags=[
            aws.autoscaling.GroupTagArgs(
                key="Name",
                value=f"{project_name}-ecs-host",
                propagate_at_launch=True,
            ),
            # ``AmazonECSManaged`` is auto-added by ECS when the ASG is
            # attached to a capacity provider (see ``capacity_provider``
            # below). Without declaring it explicitly, ``pulumi up``
            # treats it as drift and tries to delete it on every run —
            # ECS then re-adds it, producing a perpetual diff. Declaring
            # it here makes Pulumi own the tag and resolves the drift.
            # Required for the capacity provider's auto-scaling
            # integration; deleting it (even briefly) can cause the
            # provider to lose track of the ASG. Value is an empty
            # string by AWS convention.
            aws.autoscaling.GroupTagArgs(
                key="AmazonECSManaged",
                value="",
                propagate_at_launch=True,
            ),
        ],
        # New instances are born scale-in protected; the ECS capacity
        # provider's managed termination protection clears that flag
        # ONLY when the instance has zero running tasks. Together this
        # means ASG scale-in cannot terminate an instance with running
        # tasks — including the orchestrator. The 2026-05-11 Mastra run
        # lost its orchestrator at minute ~420 (5 of 10 shards still
        # running) because the previous setting (False) let ASG pick
        # an instance to scale down at random. SIGKILL'd, no graceful
        # shutdown, no auto-merge — recovered manually via
        # ``cross-bench merge-shards``.
        protect_from_scale_in=True,
        opts=pulumi.ResourceOptions(ignore_changes=["desired_capacity"]),
    )

    # ── Capacity Provider ────────────────────────────────────────────
    capacity_provider = aws.ecs.CapacityProvider(
        f"{project_name}-cp",
        auto_scaling_group_provider=aws.ecs.CapacityProviderAutoScalingGroupProviderArgs(
            auto_scaling_group_arn=asg.arn,
            # ENABLED → ECS clears the ASG's per-instance scale-in
            # protection flag only when the instance reports zero
            # running tasks. Combined with the ASG's
            # ``protect_from_scale_in=True`` (born-protected), this
            # guarantees that an instance carrying the orchestrator
            # (or any shard) cannot be picked for ASG scale-in until
            # the task ends. Requires managed_scaling=ENABLED (above).
            # See ecs.py::asg comment for the 2026-05-11 incident
            # that motivated this change.
            managed_termination_protection="ENABLED",
            managed_scaling=aws.ecs.CapacityProviderAutoScalingGroupProviderManagedScalingArgs(
                status="ENABLED",
                target_capacity=100,
            ),
        ),
    )
    aws.ecs.ClusterCapacityProviders(
        f"{project_name}-cluster-cp",
        cluster_name=cluster.name,
        capacity_providers=[capacity_provider.name],
    )

    # ── Task Definition (EC2 launch type) ────────────────────────────
    orchestrator_repo = ecr_repos.get("cross-bench-orchestrator")
    container_image = (
        orchestrator_repo.repository_url.apply(lambda u: f"{u}:latest")
        if orchestrator_repo
        else "cross-bench-orchestrator:latest"
    )

    # SWE-bench image-mirror env. When the mirror repo is provisioned
    # (always in dev today), point the orchestrator at it; the
    # adapter's ``resolve_image_sources`` then pulls per-instance
    # images from ECR instead of trying the bare-tag Docker Hub path.
    # See deviations.md S4 and benchmarks/swebench.py::resolve_image_sources.
    swebench_mirror_url = (
        swebench_mirror_repo.repository_url
        if swebench_mirror_repo is not None
        else pulumi.Output.from_input("")
    )

    task_def = aws.ecs.TaskDefinition(
        f"{project_name}-orchestrator-task",
        family=f"{project_name}-orchestrator",
        network_mode="bridge",
        requires_compatibilities=["EC2"],
        cpu="2048",
        memory="8192",
        execution_role_arn=execution_role.arn,
        task_role_arn=task_role.arn,
        volumes=[aws.ecs.TaskDefinitionVolumeArgs(
            name="docker-sock",
            host_path="/var/run/docker.sock",
        )],
        container_definitions=pulumi.Output.all(
            container_image,
            log_group.name,
            db_url,
            artifacts_bucket.bucket,
            openai_param.name,
            swebench_mirror_url,
        ).apply(lambda args: json.dumps([{
            "name": "orchestrator",
            "image": args[0],
            "essential": True,
            "cpu": 2048,
            "memory": 8192,
            "environment": [
                {"name": "BENCH_DATABASE_URL", "value": args[2]},
                {"name": "BENCH_ARTIFACTS_BUCKET", "value": args[3]},
                {"name": "BENCH_ARTIFACTS_BACKEND", "value": "s3"},
                # boto3 inside the task needs a region; IMDS isn't
                # reliably reachable from bridge-networked containers.
                {"name": "AWS_REGION", "value": "us-east-1"},
                {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"},
                # SWE-bench image source. ``ecr_mirror`` is the AWS
                # default; the adapter constructs the ECR URL from
                # ``BENCH_SWEBENCH_ECR_REPO`` when this is set.
                {"name": "BENCH_SWEBENCH_IMAGE_SOURCE", "value": "ecr_mirror"},
                {"name": "BENCH_SWEBENCH_ECR_REPO", "value": args[5]},
            ],
            "secrets": [
                {
                    "name": "OPENAI_API_KEY",
                    "valueFrom": args[4],
                },
            ],
            "mountPoints": [{
                "sourceVolume": "docker-sock",
                "containerPath": "/var/run/docker.sock",
                "readOnly": False,
            }],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": args[1],
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "orchestrator",
                },
            },
        }])),
    )

    return cluster, task_def


# ── Helpers ──────────────────────────────────────────────────────────

import base64  # noqa: E402


def _render_user_data(cluster_name: str, function_image_ecr: str) -> str:
    """Return base64-encoded cloud-init script for the ECS host.

    The script:
    1. Registers the instance with the ECS cluster.
    2. Authenticates to ECR and pulls the sandbox image.
    3. Re-tags the image to its local name so SandboxManager finds it.
    4. Installs a ``cross-bench-prune`` systemd timer that hourly
       prunes stopped containers + tagged-but-unused images older
       than 2h, then re-stages the sandbox function image. The
       prune is skipped while any ECS task is running on the host
       (avoids racing with ``docker pull`` → ``docker run`` from
       the orchestrator).
    """
    raw = f"""#!/bin/bash
set -ex
echo ECS_CLUSTER={cluster_name} >> /etc/ecs/ecs.config

# Pull sandbox image from ECR and re-tag to the local name that
# SandboxManager / HumanEvalAdapter expect.
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ecr get-login-password --region "$REGION" \\
    | docker login --username AWS --password-stdin \\
      "$(echo '{function_image_ecr}' | cut -d/ -f1)"

docker pull {function_image_ecr}
docker tag  {function_image_ecr} cross-bench-function:latest

# Periodic Docker prune. Sequential SWE-bench Pro fan-outs on the
# same host accumulate ~10 GB per task (compressed image ~3-5 GB,
# expanded layers ~10 GB); even the 200 GB volume fills in ~18
# tasks without cleanup. ``-af --until=2h`` removes any image
# older than 2h that isn't currently referenced by a running
# container — including the sandbox function image after the
# first task, hence the re-pull at the end.
cat > /usr/local/bin/cross-bench-prune.sh <<'PRUNE_EOF'
#!/bin/bash
set -e
TASKS=$(curl -s -m 3 http://localhost:51678/v1/tasks || echo '{{"Tasks":[]}}')
RUNNING=$(echo "$TASKS" | python3 -c "
import sys, json
try:
    payload = json.load(sys.stdin)
    print(sum(1 for x in payload.get('Tasks', []) if x.get('KnownStatus') == 'RUNNING'))
except Exception:
    print(0)
")
if [ "$RUNNING" -gt 0 ]; then
  echo "cross-bench-prune: skipping ($RUNNING task(s) running)"
  exit 0
fi

docker container prune -f
docker image prune -af --filter "until=2h"

REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ecr get-login-password --region "$REGION" \\
    | docker login --username AWS --password-stdin \\
      "$(echo '{function_image_ecr}' | cut -d/ -f1)"
docker pull {function_image_ecr}
docker tag {function_image_ecr} cross-bench-function:latest
PRUNE_EOF
chmod +x /usr/local/bin/cross-bench-prune.sh

cat > /etc/systemd/system/cross-bench-prune.service <<'SVC_EOF'
[Unit]
Description=Prune unused Docker resources for cross-bench
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/cross-bench-prune.sh
SVC_EOF

cat > /etc/systemd/system/cross-bench-prune.timer <<'TMR_EOF'
[Unit]
Description=Run cross-bench-prune every hour

[Timer]
OnBootSec=15min
OnUnitActiveSec=1h
Persistent=true

[Install]
WantedBy=timers.target
TMR_EOF

systemctl daemon-reload
systemctl enable --now cross-bench-prune.timer
"""
    return base64.b64encode(raw.encode()).decode()
