"""S3 buckets for artifacts and dataset caching."""

import pulumi_aws as aws


def create_s3_buckets(project_name: str):
    artifacts_bucket = aws.s3.BucketV2(
        f"{project_name}-artifacts",
        tags={"Purpose": "evaluation-artifacts"},
    )

    # Lifecycle: move to Infrequent Access after 90 days
    aws.s3.BucketLifecycleConfigurationV2(
        f"{project_name}-artifacts-lifecycle",
        bucket=artifacts_bucket.id,
        rules=[aws.s3.BucketLifecycleConfigurationV2RuleArgs(
            id="archive-old-artifacts",
            status="Enabled",
            transitions=[aws.s3.BucketLifecycleConfigurationV2RuleTransitionArgs(
                days=90,
                storage_class="STANDARD_IA",
            )],
        )],
    )

    # Versioning for immutability
    aws.s3.BucketVersioningV2(
        f"{project_name}-artifacts-versioning",
        bucket=artifacts_bucket.id,
        versioning_configuration=aws.s3.BucketVersioningV2VersioningConfigurationArgs(
            status="Enabled",
        ),
    )

    datasets_bucket = aws.s3.BucketV2(
        f"{project_name}-datasets",
        tags={"Purpose": "benchmark-datasets"},
    )

    return artifacts_bucket, datasets_bucket
