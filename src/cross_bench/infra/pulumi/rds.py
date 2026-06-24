"""RDS Postgres instance for results storage."""

import pulumi
import pulumi_aws as aws


def create_database(project_name, vpc, subnets, security_groups):
    config = pulumi.Config()
    db_password = config.require_secret("db_password")

    subnet_group = aws.rds.SubnetGroup(
        f"{project_name}-db-subnet",
        subnet_ids=[s.id for s in subnets["private"]],
    )

    db_instance = aws.rds.Instance(
        f"{project_name}-db",
        engine="postgres",
        engine_version="16.4",
        instance_class="db.t4g.medium",
        allocated_storage=50,
        db_name="cross_bench",
        username="bench",
        password=db_password,
        db_subnet_group_name=subnet_group.name,
        vpc_security_group_ids=[security_groups["rds"].id],
        skip_final_snapshot=True,
        storage_encrypted=True,
        backup_retention_period=7,
        tags={"Name": f"{project_name}-db"},
    )

    db_url = pulumi.Output.all(db_instance.endpoint, db_password).apply(
        lambda args: f"postgresql+asyncpg://bench:{args[1]}@{args[0]}/cross_bench"
    )

    return db_instance, db_url
