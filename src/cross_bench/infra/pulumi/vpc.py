"""VPC with public and private subnets for ECS and RDS."""

import pulumi_aws as aws


def create_vpc(project_name: str):
    vpc = aws.ec2.Vpc(
        f"{project_name}-vpc",
        cidr_block="10.0.0.0/16",
        enable_dns_hostnames=True,
        enable_dns_support=True,
        tags={"Name": f"{project_name}-vpc"},
    )

    # Two AZs for HA
    azs = ["us-east-1a", "us-east-1b"]

    private_subnets = []
    public_subnets = []
    for i, az in enumerate(azs):
        public_subnets.append(
            aws.ec2.Subnet(
                f"{project_name}-public-{i}",
                vpc_id=vpc.id,
                cidr_block=f"10.0.{i * 2}.0/24",
                availability_zone=az,
                map_public_ip_on_launch=True,
                tags={"Name": f"{project_name}-public-{i}"},
            )
        )
        private_subnets.append(
            aws.ec2.Subnet(
                f"{project_name}-private-{i}",
                vpc_id=vpc.id,
                cidr_block=f"10.0.{i * 2 + 1}.0/24",
                availability_zone=az,
                tags={"Name": f"{project_name}-private-{i}"},
            )
        )

    # Internet gateway for public subnets
    igw = aws.ec2.InternetGateway(
        f"{project_name}-igw",
        vpc_id=vpc.id,
    )

    public_rt = aws.ec2.RouteTable(
        f"{project_name}-public-rt",
        vpc_id=vpc.id,
        routes=[aws.ec2.RouteTableRouteArgs(cidr_block="0.0.0.0/0", gateway_id=igw.id)],
    )

    for i, subnet in enumerate(public_subnets):
        aws.ec2.RouteTableAssociation(
            f"{project_name}-public-rta-{i}",
            subnet_id=subnet.id,
            route_table_id=public_rt.id,
        )

    # Security groups
    ecs_sg = aws.ec2.SecurityGroup(
        f"{project_name}-ecs-sg",
        vpc_id=vpc.id,
        description="ECS tasks",
        egress=[aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"]
        )],
    )

    rds_sg = aws.ec2.SecurityGroup(
        f"{project_name}-rds-sg",
        vpc_id=vpc.id,
        description="RDS Postgres",
        ingress=[aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=5432, to_port=5432,
            security_groups=[ecs_sg.id],
        )],
    )

    subnets = {"public": public_subnets, "private": private_subnets}
    security_groups = {"ecs": ecs_sg, "rds": rds_sg}

    return vpc, subnets, security_groups
