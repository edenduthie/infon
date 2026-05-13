#!/usr/bin/env python3
"""
Deploy and run the SIRIUS pipeline as an AWS Step Function.

Provisions all infrastructure:
  - IAM roles (Lambda, Step Functions, Batch, GPU EC2)
  - ECR repositories
  - AWS Batch compute environments (CPU on-demand)
  - AWS Batch job definitions
  - EC2 GPU instances via Lambda (replaces Batch GPU)
  - Lambda functions (5 zip-packaged + 2 container-image)
  - Step Functions state machines (Pipeline 1: Extraction, Pipeline 2: Analysis)

Usage:
    python deploy.py build                   # Build all Docker images via CodeBuild
    python deploy.py build --image fast-pipeline lambda-heavy  # Build specific images
    python deploy.py deploy                  # Deploy all infrastructure
    python deploy.py run                     # Start pipeline (60 GPU instances)
    python deploy.py run --gpu-instances 1 --limit 100  # Test run
    python deploy.py status                  # Check execution status
    python deploy.py teardown                # Remove all infrastructure
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "038462780313")

# Naming
PROJECT = "sirius"
PIPELINE_NAME = f"{PROJECT}-pipeline"

# S3
S3_BUCKET = "sirius-dimefiled-results"
LANCEDB_URI = f"s3://{S3_BUCKET}/hypergraph/"

# ECR repos
ECR_LAMBDA_HEAVY = f"{PROJECT}-pipeline-lambda-heavy"
ECR_STRATEGIC = f"{PROJECT}-strategic-analysis"
ECR_MINE_PATTERNS = f"{PROJECT}-mine-patterns"
ECR_FAST_PIPELINE = f"{PROJECT}-fast-pipeline"

# Docker image builds (CodeBuild)
CODEBUILD_SERVICE_ROLE = f"arn:aws:iam::{ACCOUNT_ID}:role/codebuild-step3-role"
S3_CODEBUILD_PREFIX = "codebuild"

# Each image: ECR repo, CodeBuild project, build context (relative to
# step_function/), optional Dockerfile path and selective includes.
DOCKER_IMAGES = {
    "lambda-heavy": {
        "ecr_repo": ECR_LAMBDA_HEAVY,
        "codebuild_project": "sirius-lambda-heavy-build",
        "build_context": ".",
        "dockerfile": "batch/lambda_heavy/Dockerfile",
        "includes": [
            "batch/lambda_heavy/",
            "lambdas/build_sqlite.py",
            "lambdas/export_workbook.py",
            "lambdas/mine_patterns.py",
        ],
    },
    "strategic-analysis": {
        "ecr_repo": ECR_STRATEGIC,
        "codebuild_project": "sirius-strategic-analysis-build",
        "build_context": ".",
        "dockerfile": "batch/strategic_analysis/Dockerfile",
        "includes": [
            "batch/strategic_analysis/",
            "batch/ingest_normalize/shared/",
        ],
    },
    "mine-patterns": {
        "ecr_repo": ECR_MINE_PATTERNS,
        "codebuild_project": "sirius-mine-patterns-build",
        "build_context": ".",
        "dockerfile": "batch/mine_patterns/Dockerfile",
        "includes": [
            "batch/mine_patterns/",
            "lambdas/mine_patterns.py",
        ],
    },
    "fast-pipeline": {
        "ecr_repo": ECR_FAST_PIPELINE,
        "codebuild_project": "sirius-fast-pipeline-build",
        "build_context": "../..",
        "dockerfile": "sbl-pipeline/step_function/batch/fast_pipeline/Dockerfile",
        "includes": [
            "local_pipeline.py",
            "location_gazetteer.py",
            "sbl-pipeline/step_function/batch/fast_pipeline/",
        ],
    },
}

# Batch (CPU only — GPU extraction uses direct EC2 instances)
BATCH_CPU_CE = f"{PROJECT}-cpu-compute"
BATCH_CPU_JQ = f"{PROJECT}-cpu-queue"
BATCH_INGEST_JOB_DEF = f"{PROJECT}-fast-pipeline"
BATCH_STRATEGIC_JOB_DEF = f"{PROJECT}-strategic-analysis"
BATCH_MINE_PATTERNS_JOB_DEF = f"{PROJECT}-mine-patterns"

# GPU EC2 direct launch (replaces Batch GPU)
GPU_EC2_ROLE_NAME = f"{PROJECT}-gpu-ec2-role"
GPU_EC2_INSTANCE_PROFILE = f"{PROJECT}-gpu-nuextract-profile"
GPU_SCRIPTS_PREFIX = "gpu-scripts/"

# Lambda
LAMBDA_PREFIX = f"{PROJECT}-pipeline"

# Lightweight Lambdas deployed as zip packages
LAMBDA_ZIP_FUNCTIONS = {
    "query_cc_index": {"memory": 1024, "timeout": 900, "handler": "query_cc_index"},
    "split_csv": {"memory": 2048, "timeout": 900, "handler": "split_csv"},
    "dispatch_sqs": {"memory": 2048, "timeout": 900, "handler": "dispatch_sqs"},
    "check_extraction": {"memory": 512, "timeout": 120, "handler": "check_extraction"},
    "launch_gpu_instances": {"memory": 512, "timeout": 300, "handler": "launch_gpu_instances"},
}

# Heavy Lambdas deployed as container images (need lancedb/pandas/model2vec)
LAMBDA_IMAGE_FUNCTIONS = {
    "export_workbook": {"memory": 2048, "timeout": 600, "handler": "export_workbook"},
    "build_sqlite": {"memory": 2048, "timeout": 900, "handler": "build_sqlite"},
}

# Step Functions
STATE_MACHINE_NAME = f"{PROJECT}-pipeline"
ANALYSIS_STATE_MACHINE_NAME = f"{PROJECT}-pipeline-analysis"

# Batch instance config (CPU only)
CPU_INSTANCE_TYPES = ["r6i.4xlarge", "r6i.8xlarge", "r5.4xlarge", "r5.8xlarge"]
CPU_MAX_VCPUS = 64

# IAM role names
ROLE_LAMBDA = f"{PROJECT}-pipeline-lambda-role"
ROLE_SFN = f"{PROJECT}-pipeline-sfn-role"
ROLE_BATCH_SERVICE = f"{PROJECT}-batch-service-role"
ROLE_BATCH_INSTANCE = f"{PROJECT}-batch-instance-role"
ROLE_BATCH_JOB = f"{PROJECT}-batch-job-role"
INSTANCE_PROFILE_BATCH = f"{PROJECT}-batch-instance-profile"

# Bedrock model IDs (passed to Lambdas as env vars)
SONNET_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
HAIKU_MODEL_ID = "us.anthropic.claude-haiku-4-5-20250514-v1:0"

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

iam = boto3.client("iam", region_name=REGION)
ecr = boto3.client("ecr", region_name=REGION)
batch = boto3.client("batch", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)
sfn = boto3.client("stepfunctions", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
codebuild = boto3.client("codebuild", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)


# ---------------------------------------------------------------------------
# IAM Role Helpers
# ---------------------------------------------------------------------------

def _create_role(role_name: str, service: str, managed_policies: list,
                 inline_policies: dict = None) -> str:
    """Create IAM role if it doesn't exist. Returns ARN."""
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": service},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        role = iam.get_role(RoleName=role_name)
        print(f"  Role exists: {role_name}")
        role_arn = role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        print(f"  Creating role: {role_name}")
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description=f"SIRIUS pipeline: {role_name}",
        )
        for arn in managed_policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=arn)
        time.sleep(10)  # IAM propagation
        role = iam.get_role(RoleName=role_name)
        role_arn = role["Role"]["Arn"]

    # Always ensure inline policies are up-to-date
    if inline_policies:
        for name, doc in inline_policies.items():
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName=name,
                PolicyDocument=json.dumps(doc),
            )
    return role_arn


def ensure_iam_roles() -> dict:
    """Create all IAM roles. Returns dict of name→ARN."""
    print("\n--- IAM Roles ---")
    arns = {}

    # Lambda execution role
    arns["lambda"] = _create_role(
        ROLE_LAMBDA, "lambda.amazonaws.com",
        [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
            "arn:aws:iam::aws:policy/AmazonBedrockFullAccess",
            "arn:aws:iam::aws:policy/AmazonAthenaFullAccess",
        ],
        inline_policies={
            "EC2LaunchGPU": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "ec2:RunInstances",
                            "ec2:DescribeImages",
                            "ec2:CreateTags",
                        ],
                        "Resource": "*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": f"arn:aws:iam::{ACCOUNT_ID}:role/{GPU_EC2_ROLE_NAME}",
                    },
                ],
            },
        },
    )

    # Step Functions role
    arns["sfn"] = _create_role(
        ROLE_SFN, "states.amazonaws.com",
        [],
        inline_policies={
            "InvokeLambdaAndBatch": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["lambda:InvokeFunction"],
                        "Resource": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "batch:SubmitJob",
                            "batch:DescribeJobs",
                            "batch:TerminateJob",
                        ],
                        "Resource": "*",
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "events:PutTargets",
                            "events:PutRule",
                            "events:DescribeRule",
                        ],
                        "Resource": f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/StepFunctionsGetEventsForBatchJobsRule",
                    },
                ],
            },
        },
    )

    # Batch service role (AWSBatchServiceRole)
    try:
        role = iam.get_role(RoleName="AWSBatchServiceRole")
        arns["batch_service"] = role["Role"]["Arn"]
        print(f"  Role exists: AWSBatchServiceRole")
    except iam.exceptions.NoSuchEntityException:
        arns["batch_service"] = _create_role(
            "AWSBatchServiceRole", "batch.amazonaws.com",
            ["arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"],
        )

    # Batch instance role (for EC2 instances in compute env)
    arns["batch_instance"] = _create_role(
        ROLE_BATCH_INSTANCE, "ec2.amazonaws.com",
        [
            "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
        ],
    )

    # Instance profile for Batch EC2
    try:
        iam.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_BATCH)
        print(f"  Instance profile exists: {INSTANCE_PROFILE_BATCH}")
    except iam.exceptions.NoSuchEntityException:
        print(f"  Creating instance profile: {INSTANCE_PROFILE_BATCH}")
        iam.create_instance_profile(InstanceProfileName=INSTANCE_PROFILE_BATCH)
        iam.add_role_to_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_BATCH,
            RoleName=ROLE_BATCH_INSTANCE,
        )
        time.sleep(10)

    # Batch job role (for containers)
    arns["batch_job"] = _create_role(
        ROLE_BATCH_JOB, "ecs-tasks.amazonaws.com",
        [
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
            "arn:aws:iam::aws:policy/AmazonBedrockFullAccess",
        ],
    )

    return arns


def ensure_gpu_ec2_role():
    """Create IAM role + instance profile for GPU EC2 instances.

    Grants S3/SQS access and EC2 self-terminate permission so instances
    can shut themselves down when the work queue is drained.
    """
    print("\n--- GPU EC2 IAM Role ---")

    # Create role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        role = iam.get_role(RoleName=GPU_EC2_ROLE_NAME)
        print(f"  Role exists: {GPU_EC2_ROLE_NAME}")
        role_arn = role["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        print(f"  Creating role: {GPU_EC2_ROLE_NAME}")
        resp = iam.create_role(
            RoleName=GPU_EC2_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SIRIUS GPU EC2 instances for NuExtract extraction",
        )
        role_arn = resp["Role"]["Arn"]

    # Attach managed policies
    for policy_arn in [
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
    ]:
        try:
            iam.attach_role_policy(RoleName=GPU_EC2_ROLE_NAME, PolicyArn=policy_arn)
        except Exception:
            pass

    # Inline policy: EC2 self-terminate (scoped to tagged instances)
    terminate_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "ec2:TerminateInstances",
            "Resource": f"arn:aws:ec2:{REGION}:{ACCOUNT_ID}:instance/*",
            "Condition": {
                "StringEquals": {
                    "ec2:ResourceTag/Name": "NuExtract-GPU-Extractor",
                },
            },
        }],
    }
    iam.put_role_policy(
        RoleName=GPU_EC2_ROLE_NAME,
        PolicyName="EC2-SelfTerminate",
        PolicyDocument=json.dumps(terminate_policy),
    )

    # Create instance profile
    try:
        iam.get_instance_profile(InstanceProfileName=GPU_EC2_INSTANCE_PROFILE)
        print(f"  Instance profile exists: {GPU_EC2_INSTANCE_PROFILE}")
    except iam.exceptions.NoSuchEntityException:
        print(f"  Creating instance profile: {GPU_EC2_INSTANCE_PROFILE}")
        iam.create_instance_profile(InstanceProfileName=GPU_EC2_INSTANCE_PROFILE)
        iam.add_role_to_instance_profile(
            InstanceProfileName=GPU_EC2_INSTANCE_PROFILE,
            RoleName=GPU_EC2_ROLE_NAME,
        )
        time.sleep(10)  # IAM propagation

    return role_arn


def upload_gpu_scripts():
    """Upload GPU extraction scripts from batch/gpu_extract/ to S3."""
    print("\n--- GPU Extraction Scripts ---")
    script_dir = Path(__file__).parent / "batch" / "gpu_extract"

    scripts = ["gpu_extract.py", "nuextract_template.py", "warc_fetch.py", "postprocess.py"]
    for name in scripts:
        src = script_dir / name
        if not src.exists():
            print(f"  [ERROR] Missing: {src}")
            raise FileNotFoundError(f"Required script not found: {src}")
        s3.upload_file(str(src), S3_BUCKET, f"{GPU_SCRIPTS_PREFIX}{name}")
        print(f"  {name} -> s3://{S3_BUCKET}/{GPU_SCRIPTS_PREFIX}{name}")


# ---------------------------------------------------------------------------
# ECR
# ---------------------------------------------------------------------------

def ensure_ecr_repos():
    """Create ECR repositories."""
    print("\n--- ECR Repositories ---")
    for repo_name in [ECR_FAST_PIPELINE, ECR_LAMBDA_HEAVY,
                       ECR_STRATEGIC, ECR_MINE_PATTERNS]:
        try:
            ecr.describe_repositories(repositoryNames=[repo_name])
            print(f"  Repo exists: {repo_name}")
        except ecr.exceptions.RepositoryNotFoundException:
            print(f"  Creating: {repo_name}")
            ecr.create_repository(repositoryName=repo_name)


# ---------------------------------------------------------------------------
# Docker Image Builds (CodeBuild)
# ---------------------------------------------------------------------------

def _zip_build_context(image_cfg: dict) -> bytes:
    """Create a zip of the Docker build context for a given image config."""
    import io

    base = Path(__file__).parent
    # Resolve context root from build_context (defaults to base itself).
    # For most images build_context is "." so context_root == base.
    # Images needing files outside step_function/ can use ".." to reach the
    # repo root (e.g. sbl-pipeline/) and adjust includes/dockerfile paths.
    context_root = (base / image_cfg.get("build_context", ".")).resolve()
    includes = image_cfg.get("includes")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if includes:
            # Selective includes relative to context_root
            for pattern in includes:
                if pattern.endswith("/"):
                    dir_path = context_root / pattern
                    for root, dirs, files in os.walk(dir_path):
                        dirs[:] = [d for d in dirs if d != "__pycache__"]
                        for f in files:
                            if f.endswith(".pyc") or f == ".DS_Store":
                                continue
                            full = Path(root) / f
                            zf.write(full, str(full.relative_to(context_root)))
                else:
                    full = context_root / pattern
                    if full.exists():
                        zf.write(full, pattern)
        else:
            # Zip entire build_context directory
            for root, dirs, files in os.walk(context_root):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    if f.endswith(".pyc") or f == ".DS_Store":
                        continue
                    full = Path(root) / f
                    zf.write(full, str(full.relative_to(context_root)))
    return buf.getvalue()


def _make_buildspec(image_name: str, ecr_uri: str,
                    dockerfile: str = None) -> str:
    """Generate a CodeBuild buildspec for building and pushing a Docker image."""
    if dockerfile:
        build_cmd = f"docker build -f {dockerfile} -t {image_name} ."
    else:
        build_cmd = f"docker build -t {image_name} ."

    return (
        "version: 0.2\n"
        "phases:\n"
        "  pre_build:\n"
        "    commands:\n"
        f"      - aws ecr get-login-password --region {REGION} | "
        f"docker login --username AWS --password-stdin "
        f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com\n"
        "  build:\n"
        "    commands:\n"
        f"      - {build_cmd}\n"
        f"      - docker tag {image_name}:latest {ecr_uri}:latest\n"
        f"      - docker push {ecr_uri}:latest\n"
    )


def _ensure_codebuild_projects():
    """Ensure CodeBuild projects exist for all Docker images."""
    print("\n--- CodeBuild Projects ---")
    for name, cfg in DOCKER_IMAGES.items():
        project_name = cfg["codebuild_project"]
        resp = codebuild.batch_get_projects(names=[project_name])
        if resp["projects"]:
            print(f"  Project exists: {project_name}")
            continue

        print(f"  Creating: {project_name}")
        codebuild.create_project(
            name=project_name,
            source={
                "type": "NO_SOURCE",
                "buildspec": ("version: 0.2\nphases:\n  build:\n"
                              "    commands:\n      - echo placeholder"),
            },
            artifacts={"type": "NO_ARTIFACTS"},
            environment={
                "type": "LINUX_CONTAINER",
                "image": "aws/codebuild/standard:7.0",
                "computeType": "BUILD_GENERAL1_LARGE",
                "privilegedMode": True,
                "imagePullCredentialsType": "CODEBUILD",
            },
            serviceRole=CODEBUILD_SERVICE_ROLE,
        )


def build(images: list = None):
    """Build and push Docker images to ECR via CodeBuild."""
    print("=" * 70)
    print("SIRIUS Pipeline — Docker Image Builds")
    print("=" * 70)

    ensure_ecr_repos()
    _ensure_codebuild_projects()

    targets = {k: v for k, v in DOCKER_IMAGES.items()
               if not images or k in images}
    if not targets:
        print(f"\nNo matching images. Available: {', '.join(DOCKER_IMAGES)}")
        return

    # Package source, upload to S3, and start builds
    build_ids = {}
    for name, cfg in targets.items():
        ecr_uri = (f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/"
                   f"{cfg['ecr_repo']}")
        dockerfile = cfg.get("dockerfile")

        print(f"\n  [{name}] Packaging source...")
        zip_bytes = _zip_build_context(cfg)
        s3_key = f"{S3_CODEBUILD_PREFIX}/{cfg['ecr_repo']}-src.zip"
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=zip_bytes)

        buildspec = _make_buildspec(cfg["ecr_repo"], ecr_uri, dockerfile)
        print(f"  [{name}] Starting CodeBuild: {cfg['codebuild_project']}")
        resp = codebuild.start_build(
            projectName=cfg["codebuild_project"],
            sourceTypeOverride="S3",
            sourceLocationOverride=f"{S3_BUCKET}/{s3_key}",
            buildspecOverride=buildspec,
        )
        build_ids[name] = resp["build"]["id"]
        print(f"  [{name}] Build ID: {build_ids[name]}")

    # Poll all builds until complete
    print(f"\nWaiting for {len(build_ids)} build(s)...")
    completed = {}
    remaining = dict(build_ids)

    while remaining:
        time.sleep(30)
        resp = codebuild.batch_get_builds(ids=list(remaining.values()))
        for b in resp["builds"]:
            bid = b["id"]
            name = next(n for n, i in remaining.items() if i == bid)
            build_status = b["buildStatus"]
            phase = b["currentPhase"]

            if build_status != "IN_PROGRESS":
                completed[name] = build_status
                del remaining[name]
                tag = "OK" if build_status == "SUCCEEDED" else "FAIL"
                print(f"  [{name}] {tag} ({build_status})")
            else:
                print(f"  [{name}] building... ({phase})")

    # Summary
    print(f"\n{'=' * 70}")
    print("Build Summary")
    print(f"{'=' * 70}")
    all_ok = True
    for name, build_status in completed.items():
        tag = "OK" if build_status == "SUCCEEDED" else "FAIL"
        print(f"  [{tag}] {name}: {build_status}")
        if build_status != "SUCCEEDED":
            all_ok = False

    if all_ok:
        print("\nAll images built and pushed successfully.")
        print("Next: python deploy.py deploy")
    else:
        print("\nSome builds failed. Check CodeBuild logs:")
        for name, build_status in completed.items():
            if build_status != "SUCCEEDED":
                bid = build_ids[name]
                project = bid.split(":")[0]
                print(f"  aws logs tail /aws/codebuild/{project} --since 30m")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Networking — find default VPC subnets and security group
# ---------------------------------------------------------------------------

def get_default_vpc_config() -> dict:
    """Get default VPC subnets and security group for Batch."""
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise RuntimeError("No default VPC found. Create one or specify VPC config.")

    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]]

    sgs = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": ["default"]},
        ]
    )
    sg_ids = [sg["GroupId"] for sg in sgs["SecurityGroups"]]

    return {"subnets": subnet_ids, "securityGroups": sg_ids}


# ---------------------------------------------------------------------------
# AWS Batch
# ---------------------------------------------------------------------------

def ensure_batch_compute_environments(role_arns: dict, vpc_config: dict):
    """Create CPU Batch compute environment."""
    print("\n--- Batch Compute Environments ---")

    for ce_name, instance_types, max_vcpus, launch_template in [
        (BATCH_CPU_CE, CPU_INSTANCE_TYPES, CPU_MAX_VCPUS, None),
    ]:
        exists = False
        try:
            resp = batch.describe_compute_environments(computeEnvironments=[ce_name])
            if resp["computeEnvironments"]:
                exists = True
        except Exception:
            pass

        if exists:
            print(f"  CE exists: {ce_name}")
            continue

        print(f"  Creating: {ce_name}")
        compute_resources = {
            "type": "EC2",
            "allocationStrategy": "BEST_FIT_PROGRESSIVE",
            "minvCpus": 0,
            "maxvCpus": max_vcpus,
            "instanceTypes": instance_types,
            "subnets": vpc_config["subnets"],
            "securityGroupIds": vpc_config["securityGroups"],
            "instanceRole": f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/{INSTANCE_PROFILE_BATCH}",
        }
        if launch_template:
            compute_resources["launchTemplate"] = launch_template

        batch.create_compute_environment(
            computeEnvironmentName=ce_name,
            type="MANAGED",
            state="ENABLED",
            computeResources=compute_resources,
        )

    # Wait for CEs to become VALID
    for ce_name in [BATCH_CPU_CE]:
        print(f"  Waiting for {ce_name} to become VALID...")
        for _ in range(30):
            resp = batch.describe_compute_environments(computeEnvironments=[ce_name])
            if resp["computeEnvironments"]:
                status = resp["computeEnvironments"][0]["status"]
                if status == "VALID":
                    break
                elif status == "INVALID":
                    reason = resp["computeEnvironments"][0].get("statusReason", "unknown")
                    print(f"    WARNING: {ce_name} is INVALID: {reason}")
                    break
            time.sleep(10)


def ensure_batch_job_queues():
    """Create Batch job queues."""
    print("\n--- Batch Job Queues ---")

    for jq_name, ce_name in [
        (BATCH_CPU_JQ, BATCH_CPU_CE),
    ]:
        try:
            resp = batch.describe_job_queues(jobQueues=[jq_name])
            if resp["jobQueues"]:
                print(f"  JQ exists: {jq_name}")
                continue
        except Exception:
            pass

        print(f"  Creating: {jq_name}")
        batch.create_job_queue(
            jobQueueName=jq_name,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[{
                "order": 1,
                "computeEnvironment": ce_name,
            }],
        )


def ensure_batch_job_definitions(role_arns: dict):
    """Create Batch job definitions (CPU only — GPU uses EC2 direct)."""
    print("\n--- Batch Job Definitions ---")

    fast_ecr_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_FAST_PIPELINE}:latest"

    # Fast DuckDB pipeline (replaces legacy 60GB LanceDB ingest)
    print(f"  Registering: {BATCH_INGEST_JOB_DEF}")
    batch.register_job_definition(
        jobDefinitionName=BATCH_INGEST_JOB_DEF,
        type="container",
        containerProperties={
            "image": fast_ecr_uri,
            "vcpus": 16,
            "memory": 32000,  # 32 GB — 98K events needs headroom for Stage 2 relations
            "jobRoleArn": role_arns["batch_job"],
            "environment": [
                {"name": "NUEXTRACT_S3_BUCKET", "value": S3_BUCKET},
                {"name": "NUEXTRACT_S3_PREFIX", "value": "nuextract-v2-events/"},
                {"name": "OUTPUT_S3_PREFIX", "value": "pipeline-output/"},
                {"name": "AWS_REGION", "value": REGION},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": f"/aws/batch/{BATCH_INGEST_JOB_DEF}",
                    "awslogs-region": REGION,
                    "awslogs-stream-prefix": "fast-pipeline",
                    "awslogs-create-group": "true",
                },
            },
        },
        retryStrategy={"attempts": 1},
        timeout={"attemptDurationSeconds": 7200},  # 2 hours — 98K events takes ~90 min
    )

    # Strategic analysis job
    strategic_ecr_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_STRATEGIC}:latest"
    print(f"  Registering: {BATCH_STRATEGIC_JOB_DEF}")
    batch.register_job_definition(
        jobDefinitionName=BATCH_STRATEGIC_JOB_DEF,
        type="container",
        containerProperties={
            "image": strategic_ecr_uri,
            "vcpus": 8,
            "memory": 30000,  # 30 GB — loading 193K events + 168K clusters
            "jobRoleArn": role_arns["batch_job"],
            "environment": [
                {"name": "LANCEDB_URI", "value": LANCEDB_URI},
                {"name": "BEDROCK_MODEL_ID", "value": SONNET_MODEL_ID},
                {"name": "S3_BUCKET", "value": S3_BUCKET},
                {"name": "S3_RESULTS_PREFIX", "value": "strategic-analysis/"},
                {"name": "AWS_REGION", "value": REGION},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": f"/aws/batch/{BATCH_STRATEGIC_JOB_DEF}",
                    "awslogs-region": REGION,
                    "awslogs-stream-prefix": "strategic",
                    "awslogs-create-group": "true",
                },
            },
        },
        retryStrategy={"attempts": 1},
        timeout={"attemptDurationSeconds": 7200},  # 2 hours
    )

    # Mine patterns job
    mine_ecr_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_MINE_PATTERNS}:latest"
    print(f"  Registering: {BATCH_MINE_PATTERNS_JOB_DEF}")
    batch.register_job_definition(
        jobDefinitionName=BATCH_MINE_PATTERNS_JOB_DEF,
        type="container",
        containerProperties={
            "image": mine_ecr_uri,
            "vcpus": 4,
            "memory": 8000,  # 8 GB
            "jobRoleArn": role_arns["batch_job"],
            "environment": [
                {"name": "S3_BUCKET", "value": S3_BUCKET},
                {"name": "S3_KEY_PREFIX", "value": "sqlite/"},
                {"name": "AWS_REGION", "value": REGION},
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": f"/aws/batch/{BATCH_MINE_PATTERNS_JOB_DEF}",
                    "awslogs-region": REGION,
                    "awslogs-stream-prefix": "mine",
                    "awslogs-create-group": "true",
                },
            },
        },
        retryStrategy={"attempts": 1},
        timeout={"attemptDurationSeconds": 3600},  # 1 hour
    )


# ---------------------------------------------------------------------------
# Lambda Functions
# ---------------------------------------------------------------------------

def _zip_lambda(handler_name: str) -> bytes:
    """Create a zip package for a Lambda handler."""
    import io

    handler_path = Path(__file__).parent / "lambdas" / f"{handler_name}.py"
    if not handler_path.exists():
        raise FileNotFoundError(f"Lambda handler not found: {handler_path}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, f"{handler_name}.py")
    return buf.getvalue()


def ensure_lambda_functions(role_arn: str):
    """Create or update all Lambda functions."""
    print("\n--- Lambda Functions (zip) ---")

    common_env = {
        "S3_BUCKET": S3_BUCKET,
        "LANCEDB_URI": LANCEDB_URI,
        "AWS_REGION_OVERRIDE": REGION,
        "BEDROCK_MODEL_ID": SONNET_MODEL_ID,
        "HAIKU_MODEL_ID": HAIKU_MODEL_ID,
        "GPU_INSTANCE_PROFILE": GPU_EC2_INSTANCE_PROFILE,
        "GPU_SCRIPTS_PREFIX": GPU_SCRIPTS_PREFIX,
        "AWS_ACCOUNT_ID": ACCOUNT_ID,
    }

    # --- Zip-packaged Lambdas (lightweight) ---
    for name, cfg in LAMBDA_ZIP_FUNCTIONS.items():
        func_name = f"{LAMBDA_PREFIX}-{name}"
        handler_module = cfg["handler"]

        try:
            lambda_client.get_function(FunctionName=func_name)
            print(f"  Updating: {func_name}")
            zip_bytes = _zip_lambda(handler_module)
            lambda_client.update_function_code(
                FunctionName=func_name,
                ZipFile=zip_bytes,
            )
            waiter = lambda_client.get_waiter("function_updated_v2")
            waiter.wait(FunctionName=func_name)
            lambda_client.update_function_configuration(
                FunctionName=func_name,
                MemorySize=cfg["memory"],
                Timeout=cfg["timeout"],
                Environment={"Variables": common_env},
            )
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"  Creating: {func_name}")
            zip_bytes = _zip_lambda(handler_module)
            lambda_client.create_function(
                FunctionName=func_name,
                Role=role_arn,
                Code={"ZipFile": zip_bytes},
                Handler=f"{handler_module}.handler",
                Runtime="python3.11",
                MemorySize=cfg["memory"],
                Timeout=cfg["timeout"],
                Environment={"Variables": common_env},
            )
            waiter = lambda_client.get_waiter("function_active_v2")
            waiter.wait(FunctionName=func_name)

    # --- Container-image Lambdas (heavy: lancedb/model2vec/pandas) ---
    print("\n--- Lambda Functions (container image) ---")
    image_uri = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_LAMBDA_HEAVY}:latest"

    for name, cfg in LAMBDA_IMAGE_FUNCTIONS.items():
        func_name = f"{LAMBDA_PREFIX}-{name}"
        handler_module = cfg["handler"]
        cmd_override = [f"{handler_module}.handler"]

        try:
            fn_info = lambda_client.get_function(FunctionName=func_name)
            existing_pkg = fn_info["Configuration"].get("PackageType", "Zip")

            if existing_pkg == "Zip":
                # Must delete zip-based Lambda and recreate as Image
                print(f"  Replacing zip→image: {func_name}")
                lambda_client.delete_function(FunctionName=func_name)
                time.sleep(2)
                raise lambda_client.exceptions.ResourceNotFoundException(
                    {"Error": {"Code": "ResourceNotFoundException"}},
                    "GetFunction",
                )

            # Already an image Lambda — update in place
            print(f"  Updating: {func_name}")
            lambda_client.update_function_code(
                FunctionName=func_name,
                ImageUri=image_uri,
            )
            waiter = lambda_client.get_waiter("function_updated_v2")
            waiter.wait(FunctionName=func_name)
            lambda_client.update_function_configuration(
                FunctionName=func_name,
                MemorySize=cfg["memory"],
                Timeout=cfg["timeout"],
                Environment={"Variables": common_env},
                ImageConfig={"Command": cmd_override},
            )
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"  Creating (image): {func_name}")
            lambda_client.create_function(
                FunctionName=func_name,
                Role=role_arn,
                PackageType="Image",
                Code={"ImageUri": image_uri},
                MemorySize=cfg["memory"],
                Timeout=cfg["timeout"],
                Environment={"Variables": common_env},
                ImageConfig={"Command": cmd_override},
            )
            waiter = lambda_client.get_waiter("function_active_v2")
            waiter.wait(FunctionName=func_name)


# ---------------------------------------------------------------------------
# Step Functions State Machine
# ---------------------------------------------------------------------------

def ensure_state_machine(sfn_role_arn: str) -> str:
    """Create or update the Step Functions state machine."""
    print("\n--- Step Functions State Machine ---")

    from definition import build_definition

    config = {
        # Lambda ARNs (extraction-phase only)
        "lambda_query_cc_index": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-query_cc_index",
        "lambda_split_csv": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-split_csv",
        "lambda_dispatch_sqs": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-dispatch_sqs",
        "lambda_check_extraction": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-check_extraction",
        "lambda_launch_gpu_instances": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-launch_gpu_instances",
        # S3
        "s3_bucket": S3_BUCKET,
    }

    definition = build_definition(config)
    definition_json = json.dumps(definition)

    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}"

    try:
        sfn.describe_state_machine(stateMachineArn=sm_arn)
        print(f"  Updating: {STATE_MACHINE_NAME}")
        sfn.update_state_machine(
            stateMachineArn=sm_arn,
            definition=definition_json,
            roleArn=sfn_role_arn,
        )
    except sfn.exceptions.StateMachineDoesNotExist:
        print(f"  Creating: {STATE_MACHINE_NAME}")
        resp = sfn.create_state_machine(
            name=STATE_MACHINE_NAME,
            definition=definition_json,
            roleArn=sfn_role_arn,
            type="STANDARD",
        )
        sm_arn = resp["stateMachineArn"]

    print(f"  ARN: {sm_arn}")
    return sm_arn


def ensure_analysis_state_machine(sfn_role_arn: str) -> str:
    """Create or update the analysis-only Step Functions state machine (Pipeline 2)."""
    print("\n--- Analysis State Machine (Pipeline 2) ---")

    from definition import build_analysis_definition

    config = {
        # Lambda ARNs (analysis-phase Lambdas)
        "lambda_export_workbook": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-export_workbook",
        "lambda_build_sqlite": f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_PREFIX}-build_sqlite",
        # Batch (CPU — ingest + analysis)
        "batch_ingest_job_def": BATCH_INGEST_JOB_DEF,
        "batch_strategic_analysis_job_def": BATCH_STRATEGIC_JOB_DEF,
        "batch_cpu_job_queue": BATCH_CPU_JQ,
        "batch_mine_patterns_job_def": BATCH_MINE_PATTERNS_JOB_DEF,
        "batch_cpu_queue": BATCH_CPU_JQ,
        # S3
        "s3_bucket": S3_BUCKET,
        "lancedb_uri": LANCEDB_URI,
        # Model
        "region": REGION,
        "bedrock_model_id": SONNET_MODEL_ID,
    }

    definition = build_analysis_definition(config)
    definition_json = json.dumps(definition)

    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{ANALYSIS_STATE_MACHINE_NAME}"

    try:
        sfn.describe_state_machine(stateMachineArn=sm_arn)
        print(f"  Updating: {ANALYSIS_STATE_MACHINE_NAME}")
        sfn.update_state_machine(
            stateMachineArn=sm_arn,
            definition=definition_json,
            roleArn=sfn_role_arn,
        )
    except sfn.exceptions.StateMachineDoesNotExist:
        print(f"  Creating: {ANALYSIS_STATE_MACHINE_NAME}")
        resp = sfn.create_state_machine(
            name=ANALYSIS_STATE_MACHINE_NAME,
            definition=definition_json,
            roleArn=sfn_role_arn,
            type="STANDARD",
        )
        sm_arn = resp["stateMachineArn"]

    print(f"  ARN: {sm_arn}")
    return sm_arn


# ---------------------------------------------------------------------------
# Deploy Command
# ---------------------------------------------------------------------------

def deploy():
    """Deploy all infrastructure."""
    print("=" * 70)
    print("SIRIUS Pipeline — AWS Step Function Deployment")
    print("=" * 70)

    role_arns = ensure_iam_roles()
    ensure_gpu_ec2_role()
    upload_gpu_scripts()
    ensure_ecr_repos()

    vpc_config = get_default_vpc_config()
    print(f"\n  VPC subnets: {vpc_config['subnets'][:3]}...")
    print(f"  Security groups: {vpc_config['securityGroups']}")

    ensure_batch_compute_environments(role_arns, vpc_config)
    ensure_batch_job_queues()
    ensure_batch_job_definitions(role_arns)
    ensure_lambda_functions(role_arns["lambda"])
    sm_arn = ensure_state_machine(role_arns["sfn"])
    analysis_sm_arn = ensure_analysis_state_machine(role_arns["sfn"])

    print(f"\n{'=' * 70}")
    print("Deployment complete!")
    print(f"{'=' * 70}")
    print(f"\n  Pipeline 1 (Extraction):       {sm_arn}")
    print(f"  Pipeline 2 (Ingest+Analysis):  {analysis_sm_arn}")
    print(f"  GPU EC2 profile: {GPU_EC2_INSTANCE_PROFILE}")
    print(f"  CPU Batch CE:  {BATCH_CPU_CE}")
    print(f"  Lambda prefix: {LAMBDA_PREFIX}-*")
    print(f"  GPU scripts:   s3://{S3_BUCKET}/{GPU_SCRIPTS_PREFIX}")
    print(f"  LanceDB:       {LANCEDB_URI}")
    print(f"  S3 bucket:     {S3_BUCKET}")
    print()
    print("Next steps:")
    print("  1. Build images (if not already): python deploy.py build")
    print("  2. Run: python deploy.py run")
    print()
    return sm_arn


# ---------------------------------------------------------------------------
# Run Command
# ---------------------------------------------------------------------------

def run(limit: int = None, gpu_instances: int = 60, crawl_min: str = None,
        no_wait: bool = False):
    """Start a pipeline execution."""
    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}"

    execution_input = {
        "crawl_min": crawl_min or "CC-MAIN-2024-42",
        "limit": limit,
        "setup_table": False,
        "gpu_instance_count": gpu_instances,
    }

    execution_name = f"pipeline-{int(time.time())}"
    print(f"Starting execution: {execution_name}")
    print(f"  Input: {json.dumps(execution_input, indent=2)}")

    resp = sfn.start_execution(
        stateMachineArn=sm_arn,
        name=execution_name,
        input=json.dumps(execution_input),
    )
    execution_arn = resp["executionArn"]
    print(f"  Execution ARN: {execution_arn}")

    if not no_wait:
        print("\nMonitoring execution (Ctrl+C to detach)...")
        try:
            _wait_for_execution(execution_arn)
        except KeyboardInterrupt:
            print("\nDetached. Check status with:")
            print(f"  python deploy.py status --execution-arn {execution_arn}")


def run_analysis(no_wait: bool = False):
    """Start an analysis-only pipeline execution (Pipeline 2).

    Runs FastPipeline (DuckDB) → Strategic → Export → SQLite → Mine.
    """
    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{ANALYSIS_STATE_MACHINE_NAME}"

    execution_input = {
        "s3_bucket": S3_BUCKET,
        "lancedb_uri": LANCEDB_URI,  # Still needed by StrategicAnalysis downstream
    }

    execution_name = f"analysis-{int(time.time())}"
    print(f"Starting analysis execution: {execution_name}")
    print(f"  Input: {json.dumps(execution_input, indent=2)}")

    resp = sfn.start_execution(
        stateMachineArn=sm_arn,
        name=execution_name,
        input=json.dumps(execution_input),
    )
    execution_arn = resp["executionArn"]
    print(f"  Execution ARN: {execution_arn}")

    if not no_wait:
        print("\nMonitoring execution (Ctrl+C to detach)...")
        try:
            _wait_for_execution(execution_arn)
        except KeyboardInterrupt:
            print("\nDetached. Check status with:")
            print(f"  python deploy.py status --execution-arn {execution_arn}")


def _wait_for_execution(execution_arn: str):
    """Poll until execution completes."""
    while True:
        resp = sfn.describe_execution(executionArn=execution_arn)
        status = resp["status"]

        if status == "RUNNING":
            # Get latest state
            hist = sfn.get_execution_history(
                executionArn=execution_arn,
                maxResults=5,
                reverseOrder=True,
            )
            latest_state = "Starting..."
            for evt in hist["events"]:
                if evt["type"] == "TaskStateEntered":
                    details = evt.get("stateEnteredEventDetails", {})
                    latest_state = details.get("name", "Unknown")
                    break
                elif evt["type"] == "MapStateEntered":
                    details = evt.get("stateEnteredEventDetails", {})
                    latest_state = f"Map: {details.get('name', 'Unknown')}"
                    break

            print(f"  RUNNING — current state: {latest_state}")
            time.sleep(60)
        elif status == "SUCCEEDED":
            output = json.loads(resp.get("output", "{}"))
            print(f"\n  SUCCEEDED!")
            if output.get("step5", {}).get("hypergraph_db_uri"):
                print(f"  SQLite: {output['step5']['hypergraph_db_uri']}")
            if output.get("step4", {}).get("workbook_s3_uri"):
                print(f"  Excel:  {output['step4']['workbook_s3_uri']}")
            return True
        else:
            print(f"\n  {status}")
            if resp.get("error"):
                print(f"  Error: {resp.get('error')}")
            if resp.get("cause"):
                print(f"  Cause: {resp.get('cause')}")
            return False


# ---------------------------------------------------------------------------
# Status Command
# ---------------------------------------------------------------------------

def status(execution_arn: str = None):
    """Check pipeline status."""
    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}"

    if execution_arn:
        resp = sfn.describe_execution(executionArn=execution_arn)
        print(f"Execution: {execution_arn}")
        print(f"  Status:  {resp['status']}")
        print(f"  Started: {resp.get('startDate', '')}")
        if resp.get("stopDate"):
            print(f"  Stopped: {resp['stopDate']}")
        if resp.get("output"):
            output = json.loads(resp["output"])
            print(f"  Output:  {json.dumps(output, indent=2)[:500]}")
    else:
        # List recent executions
        resp = sfn.list_executions(
            stateMachineArn=sm_arn,
            maxResults=10,
        )
        print(f"Recent executions for {STATE_MACHINE_NAME}:")
        for ex in resp["executions"]:
            print(f"  {ex['name']:30s}  {ex['status']:12s}  {ex['startDate']}")


# ---------------------------------------------------------------------------
# Teardown Command
# ---------------------------------------------------------------------------

def teardown():
    """Remove all pipeline infrastructure."""
    print("=" * 70)
    print("SIRIUS Pipeline — Teardown")
    print("=" * 70)
    print("\nThis will delete:")
    print(f"  - State Machine: {STATE_MACHINE_NAME}")
    print(f"  - Lambda functions: {LAMBDA_PREFIX}-*")
    print(f"  - Batch job queues: {BATCH_CPU_JQ}")
    print(f"  - Batch compute envs: {BATCH_CPU_CE}")
    print(f"  - GPU EC2 IAM role: {GPU_EC2_ROLE_NAME}")
    print(f"  - IAM roles: {ROLE_LAMBDA}, {ROLE_SFN}, etc.")
    print()
    confirm = input("Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    # Delete state machine
    sm_arn = f"arn:aws:states:{REGION}:{ACCOUNT_ID}:stateMachine:{STATE_MACHINE_NAME}"
    try:
        sfn.delete_state_machine(stateMachineArn=sm_arn)
        print(f"  Deleted state machine: {STATE_MACHINE_NAME}")
    except Exception as e:
        print(f"  State machine: {e}")

    # Delete Lambdas
    for name in {**LAMBDA_ZIP_FUNCTIONS, **LAMBDA_IMAGE_FUNCTIONS}:
        func_name = f"{LAMBDA_PREFIX}-{name}"
        try:
            lambda_client.delete_function(FunctionName=func_name)
            print(f"  Deleted Lambda: {func_name}")
        except Exception:
            pass

    # Disable and delete Batch job queues
    for jq_name in [BATCH_CPU_JQ]:
        try:
            batch.update_job_queue(jobQueue=jq_name, state="DISABLED")
            time.sleep(5)
            batch.delete_job_queue(jobQueue=jq_name)
            print(f"  Deleted job queue: {jq_name}")
        except Exception:
            pass

    # Disable and delete Batch compute environments
    for ce_name in [BATCH_CPU_CE]:
        try:
            batch.update_compute_environment(
                computeEnvironment=ce_name, state="DISABLED"
            )
            time.sleep(10)
            batch.delete_compute_environment(computeEnvironment=ce_name)
            print(f"  Deleted compute env: {ce_name}")
        except Exception:
            pass

    # Delete IAM roles (detach policies first)
    for role_name in [ROLE_LAMBDA, ROLE_SFN, ROLE_BATCH_INSTANCE, ROLE_BATCH_JOB,
                      GPU_EC2_ROLE_NAME]:
        try:
            # Detach managed policies
            attached = iam.list_attached_role_policies(RoleName=role_name)
            for pol in attached.get("AttachedPolicies", []):
                iam.detach_role_policy(
                    RoleName=role_name, PolicyArn=pol["PolicyArn"]
                )
            # Delete inline policies
            inline = iam.list_role_policies(RoleName=role_name)
            for pol_name in inline.get("PolicyNames", []):
                iam.delete_role_policy(RoleName=role_name, PolicyName=pol_name)
            iam.delete_role(RoleName=role_name)
            print(f"  Deleted IAM role: {role_name}")
        except Exception:
            pass

    # Delete instance profiles
    for profile_name, profile_role in [
        (INSTANCE_PROFILE_BATCH, ROLE_BATCH_INSTANCE),
        (GPU_EC2_INSTANCE_PROFILE, GPU_EC2_ROLE_NAME),
    ]:
        try:
            iam.remove_role_from_instance_profile(
                InstanceProfileName=profile_name,
                RoleName=profile_role,
            )
            iam.delete_instance_profile(InstanceProfileName=profile_name)
            print(f"  Deleted instance profile: {profile_name}")
        except Exception:
            pass

    print("\nTeardown complete. ECR repos and S3 data preserved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SIRIUS Pipeline — AWS Step Function Deployment"
    )
    sub = parser.add_subparsers(dest="command")

    build_parser = sub.add_parser("build",
                                  help="Build and push Docker images via CodeBuild")
    build_parser.add_argument("--image", nargs="+", default=None,
                              choices=list(DOCKER_IMAGES.keys()),
                              help="Build specific images (default: all)")

    sub.add_parser("deploy", help="Deploy all infrastructure")

    run_parser = sub.add_parser("run", help="Start a pipeline execution")
    run_parser.add_argument("--limit", type=int, default=None,
                            help="Limit WARC records (for testing)")
    run_parser.add_argument("--gpu-instances", type=int, default=60,
                            help="Number of GPU instances for extraction")
    run_parser.add_argument("--crawl-min", default=None,
                            help="Minimum CC crawl version (default: CC-MAIN-2024-42)")
    run_parser.add_argument("--no-wait", action="store_true",
                            help="Don't wait for execution to complete")

    status_parser = sub.add_parser("status", help="Check execution status")
    status_parser.add_argument("--execution-arn", default=None,
                               help="Specific execution ARN to check")

    analysis_parser = sub.add_parser("run-analysis",
                                      help="Start analysis-only pipeline (Pipeline 2)")
    analysis_parser.add_argument("--no-wait", action="store_true",
                                  help="Don't wait for execution to complete")

    sub.add_parser("teardown", help="Remove all infrastructure")

    args = parser.parse_args()

    if args.command == "build":
        build(images=args.image)
    elif args.command == "deploy":
        deploy()
    elif args.command == "run":
        run(
            limit=args.limit,
            gpu_instances=args.gpu_instances,
            crawl_min=args.crawl_min,
            no_wait=args.no_wait,
        )
    elif args.command == "run-analysis":
        run_analysis(no_wait=args.no_wait)
    elif args.command == "status":
        status(execution_arn=args.execution_arn)
    elif args.command == "teardown":
        teardown()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
