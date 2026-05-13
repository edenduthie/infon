"""Lambda handler: Launch EC2 GPU instances for NuExtract-2.0-8B extraction.

Replaces the AWS Batch GPU step with direct EC2 instances using the
Deep Learning OSS Nvidia Driver AMI and user-data bootstrap.  Each instance:
  1. Installs Python 3.11 + vLLM + deps
  2. Downloads extraction scripts from S3
  3. Runs gpu_extract.py against the SQS work queue
  4. Self-terminates when the queue is drained
"""

import base64
import json
import os

import boto3

REGION = os.environ.get("AWS_REGION_OVERRIDE",
                        os.environ.get("AWS_REGION", "us-east-1"))
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "038462780313")
S3_BUCKET = os.environ.get("S3_BUCKET", "sirius-dimefiled-results")
SCRIPTS_PREFIX = os.environ.get("GPU_SCRIPTS_PREFIX", "gpu-scripts/")
INSTANCE_PROFILE_NAME = os.environ.get("GPU_INSTANCE_PROFILE",
                                       "sirius-gpu-nuextract-profile")
INSTANCE_TYPE = os.environ.get("GPU_INSTANCE_TYPE", "g6e.2xlarge")
INSTANCE_TAG = "NuExtract-GPU-Extractor"

ec2 = boto3.client("ec2", region_name=REGION)

# ---------------------------------------------------------------------------
# User-data script (runs on EC2 startup)
# ---------------------------------------------------------------------------

USER_DATA_TEMPLATE = """#!/bin/bash
set -e
exec > >(tee /var/log/user-data.log) 2>&1

echo "============================================"
echo "NuExtract-2.0-8B GPU Extraction -- Instance Bootstrap"
echo "============================================"
date
nvidia-smi

# The OSS Nvidia Driver AMI ships Python 3.9, but vLLM >=0.11 requires Python 3.10+
# (uses PEP 604 union types: `type | None`). Install Python 3.11 and create a venv.
echo "[1/6] Installing Python 3.11, pip, and build tools..."
dnf install -y python3.11 python3.11-pip python3.11-devel gcc

echo "[2/6] Creating Python 3.11 virtualenv..."
python3.11 -m venv /opt/nuextract
source /opt/nuextract/bin/activate

echo "[3/6] Installing vLLM and extraction dependencies..."
pip install --upgrade pip
pip install vllm dateparser beautifulsoup4 lxml boto3

# Verify vLLM + CUDA
python3 -c "import vllm; print(f'vLLM {{vllm.__version__}}')"
python3 -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA {{torch.cuda.is_available()}}, GPUs {{torch.cuda.device_count()}}')"

echo "[4/6] Downloading scripts from S3..."
mkdir -p /workspace
cd /workspace
aws s3 cp s3://{output_bucket}/{scripts_prefix}gpu_extract.py .
aws s3 cp s3://{output_bucket}/{scripts_prefix}nuextract_template.py .
aws s3 cp s3://{output_bucket}/{scripts_prefix}warc_fetch.py .
aws s3 cp s3://{output_bucket}/{scripts_prefix}postprocess.py .

echo "[5/6] Pre-downloading NuExtract model weights..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('numind/NuExtract-2.0-8B')" || echo "Model pre-download (non-fatal)"

# Prevent runtime HF Hub deadlocks (model2vec, tokenizer reloads)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[6/6] Running GPU extraction (SQS mode)..."
python3 gpu_extract.py \\
    --sqs-queue-url {sqs_queue_url} \\
    --num-gpus {num_gpus}

echo "Extraction complete -- self-terminating..."
INSTANCE_ID=$(ec2-metadata --instance-id | cut -d ' ' -f 2)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region {region}
echo "Instance terminating..."
"""


# ---------------------------------------------------------------------------
# AMI lookup
# ---------------------------------------------------------------------------

def find_deep_learning_ami() -> str:
    """Find the latest AWS Deep Learning OSS Nvidia Driver AMI (AL2023, PyTorch)."""
    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name",
             "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch *(Amazon Linux 2023)*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = response.get("Images", [])
    if not images:
        # Known-good fallback AMI
        print("[WARN] Could not find Deep Learning GPU AMI. Using known fallback.")
        return "ami-031be0603a862f6e0"

    images.sort(key=lambda x: x.get("CreationDate", ""), reverse=True)
    ami = images[0]
    print(f"AMI: {ami['ImageId']} -- {ami['Name']}")
    return ami["ImageId"]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event, context):
    """Launch N EC2 GPU instances for SQS-based NuExtract extraction.

    Input:
        queue_url:           SQS queue URL with WARC records
        gpu_instance_count:  Number of g6e.2xlarge instances to launch

    Output:
        instance_ids:   List of launched EC2 instance IDs
        launched_count: Number successfully launched
        instance_type:  EC2 instance type used
    """
    queue_url = event["queue_url"]
    instance_count = int(event.get("gpu_instance_count", 1))
    num_gpus = 1  # g6e.2xlarge has 1 L40S GPU

    # Resolve AMI
    ami_id = find_deep_learning_ami()

    # Build user-data
    user_data = USER_DATA_TEMPLATE.format(
        output_bucket=S3_BUCKET,
        scripts_prefix=SCRIPTS_PREFIX,
        sqs_queue_url=queue_url,
        num_gpus=num_gpus,
        region=REGION,
    )
    user_data_b64 = base64.b64encode(user_data.encode()).decode()

    # Launch instances
    instance_ids = []
    for i in range(instance_count):
        try:
            resp = ec2.run_instances(
                ImageId=ami_id,
                InstanceType=INSTANCE_TYPE,
                MinCount=1,
                MaxCount=1,
                UserData=user_data_b64,
                IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
                BlockDeviceMappings=[{
                    "DeviceName": "/dev/xvda",
                    "Ebs": {
                        "VolumeSize": 150,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }],
                TagSpecifications=[{
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": INSTANCE_TAG},
                        {"Key": "Project", "Value": "SiriusBeta-Pipeline"},
                        {"Key": "AutoTerminate", "Value": "true"},
                        {"Key": "SQSQueue", "Value": queue_url},
                    ],
                }],
            )
            inst_id = resp["Instances"][0]["InstanceId"]
            instance_ids.append(inst_id)
            print(f"  Launched {i+1}/{instance_count}: {inst_id}")
        except Exception as e:
            print(f"  Instance {i+1}/{instance_count}: FAILED -- {e}")
            if ("InsufficientInstanceCapacity" in str(e)
                    or "InstanceLimitExceeded" in str(e)):
                print(f"  Hit capacity limit after {len(instance_ids)} instances")
                break

    print(f"Launched {len(instance_ids)}/{instance_count} {INSTANCE_TYPE} instances")
    print(f"SQS queue: {queue_url}")

    return {
        "instance_ids": instance_ids,
        "launched_count": len(instance_ids),
        "instance_type": INSTANCE_TYPE,
    }
