"""Lambda container image packaging — Python-only deploy path.

Why a container instead of a layer: torch+transformers+pyarrow unzipped
is ~1.8GB, well over Lambda's 250MB layer limit. Container images have
a 10GB limit, more than enough headroom.

What this module does:
  1. Generate a Dockerfile + build context (handler.py, requirements.txt)
  2. Build via `docker buildx build --platform linux/amd64`
     (cross-arch on Apple Silicon)
  3. Auth with ECR via boto3.get_authorization_token
  4. Push the image
  5. Create/update a Lambda function with PackageType=Image

Cold start is ~2-3s first invocation, ~100ms warm. The extra cold-start
vs a layer is the price of unlimited size.

All operations stay in Python. Requires docker CLI on PATH (buildx
is standard since Docker 20). boto3 is optional until push/deploy.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
from pathlib import Path


# AWS Lambda's Python base image — maintained by AWS, includes the
# runtime interface emulator so the container responds to Lambda events.
_BASE_IMAGE = "public.ecr.aws/lambda/python:3.11"

# Lambda expects exactly x86_64 or arm64; cross-build tag is standard.
_LAMBDA_PLATFORM = "linux/amd64"

# Default requirements (mirrors layer defaults — but no version pin
# panic since we have 10GB not 250MB to work with).
_DEFAULT_REQS = [
    "torch>=2.6,<2.11 --index-url https://download.pytorch.org/whl/cpu",
    "transformers>=4.40,<5.0",
    "numpy>=1.24,<2.2",
    "pyarrow>=15,<24",
    "fsspec>=2024.1",
    "s3fs>=2024.1",
]


_DOCKERFILE_TEMPLATE = """\
# syntax=docker/dockerfile:1
FROM {base_image}

# Install runtime deps first so they cache independently of code edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the infon package (bundles 17MB SPLADE + heads in infon/model)
COPY infon/ ${{LAMBDA_TASK_ROOT}}/infon/

# Handler script (tiny — just unwraps events into the worker)
COPY handler.py ${{LAMBDA_TASK_ROOT}}/

CMD ["handler.handler"]
"""


def _run(cmd: list[str], cwd: str | None = None, capture: bool = False) -> str:
    """Subprocess with streaming output (or capture); raise on non-zero."""
    print(f"  $ {' '.join(cmd)}")
    if capture:
        p = subprocess.run(cmd, cwd=cwd, check=False,
                           capture_output=True, text=True)
    else:
        p = subprocess.run(cmd, cwd=cwd, check=False, text=True)
    if p.returncode != 0:
        msg = f"command failed (rc={p.returncode}): {' '.join(cmd)}"
        if capture and p.stderr:
            msg += f"\n  stderr: {p.stderr.strip()}"
        raise RuntimeError(msg)
    return (p.stdout or "") if capture else ""


def _write_build_context(ctx_dir: Path, requirements: list[str],
                          base_image: str) -> None:
    """Lay out Dockerfile + requirements + infon/ + handler.py.

    Everything below this dir is sent to the daemon as the build context
    — keep it small (just what the image needs)."""
    import infon

    infon_src = Path(infon.__file__).parent

    # Drop __pycache__ to avoid stale .pyc files making the image nondeterministic.
    def _ignore(d, names):
        return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]

    dst_cog = ctx_dir / "infon"
    if dst_cog.exists():
        shutil.rmtree(dst_cog)
    shutil.copytree(infon_src, dst_cog, ignore=_ignore)

    # handler.py lives inside the infon package; copy it to the root
    # so the Lambda runtime finds it via CMD ["handler.handler"].
    handler_src = infon_src / "cassette" / "handler.py"
    shutil.copy(handler_src, ctx_dir / "handler.py")

    (ctx_dir / "requirements.txt").write_text("\n".join(requirements) + "\n")
    (ctx_dir / "Dockerfile").write_text(
        _DOCKERFILE_TEMPLATE.format(base_image=base_image)
    )


def build_image(
    *,
    image_tag: str,
    build_dir: str | Path = "./build/lambda-image",
    requirements: list[str] | None = None,
    base_image: str = _BASE_IMAGE,
    platform: str = _LAMBDA_PLATFORM,
) -> str:
    """Build a Lambda-compatible container image locally.

    Args:
      image_tag: Docker tag for the built image, e.g. "infon-ingest:v1".
      build_dir: where to stage the Dockerfile + context (overwritten).
      requirements: pip requirements lines (string form; supports
        "--index-url" embedded tokens for the torch CPU wheel).
      base_image: Lambda Python base — the AWS-maintained image with RIC.
      platform: buildx platform string. Default linux/amd64 matches
        Lambda's x86_64 runtime regardless of build-host arch.

    Returns the image tag (confirms success). Does NOT push — see
    `push_image()` for ECR upload.
    """
    reqs = requirements or list(_DEFAULT_REQS)
    build_dir = Path(build_dir).resolve()
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    print(f"  → writing build context to {build_dir}")
    _write_build_context(build_dir, reqs, base_image)

    print(f"  → docker buildx (platform={platform}, base={base_image})")
    _run([
        "docker", "buildx", "build",
        "--platform", platform,
        "--provenance=false",    # cleaner manifest for Lambda
        "--load",                 # move the image into the local daemon
        "-t", image_tag,
        str(build_dir),
    ])

    # Report size so callers can see what they've built.
    out = _run(["docker", "image", "inspect", image_tag,
                "--format", "{{.Size}}"], capture=True)
    try:
        size_bytes = int(out.strip())
        print(f"  → image size: {size_bytes / 1024 / 1024:.0f}MB")
    except ValueError:
        print(f"  → image built, inspect returned: {out!r}")
    return image_tag


def ensure_ecr_repo(repo_name: str, region: str) -> str:
    """Create an ECR repo if missing; return the registry URI prefix."""
    import boto3
    from botocore.exceptions import ClientError

    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.describe_repositories(repositoryNames=[repo_name])
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "RepositoryNotFoundException":
            raise
        print(f"  → creating ECR repo {repo_name}")
        ecr.create_repository(repositoryName=repo_name,
                               imageScanningConfiguration={"scanOnPush": True})

    # The repository URI is host/name; we want host only for tagging.
    # host = "<acct>.dkr.ecr.<region>.amazonaws.com"
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    return f"{account}.dkr.ecr.{region}.amazonaws.com"


def _docker_login(registry_host: str, region: str) -> None:
    """Exchange boto3 ECR auth token for a `docker login` session."""
    import boto3

    ecr = boto3.client("ecr", region_name=region)
    auth = ecr.get_authorization_token()["authorizationData"][0]
    token = base64.b64decode(auth["authorizationToken"]).decode()
    user, password = token.split(":", 1)
    # Pipe the password via stdin so it doesn't show in `ps`.
    proc = subprocess.Popen(
        ["docker", "login", "--username", user, "--password-stdin",
         registry_host],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=password.encode())
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker login failed (rc={proc.returncode}): "
            f"{err.decode().strip()}"
        )


def push_image(
    *,
    image_tag: str,
    repo_name: str,
    region: str,
    tag: str = "latest",
) -> str:
    """Tag and push an image to ECR. Returns the full image URI.

    Uses boto3 to mint a short-lived ECR auth token, pipes it into
    `docker login`, then `docker push`. Caller must have IAM permission
    for ecr:GetAuthorizationToken + PutImage.
    """
    try:
        import boto3  # noqa: F401
    except ImportError:
        raise ImportError("push_image requires boto3. `pip install boto3`.")

    registry_host = ensure_ecr_repo(repo_name, region)
    _docker_login(registry_host, region)

    target = f"{registry_host}/{repo_name}:{tag}"
    print(f"  → tagging {image_tag} → {target}")
    _run(["docker", "tag", image_tag, target])
    print(f"  → pushing")
    _run(["docker", "push", target])

    # Resolve the digest so callers can pin rather than chase `:latest`.
    out = _run(["docker", "inspect", "--format",
                "{{index .RepoDigests 0}}", target], capture=True)
    digest_ref = out.strip() or target
    print(f"  → pushed {digest_ref}")
    return digest_ref


def publish_function(
    *,
    image_uri: str,
    function_name: str,
    region: str,
    role_arn: str,
    memory_mb: int = 2048,
    timeout_s: int = 300,
    environment: dict | None = None,
) -> dict:
    """Create or update a Lambda function backed by a container image.

    image_uri must be an ECR URI (public images aren't allowed by Lambda).
    The runtime is determined by the image, so no `Runtime` field.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError("publish_function requires boto3.")

    client = boto3.client("lambda", region_name=region)
    kwargs = dict(
        FunctionName=function_name,
        PackageType="Image",
        Code={"ImageUri": image_uri},
        Role=role_arn,
        Timeout=timeout_s,
        MemorySize=memory_mb,
    )
    if environment:
        kwargs["Environment"] = {"Variables": environment}

    try:
        resp = client.create_function(**kwargs)
        print(f"  → created {resp['FunctionArn']}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceConflictException":
            raise
        # Exists → update image then config.
        client.update_function_code(
            FunctionName=function_name, ImageUri=image_uri,
        )
        client.get_waiter("function_updated").wait(FunctionName=function_name)
        client.update_function_configuration(
            FunctionName=function_name,
            Role=role_arn,
            Timeout=timeout_s,
            MemorySize=memory_mb,
            Environment={"Variables": environment or {}},
        )
        client.get_waiter("function_updated").wait(FunctionName=function_name)
        resp = client.get_function(FunctionName=function_name)["Configuration"]
        print(f"  → updated {resp['FunctionArn']}")
    return resp
