"""Build and publish Lambda layers for cassette ingestion — Python only.

Two operations:
  build_layer(...)  → local zip file, no AWS
  publish_layer(...) → upload zip and create LayerVersion via boto3

Single-layer-for-now design: we bundle torch-cpu + transformers + cognition
+ the 17MB model into ONE layer (~180MB unzipped, well under Lambda's 250MB
limit). We can split the model into its own layer later if/when the model
gets updated more often than dependencies; for now the extra complexity
isn't justified.

Two size tricks to stay under 250MB:
  - torch==2.0.1+cpu (strips CUDA; ~180MB vs ~700MB for default torch)
  - skip tests/ and __pycache__ inside installed packages

Usage:
    from infon.cassette.lambda_package import build_layer, publish_layer

    zip_path = build_layer(output_dir="./build")
    # → .zip file, no AWS touched

    arn = publish_layer(
        zip_path=zip_path,
        layer_name="infon-runtime",
        region="us-east-1",
    )
    # → arn:aws:lambda:us-east-1:123:layer:cognition-runtime:1
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


# PyTorch CPU wheels. PyPI only serves CUDA builds so we need the
# explicit index. Version window chosen for current availability (2.0.1
# has been yanked from the CPU index as of late 2026).
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_DEFAULT_REQS = [
    "torch>=2.6,<2.11",
    "transformers>=4.40,<5.0",
    "numpy>=1.24,<2.2",
    "pyarrow>=15,<24",
    "fsspec>=2024.1",
    "s3fs>=2024.1",
]

# Lambda layers must unpack to /opt and Python picks up /opt/python on
# sys.path. This is the layout we create in the zip.
_LAYER_PYTHON_PREFIX = "python"

# Lambda's x86_64 runtime — cross-compile target when the build host is
# macOS or a different arch. `manylinux2014_x86_64` matches the AL2 base.
_LAMBDA_PLATFORM = "manylinux2014_x86_64"


def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a subprocess, stream output, raise on non-zero exit."""
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}): {' '.join(cmd)}")


def _prune_unused(site_packages: Path) -> int:
    """Strip tests, examples, .pyc, and similar bloat. Returns bytes removed.

    We aim to get under 250MB unzipped; removing these shaves ~30-50MB off
    a typical torch + transformers install.
    """
    removed = 0
    patterns = [
        "**/__pycache__", "**/tests", "**/test", "**/testing",
        "**/examples", "**/benchmarks", "**/docs", "**/*.pyi",
        "**/*.dist-info/RECORD",  # invalid after pruning anyway
    ]
    for pattern in patterns:
        for p in site_packages.glob(pattern):
            if p.is_dir():
                removed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                removed += p.stat().st_size
                p.unlink(missing_ok=True)
    # Kill bundled CUDA libs if any slipped in (they shouldn't with +cpu)
    for p in list(site_packages.glob("torch/lib/libcu*")) + \
              list(site_packages.glob("torch/lib/libnccl*")):
        removed += p.stat().st_size
        p.unlink(missing_ok=True)
    return removed


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build_layer(
    *,
    output_dir: str | Path = "./build",
    requirements: list[str] | None = None,
    include_cognition: bool = True,
    include_model: bool = True,
    model_dir: str | Path | None = None,
    python_version: str = "3.11",
) -> Path:
    """Build a Lambda layer zip. No AWS calls.

    Args:
      output_dir: where the final .zip lands.
      requirements: pip specs. Default = torch-cpu + transformers + support libs.
      include_cognition: bundle the cognition package itself.
      include_model: bundle cognition/model/ (17MB SPLADE + heads + tokenizer).
      model_dir: explicit model directory (default: auto-detect in cognition
        package). Ignored if include_model=False.
      python_version: not used directly, but cognition requires ≥3.11.

    Returns: path to the built .zip file.

    Pins torch==2.0.1+cpu via the PyTorch index to stay under Lambda's
    250MB layer limit (default torch is ~700MB unzipped with CUDA libs).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reqs = requirements or list(_DEFAULT_REQS)

    with tempfile.TemporaryDirectory() as tmp:
        build_root = Path(tmp)
        # Lambda layer layout: the zip contains "python/<packages>"
        # which gets extracted to /opt/python at runtime.
        site_packages = build_root / _LAYER_PYTHON_PREFIX
        site_packages.mkdir()

        # pip install --target — drop packages in a directory without a
        # venv. `--platform` + `--only-binary=:all:` cross-compiles for
        # Lambda's Linux x86_64 runtime regardless of build host (important
        # on macOS, where pip would otherwise grab Darwin/ARM wheels).
        print(f"  → pip installing into {site_packages}")
        pip_cmd = [
            sys.executable, "-m", "pip", "install",
            "--target", str(site_packages),
            "--upgrade",
            "--platform", _LAMBDA_PLATFORM,
            "--only-binary=:all:",
            "--python-version", python_version,
            "--extra-index-url", _TORCH_CPU_INDEX,
            *reqs,
        ]
        _run(pip_cmd)

        if include_cognition:
            print("  → copying cognition package")
            import infon
            infon_src = Path(infon.__file__).parent
            dst = site_packages / "infon"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(
                infon_src, dst,
                # Don't double-ship model files here — they go in their
                # own slot below (or not at all).
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc",
                    "model" if not include_model else "",
                ),
            )

        if include_model:
            # The 17MB SPLADE + heads + tokenizer lives inside the cognition
            # package. If the user disabled include_cognition but wants the
            # model, copy just that subtree.
            if model_dir is None:
                import infon
                model_dir = Path(infon.__file__).parent / "model"
            model_dir = Path(model_dir)
            if not model_dir.exists():
                raise FileNotFoundError(f"model dir not found: {model_dir}")
            dst_model = site_packages / "infon" / "model"
            if dst_model.exists():
                shutil.rmtree(dst_model)
            dst_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(model_dir, dst_model)

        print("  → pruning")
        bytes_removed = _prune_unused(site_packages)
        print(f"    pruned {bytes_removed / 1024 / 1024:.1f}MB")

        unzipped = _dir_size(site_packages)
        print(f"  → unzipped size: {unzipped / 1024 / 1024:.1f}MB")
        if unzipped > 250 * 1024 * 1024:
            print(f"  ⚠  exceeds Lambda layer limit (250MB unzipped). "
                  f"Consider splitting or trimming further.")

        zip_path = output_dir / "infon-runtime.zip"
        print(f"  → zipping → {zip_path}")
        with zipfile.ZipFile(zip_path, "w",
                              compression=zipfile.ZIP_DEFLATED,
                              compresslevel=6) as zf:
            for f in build_root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(build_root))

        zipped = zip_path.stat().st_size
        print(f"  → zipped size:   {zipped / 1024 / 1024:.1f}MB")
        if zipped > 50 * 1024 * 1024:
            print(f"  ⚠  zip > 50MB — must use S3 upload path for publish, "
                  f"not direct PutObject. publish_layer() handles this.")
        return zip_path


def publish_layer(
    *,
    zip_path: str | Path,
    layer_name: str,
    region: str,
    description: str = "",
    compatible_runtimes: list[str] | None = None,
    upload_bucket: str | None = None,
) -> dict:
    """Publish a layer zip to AWS Lambda. Returns the LayerVersion response.

    Args:
      zip_path: local path from build_layer().
      layer_name: Lambda layer name (created or incremented).
      region: AWS region.
      description: optional; will include zip sha256 so versions are traceable.
      compatible_runtimes: default ["python3.11"].
      upload_bucket: S3 bucket to stage the zip. Required for zips > 50MB
        (Lambda's direct-upload limit). If None and zip is small, uploads
        inline.

    Requires boto3. Uses standard AWS credential chain.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "publish_layer requires boto3. `pip install boto3` and configure "
            "AWS credentials before calling."
        )

    zip_path = Path(zip_path)
    zip_bytes = zip_path.stat().st_size
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()[:12]
    description = description or f"infon layer  sha256={sha}"

    runtimes = compatible_runtimes or ["python3.11"]
    lambda_client = boto3.client("lambda", region_name=region)

    if zip_bytes <= 50 * 1024 * 1024 and upload_bucket is None:
        # Small enough for direct upload.
        print(f"  → direct-uploading {zip_bytes / 1024 / 1024:.1f}MB")
        with open(zip_path, "rb") as f:
            resp = lambda_client.publish_layer_version(
                LayerName=layer_name,
                Description=description,
                Content={"ZipFile": f.read()},
                CompatibleRuntimes=runtimes,
            )
    else:
        # Large zips must go via S3.
        if upload_bucket is None:
            raise ValueError(
                f"zip is {zip_bytes / 1024 / 1024:.1f}MB, exceeds Lambda's "
                f"50MB direct-upload limit. Pass upload_bucket=."
            )
        s3 = boto3.client("s3", region_name=region)
        key = f"lambda-layers/{layer_name}/{sha}.zip"
        print(f"  → uploading to s3://{upload_bucket}/{key}")
        s3.upload_file(str(zip_path), upload_bucket, key)
        resp = lambda_client.publish_layer_version(
            LayerName=layer_name,
            Description=description,
            Content={"S3Bucket": upload_bucket, "S3Key": key},
            CompatibleRuntimes=runtimes,
        )

    print(f"  → published {resp['LayerVersionArn']}")
    return resp


def build_function_zip(output_dir: str | Path = "./build") -> Path:
    """Build the tiny handler zip (just handler.py — everything else in layer)."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "infon-handler.zip"
    import infon
    handler_src = Path(infon.__file__).parent / "cassette" / "handler.py"
    with zipfile.ZipFile(zip_path, "w",
                          compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_src, "handler.py")
    print(f"  → handler zip: {zip_path.stat().st_size / 1024:.1f}KB")
    return zip_path


def publish_function(
    *,
    zip_path: str | Path,
    function_name: str,
    region: str,
    layer_arns: list[str],
    role_arn: str,
    memory_mb: int = 2048,
    timeout_s: int = 300,
    runtime: str = "python3.11",
    environment: dict | None = None,
) -> dict:
    """Create or update a Lambda function. Requires boto3 + IAM role already set.

    memory_mb=2048 is a sensible default for SPLADE extraction — enough
    headroom for torch + transformers, small enough to stay cheap (~$0.03/hr
    when hot). timeout_s=300 gives room for cold starts + batches of ~100
    docs per invocation; raise if you send bigger batches.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError("publish_function requires boto3.")

    client = boto3.client("lambda", region_name=region)

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    kwargs = dict(
        FunctionName=function_name,
        Runtime=runtime,
        Role=role_arn,
        Handler="handler.handler",
        Code={"ZipFile": zip_bytes},
        Timeout=timeout_s,
        MemorySize=memory_mb,
        Layers=layer_arns,
    )
    if environment:
        kwargs["Environment"] = {"Variables": environment}

    try:
        resp = client.create_function(**kwargs)
        print(f"  → created {resp['FunctionArn']}")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceConflictException":
            raise
        # Function exists — update it.
        client.update_function_code(
            FunctionName=function_name, ZipFile=zip_bytes,
        )
        # Wait for update before config change; otherwise AWS rejects it.
        client.get_waiter("function_updated").wait(FunctionName=function_name)
        client.update_function_configuration(
            FunctionName=function_name,
            Role=role_arn,
            Handler="handler.handler",
            Timeout=timeout_s,
            MemorySize=memory_mb,
            Layers=layer_arns,
            Environment={"Variables": environment or {}},
        )
        client.get_waiter("function_updated").wait(FunctionName=function_name)
        resp = client.get_function(FunctionName=function_name)["Configuration"]
        print(f"  → updated {resp['FunctionArn']}")
    return resp
