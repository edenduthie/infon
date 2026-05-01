"""
Integration tests for ``write_mcp_config`` venv detection.

These tests verify the venv-detection logic in
``infon.cli.write_mcp_config``. They use ``monkeypatch`` to manipulate
real ``sys`` attributes and environment variables (no mocks/MagicMock
are used; ``monkeypatch`` patches real attributes).

Per Phase 13 spec ("Three Install Paths"):

- Outside a venv: ``.mcp.json`` uses the ``uvx`` form so Claude Code
  can run the latest published version with no install.
- Inside a venv: ``.mcp.json`` pins to the venv binary so the version
  is locked to whatever is installed in that environment.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from infon.cli import write_mcp_config


@pytest.fixture
def fresh_repo(tmp_path: Path) -> Path:
    """A tmp directory with no pre-existing ``.mcp.json``."""
    assert not (tmp_path / ".mcp.json").exists()
    return tmp_path


def _force_outside_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make detection believe we are outside any virtualenv."""
    # Both signals must indicate "no venv" for the detection to treat
    # the interpreter as a system / uvx Python.
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def _force_inside_venv(
    monkeypatch: pytest.MonkeyPatch,
    venv_prefix: Path,
) -> None:
    """Make detection believe we are inside ``venv_prefix``."""
    monkeypatch.setattr(sys, "prefix", str(venv_prefix))
    # Set base_prefix to something different so the standard venv check
    # ``sys.prefix != sys.base_prefix`` evaluates True.
    monkeypatch.setattr(sys, "base_prefix", str(venv_prefix.parent))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_prefix))


def _make_fake_venv(root: Path) -> Path:
    """
    Create a directory tree that looks like a venv (POSIX layout) and
    contains an executable ``infon`` binary. Returns the venv root.
    """
    if sys.platform == "win32":
        bin_dir = root / "Scripts"
        binary = bin_dir / "infon.exe"
    else:
        bin_dir = root / "bin"
        binary = bin_dir / "infon"
    bin_dir.mkdir(parents=True)
    binary.write_text("#!/usr/bin/env bash\nexec true\n")
    binary.chmod(0o755)
    return root


def test_write_mcp_config_outside_venv_uses_uvx(
    fresh_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a venv the config must use the uvx command form."""
    _force_outside_venv(monkeypatch)

    write_mcp_config(fresh_repo)

    mcp_config = json.loads((fresh_repo / ".mcp.json").read_text())
    server = mcp_config["mcpServers"]["infon"]
    assert server["type"] == "stdio"
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", "infon", "infon", "serve"]


def test_write_mcp_config_inside_venv_uses_binary_path(
    fresh_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a venv the config must use the absolute venv binary path."""
    venv_root = _make_fake_venv(tmp_path / "fake_venv")
    _force_inside_venv(monkeypatch, venv_root)

    write_mcp_config(fresh_repo)

    mcp_config = json.loads((fresh_repo / ".mcp.json").read_text())
    server = mcp_config["mcpServers"]["infon"]
    assert server["type"] == "stdio"
    assert server["args"] == ["serve"]

    expected_binary = (
        venv_root / ("Scripts/infon.exe" if sys.platform == "win32" else "bin/infon")
    ).resolve()
    assert server["command"] == str(expected_binary)
    # The recorded path must be absolute so Claude Code can launch it
    # from any working directory.
    assert os.path.isabs(server["command"])


def test_write_mcp_config_venv_missing_binary_falls_back_to_uvx(
    fresh_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    If we appear to be inside a venv but the ``infon`` binary is missing
    from it, fall back to the uvx form and emit a warning instead of
    crashing.
    """
    # Build a venv-shaped directory with NO infon binary.
    venv_root = tmp_path / "broken_venv"
    if sys.platform == "win32":
        (venv_root / "Scripts").mkdir(parents=True)
    else:
        (venv_root / "bin").mkdir(parents=True)
    _force_inside_venv(monkeypatch, venv_root)

    write_mcp_config(fresh_repo)

    mcp_config = json.loads((fresh_repo / ".mcp.json").read_text())
    server = mcp_config["mcpServers"]["infon"]
    assert server["command"] == "uvx"
    assert server["args"] == ["--from", "infon", "infon", "serve"]

    # The user must be told why we did not pin to the venv path.
    captured = capsys.readouterr()
    assert "warning" in (captured.out + captured.err).lower()


def test_write_mcp_config_does_not_overwrite_existing(
    fresh_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing ``.mcp.json`` is never overwritten by ``infon init``."""
    _force_outside_venv(monkeypatch)
    existing = {"mcpServers": {"other": {"command": "do-not-touch"}}}
    (fresh_repo / ".mcp.json").write_text(json.dumps(existing))

    write_mcp_config(fresh_repo)

    assert json.loads((fresh_repo / ".mcp.json").read_text()) == existing
