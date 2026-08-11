"""Unit tests for ncp CLI module and scripts/ncp.sh script management commands."""

import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_ncp_script_help(project_root: Path) -> None:
    script_path = project_root / "scripts" / "ncp.sh"
    assert script_path.exists(), "scripts/ncp.sh must exist"

    res = subprocess.run([str(script_path), "help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "NIM Claude Proxy (ncp) 服务管理与 CLI 工具" in res.stdout
    assert "install" in res.stdout
    assert "start" in res.stdout
    assert "stop" in res.stdout
    assert "restart" in res.stdout
    assert "status" in res.stdout


def test_ncp_script_status_stopped(project_root: Path) -> None:
    script_path = project_root / "scripts" / "ncp.sh"
    res = subprocess.run([str(script_path), "status"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "NIM Proxy 服务运行状态" in res.stdout


def test_cli_main_entrypoint(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nim_proxy.cli import main

    monkeypatch.setattr(sys, "argv", ["ncp", "help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
