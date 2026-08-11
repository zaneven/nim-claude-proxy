"""CLI entrypoint for nim-claude-proxy management commands (ncp)."""

import sys
import subprocess
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    ncp_script = project_root / "scripts" / "ncp.sh"

    if ncp_script.exists():
        args = sys.argv[1:] if len(sys.argv) > 1 else ["help"]
        result = subprocess.run([str(ncp_script), *args], cwd=str(project_root))
        sys.exit(result.returncode)
    else:
        print("[ERROR] scripts/ncp.sh script not found in project directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
