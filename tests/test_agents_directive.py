"""Tests ensuring AGENTS.md and CLAUDE.md remain identical."""

from pathlib import Path


def test_agents_and_claude_md_identical():
    root = Path(__file__).parent.parent
    agents_md = root / "AGENTS.md"
    claude_md = root / "CLAUDE.md"

    assert agents_md.exists(), "AGENTS.md does not exist"
    assert claude_md.exists(), "CLAUDE.md does not exist"

    agents_content = agents_md.read_text(encoding="utf-8")
    claude_content = claude_md.read_text(encoding="utf-8")

    assert agents_content == claude_content, "AGENTS.md and CLAUDE.md must be identical!"
