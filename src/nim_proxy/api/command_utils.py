"""Command parsing utilities for API optimizations.

Extracts command prefixes and file paths from command strings without
making API calls.
"""

import re
import shlex

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _is_env_assignment(part: str) -> bool:
    return bool(_ENV_ASSIGNMENT_RE.match(part))


def _strip_env_assignments(parts: list[str]) -> list[str]:
    cmd_start = 0
    for i, part in enumerate(parts):
        if _is_env_assignment(part):
            cmd_start = i + 1
        else:
            break
    return parts[cmd_start:]


def extract_command_prefix(command: str) -> str:
    """Extract the command prefix for fast prefix detection.

    Returns the command prefix (e.g. "git", "git commit", "npm install")
    or "none" if no valid command found.
    """
    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command, posix=False)
        if not parts:
            return "none"

        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if _is_env_assignment(part):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {
            "git", "npm", "docker", "kubectl", "cargo", "go", "pip", "yarn",
        }

        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            if not second_word.startswith("-"):
                return f"{first_word} {second_word}"
            return first_word
        return first_word if not env_prefix else " ".join(env_prefix) + " " + first_word

    except ValueError:
        parts = command.split()
        if not parts:
            return "none"
        cmd_parts = _strip_env_assignments(parts)
        return cmd_parts[0] if cmd_parts else "none"


def extract_filepaths_from_command(command: str, output: str) -> str:
    """Extract file paths from a command locally without API call.

    Returns the filepath extraction result in ``<filepaths>`` XML format.
    """
    listing_commands = {
        "ls", "dir", "find", "tree", "pwd", "cd", "mkdir", "rmdir", "rm",
    }
    reading_commands = {"cat", "head", "tail", "less", "more", "bat", "type"}

    try:
        parts = shlex.split(command, posix=False)
        if not parts:
            return "<filepaths>\n</filepaths>"

        cmd_parts = _strip_env_assignments(parts)
        if not cmd_parts:
            return "<filepaths>\n</filepaths>"

        base_cmd = cmd_parts[0].split("/")[-1].split("\\")[-1].lower()

        if base_cmd in listing_commands:
            return "<filepaths>\n</filepaths>"

        if base_cmd in reading_commands:
            filepaths = []
            for part in cmd_parts[1:]:
                if part.startswith("-"):
                    continue
                filepaths.append(part)
            if filepaths:
                return f"<filepaths>\n{"\n".join(filepaths)}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        if base_cmd == "grep":
            flags_with_args = {"-e", "-f", "-m", "-A", "-B", "-C"}
            pattern_provided_via_flag = False
            positional = []

            skip_next = False
            for part in cmd_parts[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if part.startswith("-"):
                    if part in flags_with_args:
                        if part in {"-e", "-f"}:
                            pattern_provided_via_flag = True
                        skip_next = True
                    continue
                positional.append(part)

            filepaths = positional if pattern_provided_via_flag else positional[1:]
            if filepaths:
                return f"<filepaths>\n{"\n".join(filepaths)}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        return "<filepaths>\n</filepaths>"

    except ValueError:
        return "<filepaths>\n</filepaths>"