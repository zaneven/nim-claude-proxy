"""Request detection utilities for API fast-path optimizations.

Detects quota checks, title generation, prefix detection, safety classifier,
suggestion mode, and filepath extraction requests to enable targeted handling
without calling the upstream NIM API.
"""

from nim_proxy.fcc_port.core.anthropic import (
    Message,
    MessagesRequest,
    extract_text_from_content,
)


def _single_user_turn(request_data: MessagesRequest) -> Message | None:
    """Return the sole conversational user turn, ignoring system context."""
    user_turn: Message | None = None
    for message in request_data.messages:
        if message.role == "system":
            continue
        if message.role != "user" or user_turn is not None:
            return None
        user_turn = message
    return user_turn


def _request_system_text(request_data: MessagesRequest) -> str:
    """Return top-level and inline system text for request-shape detection."""
    parts: list[str] = []
    if request_data.system:
        text = extract_text_from_content(request_data.system)
        if text:
            parts.append(text)
    for message in request_data.messages:
        if message.role != "system":
            continue
        text = extract_text_from_content(message.content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def is_quota_check_request(request_data: MessagesRequest) -> bool:
    """Check if this is a quota probe request (max_tokens=1, "quota" keyword)."""
    message = _single_user_turn(request_data)
    if request_data.max_tokens != 1 or message is None:
        return False
    text = extract_text_from_content(message.content)
    return "quota" in text.lower()


def is_title_generation_request(request_data: MessagesRequest) -> bool:
    """Check if this is a conversation title generation request.

    Matches Claude Code session title prompts (sentence-case title,
    JSON "title" field, etc.).
    """
    if request_data.tools:
        return False
    system_text = _request_system_text(request_data).lower()
    if "title" not in system_text:
        return False
    return "sentence-case title" in system_text or (
        "return json" in system_text
        and "field" in system_text
        and ("coding session" in system_text or "this session" in system_text)
    )


def is_prefix_detection_request(
    request_data: MessagesRequest,
) -> tuple[bool, str]:
    """Check if this is a fast prefix detection request (policy_spec + Command:)."""
    message = _single_user_turn(request_data)
    if message is None:
        return False, ""

    content = extract_text_from_content(message.content)
    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except TypeError:
            return False, ""

    return False, ""


def is_safety_classifier_request(request_data: MessagesRequest) -> bool:
    """Return whether this is Claude Code's auto-mode safety classifier prompt."""
    if request_data.tools:
        return False

    system_text = (
        extract_text_from_content(request_data.system) if request_data.system else ""
    )
    messages_text = "".join(
        extract_text_from_content(message.content)
        for message in request_data.messages
    )
    combined = f"{system_text}\n{messages_text}"
    has_verdict_instruction = "yes</block>" in combined or "no</block>" in combined
    return "<transcript>" in combined and has_verdict_instruction


def is_suggestion_mode_request(request_data: MessagesRequest) -> bool:
    """Check if this is a suggestion mode request (``[SUGGESTION MODE:`` marker)."""
    for msg in request_data.messages:
        if msg.role == "user":
            text = extract_text_from_content(msg.content)
            if "[SUGGESTION MODE:" in text:
                return True
    return False


def is_filepath_extraction_request(
    request_data: MessagesRequest,
) -> tuple[bool, str, str]:
    """Check if this is a filepath extraction request (Command: + Output: blocks).

    Returns (is_filepath_request, command, output).
    """
    message = _single_user_turn(request_data)
    if message is None:
        return False, "", ""
    if request_data.tools:
        return False, "", ""

    content = extract_text_from_content(message.content)
    if "Command:" not in content or "Output:" not in content:
        return False, "", ""

    user_has_filepaths = (
        "filepaths" in content.lower() or "<filepaths>" in content.lower()
    )
    system_text = _request_system_text(request_data)
    system_has_extract = (
        "extract any file paths" in system_text.lower()
        or "file paths that this command" in system_text.lower()
    )
    if not user_has_filepaths and not system_has_extract:
        return False, "", ""

    cmd_start = content.find("Command:") + len("Command:")
    output_marker = content.find("Output:", cmd_start)
    if output_marker == -1:
        return False, "", ""

    command = content[cmd_start:output_marker].strip()
    output = content[output_marker + len("Output:"):].strip()

    for marker in ("<", "\n\n"):
        if marker in output:
            output = output.split(marker)[0].strip()

    return True, command, output