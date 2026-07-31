"""Anthropic Protocol Models & Converters for NIM Proxy."""

import json
from typing import Any, Literal
from pydantic import BaseModel
from nim_proxy.config.settings import settings
from nim_proxy.provider.schema import sanitize_nim_tools


class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: ContentBlockImageSource


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[Any] | None = None
    is_error: bool = False


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str | None = None
    signature: str | None = None


class ContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str | None = None


ContentBlock = (
    ContentBlockText
    | ContentBlockImage
    | ContentBlockToolUse
    | ContentBlockToolResult
    | ContentBlockThinking
    | ContentBlockRedactedThinking
    | dict[str, Any]
)


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ThinkingConfig(BaseModel):
    type: str | None = None
    budget_tokens: int | None = None


class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | list[ContentBlockText] | None = None
    max_tokens: int = 4096
    temperature: float | None = 1.0
    top_p: float | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None


def anthropic_request_to_openai_body(req: MessagesRequest) -> dict[str, Any]:
    """Convert Anthropic MessagesRequest to OpenAI Chat Completions payload with tool schema sanitization."""
    messages_out = []

    # 1. System Prompt 处理
    if req.system:
        system_text = ""
        if isinstance(req.system, str):
            system_text = req.system
        elif isinstance(req.system, list):
            system_text = "\n".join(b.text for b in req.system if getattr(b, "type", None) == "text")
        if system_text:
            messages_out.append({"role": "system", "content": system_text})

    # 2. Message 转换
    for msg in req.messages:
        role = msg.role
        if isinstance(msg.content, str):
            messages_out.append({"role": role, "content": msg.content})
        elif isinstance(msg.content, list):
            text_parts = []
            tool_calls = []

            for block in msg.content:
                b_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                
                if b_type == "text":
                    text_parts.append(block.text if hasattr(block, "text") else block["text"])
                elif b_type == "thinking":
                    thinking_content = block.thinking if hasattr(block, "thinking") else block.get("thinking", "")
                    if thinking_content:
                        text_parts.append(f"<think>\n{thinking_content}\n</think>")
                elif b_type == "redacted_thinking":
                    pass  # Ignore opaque continuation data
                elif b_type == "tool_use":
                    b_id = block.id if hasattr(block, "id") else block["id"]
                    b_name = block.name if hasattr(block, "name") else block["name"]
                    b_input = block.input if hasattr(block, "input") else block["input"]
                    tool_calls.append({
                        "id": b_id,
                        "type": "function",
                        "function": {
                            "name": b_name,
                            "arguments": json.dumps(b_input, ensure_ascii=False),
                        }
                    })
                elif b_type == "tool_result":
                    tr_id = block.tool_use_id if hasattr(block, "tool_use_id") else block["tool_use_id"]
                    tr_content = block.content if hasattr(block, "content") else block.get("content")
                    if isinstance(tr_content, list):
                        tr_text = "\n".join(str(c) for c in tr_content)
                    else:
                        tr_text = str(tr_content or "")
                    messages_out.append({
                        "role": "tool",
                        "tool_call_id": tr_id,
                        "content": tr_text,
                    })

            if text_parts or tool_calls:
                msg_obj: dict[str, Any] = {
                    "role": role,
                    "content": "\n".join(text_parts) if text_parts else None
                }
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                messages_out.append(msg_obj)

    # 3. Tools 转换与 Schema 格式洗净
    tools_out = None
    if req.tools:
        raw_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                }
            }
            for t in req.tools
        ]
        tools_out, _ = sanitize_nim_tools(raw_tools)

    # 智能别名解析与模型选取
    model_name = settings.get_model_for_alias(req.model)

    body = {
        "model": model_name,
        "messages": messages_out,
        "max_tokens": req.max_tokens,
        "stream": req.stream,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if tools_out:
        body["tools"] = tools_out

    # 求解计算客户端与特定路由命中的 Reasoning 策略
    client_budget = req.thinking.budget_tokens if req.thinking else None
    resolved_reasoning = settings.get_reasoning_for_model(req.model, client_budget=client_budget)

    if resolved_reasoning.enabled and resolved_reasoning.budget_tokens:
        body["extra_body"] = {"reasoning_budget": resolved_reasoning.budget_tokens}

    return body
