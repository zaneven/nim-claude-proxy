"""Tests for Schema Sanitization, Boolean Removal, and Aliased Argument Unaliasing."""

from nim_proxy.provider.schema import sanitize_nim_tools, unalias_tool_call_arguments


def test_sanitize_boolean_schema_nodes_and_type_aliasing():
    raw_tools = [
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "description": "file type"},
                        "content": {"type": "string"}
                    },
                    "required": ["type", "content"]
                }
            }
        }
    ]

    cleaned, aliases = sanitize_nim_tools(raw_tools)
    assert cleaned is not None
    fn_params = cleaned[0]["function"]["parameters"]

    # 1. 校验 additionalProperties: False 是否被净化剔除
    assert "additionalProperties" not in fn_params

    # 2. 校验 unsafe 参数 'type' 是否被重命名别名为 '_nim_arg_type'
    props = fn_params["properties"]
    assert "_nim_arg_type" in props
    assert "type" not in props
    assert fn_params["required"] == ["_nim_arg_type", "content"]


def test_unalias_tool_call_arguments():
    # 模拟模型输出了带 _nim_arg_type 的参数字典
    model_output_args = {
        "_nim_arg_type": "python",
        "content": "print('hello')",
    }

    restored = unalias_tool_call_arguments(model_output_args)
    assert "type" in restored
    assert restored["type"] == "python"
    assert restored["content"] == "print('hello')"
    assert "_nim_arg_type" not in restored


import pytest
from unittest.mock import AsyncMock, patch
from nim_proxy.provider.client import NimClient
from nim_proxy.provider.key_pool import KeyPoolManager


@pytest.mark.asyncio
async def test_client_execute_chat_keeps_tools_as_list():
    key_pool = KeyPoolManager(["nvapi-testkey"])
    client = NimClient(key_pool)

    raw_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}
                }
            }
        ]
    }

    mock_chat = AsyncMock()
    mock_chat.completions.create = AsyncMock(return_value=AsyncMock(choices=[]))

    with patch.object(client, "_get_async_client") as mock_get_client:
        mock_openai_client = AsyncMock()
        mock_openai_client.chat = mock_chat
        mock_get_client.return_value = mock_openai_client

        await client.execute_chat(raw_body)

        create_kwargs = mock_chat.completions.create.call_args.kwargs
        assert isinstance(create_kwargs["tools"], list)
        assert not isinstance(create_kwargs["tools"], tuple)

