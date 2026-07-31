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
