"""NVIDIA NIM Tool Schema Sanitization, Boolean Node Removal, and Private Parameter Aliasing."""

import json
import copy
from typing import Any

_UNSAFE_PARAMETER_NAMES = frozenset({"type"})
_ALIAS_PREFIX = "_nim_arg_"

_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "items",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
    }
)
_SCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)


def _sanitize_schema_node(value: Any) -> tuple[bool, Any]:
    """Remove boolean JSON Schema subschemas that hosted NIM nodes reject."""
    if isinstance(value, bool):
        return False, None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SCHEMA_VALUE_KEYS:
                keep, sanitized_item = _sanitize_schema_node(item)
                if keep:
                    sanitized[key] = sanitized_item
            elif key in _SCHEMA_LIST_KEYS and isinstance(item, list):
                sanitized_items: list[Any] = []
                for schema_item in item:
                    keep, sanitized_item = _sanitize_schema_node(schema_item)
                    if keep:
                        sanitized_items.append(sanitized_item)
                if sanitized_items:
                    sanitized[key] = sanitized_items
            elif key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
                sanitized_map: dict[str, Any] = {}
                for map_key, schema_item in item.items():
                    keep, sanitized_item = _sanitize_schema_node(schema_item)
                    if keep:
                        sanitized_map[map_key] = sanitized_item
                sanitized[key] = sanitized_map
            else:
                sanitized[key] = item
        return True, sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            keep, sanitized_item = _sanitize_schema_node(item)
            if keep:
                sanitized_items.append(sanitized_item)
        return True, sanitized_items
    return True, value


def alias_nim_tool_parameters(parameters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Convert unsafe parameter names (like 'type') to '_nim_arg_type'."""
    if not isinstance(parameters, dict):
        return parameters, {}

    alias_map: dict[str, str] = {}  # alias -> original
    original_to_alias: dict[str, str] = {}  # original -> alias

    properties = parameters.get("properties")
    if isinstance(properties, dict):
        new_properties: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            if prop_name in _UNSAFE_PARAMETER_NAMES:
                alias = f"{_ALIAS_PREFIX}{prop_name}"
                alias_map[alias] = prop_name
                original_to_alias[prop_name] = alias
                new_properties[alias] = prop_schema
            else:
                new_properties[prop_name] = prop_schema
        parameters["properties"] = new_properties

    # 替换 required 列表中的参数名称
    required = parameters.get("required")
    if isinstance(required, list):
        new_required = []
        for req_name in required:
            if req_name in original_to_alias:
                new_required.append(original_to_alias[req_name])
            else:
                new_required.append(req_name)
        parameters["required"] = new_required

    return parameters, alias_map


def sanitize_nim_tools(tools: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]] | None, dict[str, dict[str, str]]]:
    """Sanitize tools schema and return aliased argument mapping."""
    if not tools or not isinstance(tools, list):
        return None, {}

    cleaned_tools = []
    global_aliases: dict[str, dict[str, str]] = {}

    for tool in tools:
        if not isinstance(tool, dict):
            cleaned_tools.append(tool)
            continue

        item = copy.deepcopy(tool)
        fn = item.get("function")
        if isinstance(fn, dict):
            fn.pop("nim_alias", None)
            params = fn.get("parameters")
            if isinstance(params, dict):
                params.pop("$schema", None)
                _, sanitized_params = _sanitize_schema_node(params)
                aliased_params, alias_map = alias_nim_tool_parameters(sanitized_params)
                fn["parameters"] = aliased_params
                tool_name = fn.get("name")
                if alias_map and isinstance(tool_name, str):
                    global_aliases[tool_name] = alias_map

        cleaned_tools.append(item)

    return cleaned_tools, global_aliases


def unalias_tool_call_arguments(args_json_or_dict: Any) -> Any:
    """Restore aliased argument names (e.g. '_nim_arg_type' -> 'type') in model outputs."""
    if isinstance(args_json_or_dict, str):
        try:
            parsed = json.loads(args_json_or_dict)
            restored = unalias_tool_call_arguments(parsed)
            return json.dumps(restored, ensure_ascii=False)
        except Exception:
            # 如果 JSON 暂不完整或无法解析，替换常规别名模式
            return args_json_or_dict.replace(f'"{_ALIAS_PREFIX}type"', '"type"')

    if isinstance(args_json_or_dict, dict):
        restored_dict = {}
        for k, v in args_json_or_dict.items():
            original_key = k
            if isinstance(k, str) and k.startswith(_ALIAS_PREFIX):
                original_key = k[len(_ALIAS_PREFIX):]
            restored_dict[original_key] = unalias_tool_call_arguments(v)
        return restored_dict

    if isinstance(args_json_or_dict, list):
        return [unalias_tool_call_arguments(item) for item in args_json_or_dict]

    return args_json_or_dict
