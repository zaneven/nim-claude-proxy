"""Tests for the server-side web tool agentic loop."""

import asyncio
from types import SimpleNamespace

from nim_proxy.core import agent_loop
from nim_proxy.core.agent_loop import run_with_web_tools


class FakeClient:
    def __init__(self):
        self.calls = 0

    async def execute_chat(self, body):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="web_search", arguments='{"query":"news"}'),
                )],
            ))])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Here are the results.", tool_calls=None,
        ))])


class FakeClientNonWeb:
    def __init__(self):
        self.calls = 0

    async def execute_chat(self, body):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="call_2",
                function=SimpleNamespace(name="edit_file", arguments='{"path":"a"}'),
            )],
        ))])


def test_run_with_web_tools_resolves_locally():
    agent_loop.handle_local_web_tool = lambda name, inp: f"result for {inp.get('query')}"
    client = FakeClient()
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{"type": "function", "function": {"name": "web_search", "parameters": {}}}],
    }
    resp = asyncio.run(run_with_web_tools(client, body))

    # 应当循环两次（先 web 工具，再最终文本）
    assert client.calls == 2
    assert resp.choices[0].message.content == "Here are the results."
    # 本地工具结果应当被回填进对话
    assert any(m.get("role") == "tool" for m in body["messages"])


def test_run_with_web_tools_aborts_on_non_web():
    agent_loop.handle_local_web_tool = lambda name, inp: "x"
    client = FakeClientNonWeb()
    body = {"model": "x", "messages": [{"role": "user", "content": "go"}]}
    resp = asyncio.run(run_with_web_tools(client, body))

    # 含无法本地处理的工具调用 -> 原样返回，不循环
    assert client.calls == 1
    assert resp.choices[0].message.tool_calls[0].function.name == "edit_file"
