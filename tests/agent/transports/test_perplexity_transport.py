"""Perplexity native search must coexist with Hermes client-side tools."""

import copy

import pytest

from agent.transports import get_transport


def function(name):
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {}},
    }}


def build(url, tools):
    return get_transport("codex_responses").build_kwargs(
        model="example/model", base_url=url,
        messages=[{"role": "user", "content": "test"}], tools=tools,
    )


def test_perplexity_native_search_preserves_other_tools_and_shared_schema():
    tools = [function("terminal"), function("web_search"), function("web_extract")]
    original = copy.deepcopy(tools)
    payload = build("https://api.perplexity.ai/v1", tools)
    assert payload["tools"][1] == {"type": "web_search"}
    assert [t["name"] for t in payload["tools"] if t["type"] == "function"] == [
        "hermes_terminal", "hermes_web_extract",
    ]
    assert tools == original
    assert payload["store"] is False


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1",
    "https://api.perplexity.ai.example.com/v1",
    "https://example.com/api.perplexity.ai/v1",
])
def test_native_search_swap_is_scoped_to_perplexity_host(url):
    payload = build(url, [function("web_search")])
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "web_search"


@pytest.mark.parametrize("tools", [[], [function("terminal")]])
def test_search_is_not_enabled_without_granted_capability(tools):
    payload = build("https://api.perplexity.ai/v1", tools)
    assert not any(t["type"] == "web_search" for t in payload.get("tools", []))


def test_client_tool_aliases_round_trip_and_replay_without_mutating_history():
    from types import SimpleNamespace

    transport = get_transport("codex_responses")
    tools = [function("search_files")]
    transport.build_kwargs(model="example/model", base_url="https://api.perplexity.ai/v1",
                           messages=[{"role": "user", "content": "find files"}], tools=tools)
    response = SimpleNamespace(status="completed", usage=None, output=[SimpleNamespace(
        type="function_call", id="fc_test", call_id="call_test",
        name="hermes_search_files", arguments='{"pattern":"test"}', status="completed",
    )])
    normalized = transport.normalize_response(response)
    assert normalized.tool_calls[0].name == "search_files"
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_test", "type": "function", "function": {
                "name": "search_files", "arguments": '{"pattern":"test"}',
            },
        }]},
        {"role": "tool", "tool_call_id": "call_test", "content": "found"},
    ]
    original = copy.deepcopy(messages)
    kwargs = transport.build_kwargs(model="example/model", base_url="https://api.perplexity.ai/v1",
                                    messages=messages, tools=tools)
    call = next(x for x in kwargs["input"] if x["type"] == "function_call")
    assert call["name"] == "hermes_search_files" and call["call_id"] == "call_test"
    assert messages == original
    other = transport.build_kwargs(model="example/model", base_url="https://api.openai.com/v1",
                                    messages=messages, tools=tools)
    assert other["tools"][0]["name"] == "search_files"
    # Upstream keeps the request-local alias map in ``_last_wire_aliases``.
    assert not transport._last_wire_aliases


def test_long_tool_names_get_distinct_bounded_aliases():
    names = ["tool_" + "x" * 59 + suffix for suffix in ("a", "b")]
    payload = build("https://api.perplexity.ai/v1", [function(n) for n in names])
    wire_names = [t["name"] for t in payload["tools"]]
    assert len(set(wire_names)) == 2 and all(len(n) <= 64 for n in wire_names)


@pytest.mark.parametrize("url,limit", [
    ("https://api.perplexity.ai/v1", 64),
    ("https://chatgpt.com/backend-api/codex", None),
])
def test_auxiliary_limit_is_forwarded_only_to_perplexity(url, limit):
    from types import SimpleNamespace
    from agent.auxiliary_client import _CodexCompletionsAdapter

    item = SimpleNamespace(type="message", role="assistant", status="completed",
                           content=[SimpleNamespace(type="output_text", text="Title")])
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_item.done", item=item),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(
            status="completed", id="resp_test", usage=None,
        )),
    ]

    class Stream:
        def __iter__(self):
            return iter(events)

        def close(self):
            return None

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return Stream()

    client = SimpleNamespace(base_url=url, responses=SimpleNamespace(create=create))
    response = _CodexCompletionsAdapter(client, "example/model").create(
        messages=[{"role": "user", "content": "Title this"}], max_tokens=64,
    )
    assert response.choices[0].message.content == "Title"
    assert captured.get("max_output_tokens") == limit
