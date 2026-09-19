"""Kiroku menu preferences narrow Hermes menus without narrowing discovery."""

import copy
import json
from types import SimpleNamespace

import pytest

from hermes_cli.chat_model_menu import filter_chat_model_menu, load_chat_model_menu


@pytest.fixture
def home(tmp_path, monkeypatch):
    import gateway.run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway.run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway.run, "_load_gateway_config", lambda: {
        "model": {"default": "selected", "provider": "one"},
    })
    return tmp_path


def save(home, allow, block=()):
    (home / "webchat-settings.json").write_text(json.dumps({
        "model_allowlist": list(allow), "model_blocklist": list(block),
    }))


def rows():
    return [
        {"slug": "one", "name": "One", "is_current": True, "authenticated": True,
         "models": [*[f"other-{i}" for i in range(60)], "selected"], "total_models": 61},
        {"slug": "two", "name": "Two", "is_current": False, "authenticated": True,
         "models": ["selected", "namespace/model"], "total_models": 2},
    ]


def test_exact_provider_pairs_and_blocklist_preserve_catalog(home):
    catalog = rows()
    before = copy.deepcopy(catalog)
    save(home, ["one/selected", "two/namespace/model"], ["one/selected"])
    filtered = filter_chat_model_menu(catalog, load_chat_model_menu())
    assert [(r["slug"], r["models"], r["total_models"]) for r in filtered] == [
        ("two", ["namespace/model"], 1),
    ]
    assert catalog == before


def test_empty_allowlist_shows_all_except_removed(home):
    save(home, [], ["two/selected"])
    filtered = filter_chat_model_menu(rows(), load_chat_model_menu())
    assert filtered[0]["models"] == rows()[0]["models"]
    assert filtered[1]["models"] == ["namespace/model"]


def test_shared_menu_hides_kiroku_excluded_providers_without_changing_discovery(home):
    catalog = rows() + [
        {"slug": "ai-gateway", "models": ["extra"]},
        {"slug": "copilot", "models": ["extra"]},
    ]
    assert filter_chat_model_menu(catalog, {}) == catalog
    save(home, [])
    assert filter_chat_model_menu(catalog, load_chat_model_menu()) == rows()
    assert len(catalog) == 4


@pytest.mark.parametrize("content", [None, "{", "[]", '{"model_allowlist": null}'])
def test_missing_or_malformed_preferences_leave_menu_usable(home, content):
    if content is not None:
        (home / "webchat-settings.json").write_text(content)
    catalog = rows()
    assert filter_chat_model_menu(catalog, load_chat_model_menu()) == catalog


def test_preferences_are_profile_scoped_and_reloaded(home, tmp_path):
    save(home, ["one/selected"])
    other = home / "profile"
    other.mkdir()
    save(other, ["two/selected"])
    assert load_chat_model_menu()["model_allowlist"] == {"one/selected"}
    assert load_chat_model_menu(other)["model_allowlist"] == {"two/selected"}
    save(home, ["two/namespace/model"])
    assert load_chat_model_menu()["model_allowlist"] == {"two/namespace/model"}


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.asyncio
async def test_gateway_model_menu_uses_live_selection_before_catalog_cap(home, monkeypatch, interactive):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    save(home, ["one/selected"])
    catalog = rows()
    def inventory(ctx, **kwargs):
        limit = kwargs.get("max_models")
        return {"providers": [{**r, "models": r["models"][:limit]} for r in catalog]}

    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", inventory)
    class Adapter:
        async def send_model_picker(self, **kwargs):
            self.providers = kwargs["providers"]
            return SimpleNamespace(success=True)

    adapter = Adapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter} if interactive else {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._voice_mode = {}
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *a: None)
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda *a: None)
    event = MessageEvent(text="/model", message_type=MessageType.TEXT,
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="test", chat_type="dm"))
    result = await runner._handle_model_command(event)
    if interactive:
        assert result is None
        assert [(r["slug"], r["models"]) for r in adapter.providers] == [("one", ["selected"])]
    else:
        assert "**One**" in result and "`selected`" in result
        assert "**Two**" not in result and "other-" not in result
    save(home, ["two/namespace/model"])
    result = await runner._handle_model_command(event)
    if interactive:
        assert [(r["slug"], r["models"]) for r in adapter.providers] == [("two", ["namespace/model"])]
    else:
        assert "**Two**" in result and "`namespace/model`" in result
        assert "**One**" not in result


def test_terminal_model_menu_uses_shared_selection(home, monkeypatch):
    from cli import HermesCLI
    from hermes_cli.inventory import ConfigContext

    save(home, ["two/namespace/model"])
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: ConfigContext("one", "selected", "", {}, []))
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", lambda *a, **k: {"providers": rows()})
    captured = []
    shell = SimpleNamespace(provider="one", model="selected", base_url="",
                            _open_model_picker=lambda providers, *a, **k: captured.extend(providers))
    HermesCLI._handle_model_switch(shell, "/model")
    assert [(r["slug"], r["models"]) for r in captured] == [("two", ["namespace/model"])]
