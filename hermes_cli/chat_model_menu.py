"""Share Kiroku's saved chat menu with Hermes slash-command pickers.

Kiroku (the Hermes web chat) stores the operator's curated model menu in
``<HERMES_HOME>/webchat-settings.json`` as exact ``provider/model`` pairs. When that file exists, the
CLI and gateway ``/model`` pickers show the same curated selection instead of the full catalog.
Discovery itself is untouched: only the menu is narrowed.
"""

import json
from pathlib import Path

from hermes_constants import get_hermes_home

SETTINGS_FILENAME = "webchat-settings.json"
# Kiroku excludes these catalogs from its menu even when its allowlist means "all".
_EXCLUDED_MENU_PROVIDERS = frozenset({"ai-gateway", "copilot"})


def load_chat_model_menu(home: Path | None = None) -> dict:
    """Read on every menu open so admin changes need no gateway restart; ``{}`` = no shared menu."""
    path = (home if home is not None else get_hermes_home()) / SETTINGS_FILENAME
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(settings, dict):
        return {}
    return {
        key: {value for value in settings[key] if isinstance(value, str)}
        for key in ("model_allowlist", "model_blocklist")
        if isinstance(settings.get(key), list)
    }


def filter_chat_model_menu(rows: list[dict], settings: dict) -> list[dict]:
    """Keep exact ``provider/model`` pairs (empty allowlist = all) minus the blocklist; rows are not mutated."""
    if not settings:
        return rows
    allow = settings.get("model_allowlist", set())
    block = settings.get("model_blocklist", set())
    result = []
    for row in rows:
        if row["slug"] in _EXCLUDED_MENU_PROVIDERS:
            continue
        models = [
            model for model in row.get("models", [])
            if (not allow or f"{row['slug']}/{model}" in allow) and f"{row['slug']}/{model}" not in block
        ]
        if models:
            result.append({**row, "models": models, "total_models": len(models)})
    return result


def list_chat_model_providers(settings: dict, *, current_provider: str = "", refresh: bool = False) -> list[dict]:
    """Authenticated provider rows from the same disk-config inventory as Kiroku's model-options
    endpoint, narrowed to the shared menu. Session overrides must not inject extra models."""
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    payload = build_model_options_payload(load_picker_context(), include_unconfigured=True, refresh=refresh)
    rows = [
        {**row, "is_current": row["slug"] == current_provider}
        for row in payload["providers"] if row.get("authenticated")
    ]
    return filter_chat_model_menu(rows, settings)


def model_menu_command(provider: str, model: str) -> str:
    """The ``/model`` command for one menu entry: the shortest configured alias that targets exactly
    this provider's model on its default endpoint, else an explicit ``--provider`` command."""
    from hermes_cli.model_switch import DIRECT_ALIASES, _ensure_direct_aliases

    _ensure_direct_aliases()
    aliases = [
        name for name, target in DIRECT_ALIASES.items()
        if target.provider == provider and target.model == model and not target.base_url
    ]
    if aliases:
        return f"/model {min(aliases, key=lambda name: (len(name), name))}"
    return f"/model {model} --provider {provider}"
