"""Share Kiroku's saved chat menu with Hermes slash-command pickers."""

import json
from pathlib import Path

from hermes_constants import get_hermes_home


def load_chat_model_menu(home: Path | None = None) -> dict:
    """Read on every menu open so admin changes require no gateway restart."""
    path = (home if home is not None else get_hermes_home()) / "webchat-settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(settings, dict):
        return {}
    return {
        key: {value for value in settings.get(key, []) if isinstance(value, str)}
        for key in ("model_allowlist", "model_blocklist")
        if isinstance(settings.get(key), list)
    }


def filter_chat_model_menu(rows: list[dict], settings: dict) -> list[dict]:
    """Filter exact provider/model pairs, preserving the full discovery catalog."""
    allow = settings.get("model_allowlist", set())
    block = settings.get("model_blocklist", set())
    if not settings:
        return rows
    result = []
    for row in rows:
        # Kiroku excludes these catalogs even when its allowlist means "all".
        if row["slug"] in {"ai-gateway", "copilot"}:
            continue
        models = [
            model for model in row.get("models", [])
            if (not allow or f"{row['slug']}/{model}" in allow)
            and f"{row['slug']}/{model}" not in block
        ]
        if models:
            result.append({**row, "models": models, "total_models": len(models)})
    return result


def list_chat_model_providers(settings: dict, *, current_provider: str = "", refresh: bool = False) -> list[dict]:
    """Use the same disk-config inventory as Kiroku's model-options endpoint.

    Session overrides must not inject additional models into the shared menu.
    """
    from hermes_cli.inventory import build_model_options_payload, load_picker_context

    payload = build_model_options_payload(
        load_picker_context(), include_unconfigured=True, refresh=refresh,
    )
    rows = [
        {**row, "is_current": row["slug"] == current_provider}
        for row in payload["providers"] if row.get("authenticated")
    ]
    return filter_chat_model_menu(rows, settings)


def model_menu_command(provider: str, model: str) -> str:
    """Offer a configured exact alias, or an explicit provider-safe command."""
    from hermes_cli.model_switch import DIRECT_ALIASES, _ensure_direct_aliases

    _ensure_direct_aliases()
    aliases = [
        name for name, target in DIRECT_ALIASES.items()
        if target.provider == provider and target.model == model
        and not target.base_url
    ]
    if aliases:
        alias = min(aliases, key=lambda name: (len(name), name))
        return f"/model {alias}"
    return f"/model {model} --provider {provider}"
