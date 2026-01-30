from __future__ import annotations

import requests

from src.utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def _require_token(token: str | None) -> str:
    token = token or TELEGRAM_BOT_TOKEN
    if not token:
        raise ValueError("Falta TELEGRAM_BOT_TOKEN en el entorno.")
    return token


def send_message(text: str, chat_id: str | None = None, token: str | None = None) -> None:
    token = _require_token(token)
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not chat_id:
        raise ValueError("Falta TELEGRAM_CHAT_ID en el entorno.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()


def get_updates(
    offset: int | None = None,
    timeout: int = 25,
    token: str | None = None,
) -> list[dict]:
    token = _require_token(token)
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params: dict[str, object] = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(url, params=params, timeout=timeout + 5)
    r.raise_for_status()
    return r.json().get("result", [])
