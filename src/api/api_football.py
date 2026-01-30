from __future__ import annotations

import time
import requests

from src.utils.config import API_FOOTBALL_KEY

BASE_URL = "https://v3.football.api-sports.io"

_session = requests.Session()
if API_FOOTBALL_KEY:
    _session.headers.update({"x-apisports-key": API_FOOTBALL_KEY})


def _get(endpoint: str, params: dict | None = None, max_retries: int = 3):
    url = f"{BASE_URL}{endpoint}"
    attempt = 0
    while True:
        r = _session.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        errors = payload.get("errors") or {}
        if errors and "rateLimit" in errors and attempt < max_retries:
            attempt += 1
            time.sleep(7)
            continue
        return payload["response"]


def get_fixtures_by_season(api_league_id: int, season: int, status: str | None = None):
    params = {"league": api_league_id, "season": season}
    if status:
        params["status"] = status
    return _get("/fixtures", params)


def get_fixtures_by_date(api_league_id: int, season: int, date_iso: str):
    params = {"league": api_league_id, "season": season, "date": date_iso}
    return _get("/fixtures", params)


def get_odds(fixture_id: int):
    params = {"fixture": fixture_id}
    return _get("/odds", params)
