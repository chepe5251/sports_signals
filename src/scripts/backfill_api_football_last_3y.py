from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.api.api_football import get_fixtures_by_season
from src.database.connection import get_connection
from src.utils.leagues import LEAGUE_MAP


def _season_for_today() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def _clear_core_tables(cur) -> None:
    cur.execute("DELETE FROM core.MatchStats")
    cur.execute("DELETE FROM core.Odds")
    cur.execute("DELETE FROM core.Matches")
    cur.execute("DELETE FROM core.Teams")
    cur.execute("DELETE FROM core.Leagues")


def _ensure_league(cur, api_league_id: int, name: str) -> int:
    cur.execute("SELECT league_id FROM core.Leagues WHERE api_league_id = ?", api_league_id)
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        """
        INSERT INTO core.Leagues (name, country, is_active, api_league_id)
        OUTPUT INSERTED.league_id
        VALUES (?, NULL, 1, ?)
        """,
        name, api_league_id,
    )
    return int(cur.fetchone()[0])


def _ensure_team(cur, league_id: int, api_team_id: int, name: str) -> int:
    cur.execute("SELECT team_id FROM core.Teams WHERE api_team_id = ?", api_team_id)
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        """
        INSERT INTO core.Teams (league_id, name, api_team_id)
        OUTPUT INSERTED.team_id
        VALUES (?, ?, ?)
        """,
        league_id, name, api_team_id,
    )
    return int(cur.fetchone()[0])


def _normalize_kickoff(raw: str, season: int) -> str:
    if not raw:
        return f"{season}-08-01 12:00:00"
    value = raw.replace("Z", "+00:00")
    try:
        dt = pd.to_datetime(value, utc=True)
        return dt.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            dt = pd.to_datetime(value)
            if dt.time().hour == 0 and dt.time().minute == 0 and dt.time().second == 0:
                dt = dt.replace(hour=12)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return f"{season}-08-01 12:00:00"


def main(reset: bool) -> None:
    latest_season = _season_for_today()
    seasons = [latest_season - 2, latest_season - 1, latest_season]

    conn = get_connection()
    cur = conn.cursor()

    if reset:
        print("Borrando tablas core...")
        _clear_core_tables(cur)
        conn.commit()

    total = 0

    for api_league_id, league_name in LEAGUE_MAP.items():
        league_id = _ensure_league(cur, api_league_id, league_name)
        for season in seasons:
            print(f"Cargando {league_name} season {season}...")
            fixtures = get_fixtures_by_season(api_league_id, season, status=None)
            for f in fixtures:
                fixture = f.get("fixture") or {}
                teams = f.get("teams") or {}
                goals = f.get("goals") or {}
                status = (fixture.get("status") or {}).get("short") or ""

                api_fixture_id = fixture.get("id")
                match_date = _normalize_kickoff(fixture.get("date") or "", season)
                home = teams.get("home") or {}
                away = teams.get("away") or {}

                home_id_api = home.get("id")
                away_id_api = away.get("id")
                home_name = home.get("name")
                away_name = away.get("name")

                if not api_fixture_id or not home_id_api or not away_id_api:
                    continue

                match_status = "finished" if status in {"FT", "AET", "PEN"} else "scheduled"
                home_goals = goals.get("home")
                away_goals = goals.get("away")

                home_id = _ensure_team(cur, league_id, int(home_id_api), str(home_name))
                away_id = _ensure_team(cur, league_id, int(away_id_api), str(away_name))

                cur.execute(
                    """
                    IF NOT EXISTS (SELECT 1 FROM core.Matches WHERE api_fixture_id = ?)
                    INSERT INTO core.Matches (
                        league_id, home_team_id, away_team_id, match_date,
                        status, home_goals, away_goals, api_fixture_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    api_fixture_id,
                    league_id, home_id, away_id, match_date,
                    match_status, home_goals, away_goals, api_fixture_id,
                )
                total += 1

            conn.commit()

    conn.close()
    print(f"Listo. Total partidos insertados: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Borra tablas core antes de recargar datos")
    args = parser.parse_args()
    main(reset=args.reset)
