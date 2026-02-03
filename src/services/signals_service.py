from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from zoneinfo import ZoneInfo
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.api_gemini import generate_answer, parse_date_range, select_value_picks
from src.api.api_football import get_fixtures_by_date, get_odds
from src.api.api_telegram import send_message
from src.database.connection import get_connection
from src.utils.config import API_FOOTBALL_KEY, APP_TIMEZONE
from src.utils.leagues import LEAGUE_MAP


FEATURES = [
    "home_avg_gf_5",
    "home_avg_ga_5",
    "home_pts_5",
    "home_winrate_5",
    "home_drawrate_5",
    "away_avg_gf_5",
    "away_avg_ga_5",
    "away_pts_5",
    "away_winrate_5",
    "away_drawrate_5",
    "diff_pts_5",
    "diff_avg_gf_5",
    "diff_avg_ga_5",
    "diff_winrate_5",
    "diff_drawrate_5",
]

N_FORM = 10
MODEL_PATH = Path("models/logreg_homewin_v2.pkl")


@dataclass
class MatchPrediction:
    match_id: int
    league: str
    home: str
    away: str
    kickoff: str
    home_win_prob: float
    score: float
    value_edge: float | None
    bet_side: str
    bet_prob: float


def _load_training_dataset() -> pd.DataFrame:
    ds_path = Path("dataset_homewin.csv")
    if not ds_path.exists():
        raise FileNotFoundError("dataset_homewin.csv no existe. Ejecuta build_dataset.py primero.")
    return pd.read_csv(ds_path)


def _train_model() -> Pipeline:
    ds = _load_training_dataset()
    X = ds[FEATURES]
    y = ds["home_win"].astype(int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X, y)
    return model


def _load_or_train_model() -> Pipeline:
    if MODEL_PATH.exists():
        import pickle

        with MODEL_PATH.open("rb") as f:
            return pickle.load(f)

    model = _train_model()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    return model


def _fetch_scheduled_matches(target_date: date) -> list[dict]:
    # Fetch via API-Football and store in DB
    _store_api_football_fixtures([target_date])

    now = _now()
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT
            m.match_id,
            m.match_date,
            m.league_id,
            l.name AS league_name,
            ht.name AS home_name,
            at.name AS away_name,
            m.home_team_id,
            m.away_team_id,
            m.api_fixture_id
        FROM core.Matches m
        JOIN core.Leagues l ON l.league_id = m.league_id
        JOIN core.Teams ht ON ht.team_id = m.home_team_id
        JOIN core.Teams at ON at.team_id = m.away_team_id
        WHERE m.status = 'scheduled'
          AND CAST(m.match_date AS date) = ?
          AND m.match_date >= ?
        ORDER BY m.match_date
        """,
        conn,
        params=[target_date, now],
    )
    conn.close()
    return df.to_dict(orient="records")


def _ensure_league_id(cur, api_league_id: int, league_name: str) -> int:
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
        league_name, api_league_id,
    )
    return int(cur.fetchone()[0])


def _ensure_team_id(cur, league_id: int, api_team_id: int, team_name: str) -> int:
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
        league_id, team_name, api_team_id
    )
    return int(cur.fetchone()[0])


def _normalize_kickoff(raw: str, target_date: date) -> str:
    if not raw:
        return f"{target_date.isoformat()} 12:00:00"
    value = raw.replace("Z", "+00:00")
    if "T" in value:
        try:
            dt = pd.to_datetime(value, utc=True)
            return dt.tz_convert(None).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    try:
        dt = pd.to_datetime(value)
        if dt.time().hour == 0 and dt.time().minute == 0 and dt.time().second == 0:
            dt = dt.replace(hour=12)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return f"{target_date.isoformat()} 12:00:00"


def _season_for_date(target_date: date) -> int:
    # API-Football usa el anio de inicio de temporada (ej: enero 2026 -> season 2025)
    return target_date.year if target_date.month >= 7 else target_date.year - 1


def _store_api_football_fixtures(dates: list[date]) -> None:
    if not dates:
        return

    conn = get_connection()
    cur = conn.cursor()

    for d in dates:
        season = _season_for_date(d)
        for api_league_id, league_name in LEAGUE_MAP.items():
            fixtures = get_fixtures_by_date(api_league_id, season, d.isoformat())
            for f in fixtures:
                fixture = f.get("fixture") or {}
                teams = f.get("teams") or {}
                goals = f.get("goals") or {}
                status = (fixture.get("status") or {}).get("short") or ""

                api_fixture_id = fixture.get("id")
                kickoff = fixture.get("date") or ""
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                home_name = home.get("name")
                away_name = away.get("name")
                home_id_api = home.get("id")
                away_id_api = away.get("id")

                if not api_fixture_id or not home_name or not away_name or not home_id_api or not away_id_api:
                    continue

                match_status = "finished" if status in {"FT", "AET", "PEN"} else "scheduled"
                home_goals = goals.get("home")
                away_goals = goals.get("away")

                league_id = _ensure_league_id(cur, api_league_id, league_name)
                home_id = _ensure_team_id(cur, league_id, int(home_id_api), str(home_name))
                away_id = _ensure_team_id(cur, league_id, int(away_id_api), str(away_name))
                match_date = _normalize_kickoff(str(kickoff), d)

                cur.execute(
                    """
                    IF NOT EXISTS (
                        SELECT 1 FROM core.Matches WHERE api_fixture_id = ?
                    )
                    INSERT INTO core.Matches (
                        league_id, home_team_id, away_team_id, match_date,
                        status, home_goals, away_goals, api_fixture_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    api_fixture_id,
                    league_id, home_id, away_id, match_date,
                    match_status, home_goals, away_goals, api_fixture_id
                )

    conn.commit()
    conn.close()


def _fetch_finished_matches_before(target_date: date) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT
            match_id,
            league_id,
            match_date,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals
        FROM core.Matches
        WHERE status = 'finished'
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
          AND match_date < ?
        """,
        conn,
        params=[target_date],
    )
    conn.close()
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date")


def _build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["match_date", "home_team_id", "home_goals", "away_goals"]].copy()
    home.rename(columns={"home_team_id": "team_id", "home_goals": "gf", "away_goals": "ga"}, inplace=True)

    away = df[["match_date", "away_team_id", "away_goals", "home_goals"]].copy()
    away.rename(columns={"away_team_id": "team_id", "away_goals": "gf", "home_goals": "ga"}, inplace=True)

    long = pd.concat([home, away], ignore_index=True)
    long.sort_values(["team_id", "match_date"], inplace=True)
    long["win"] = (long["gf"] > long["ga"]).astype(int)
    long["draw"] = (long["gf"] == long["ga"]).astype(int)
    return long


def _league_draw_rates(df: pd.DataFrame) -> dict[int, float]:
    if "league_id" not in df.columns or df.empty:
        return {}
    draws = (df["home_goals"] == df["away_goals"]).astype(int)
    rates = (
        df.assign(is_draw=draws)
        .groupby("league_id")["is_draw"]
        .mean()
        .to_dict()
    )
    return {int(k): float(v) for k, v in rates.items()}


def _estimate_draw_prob(
    league_draw_rate: float,
    home_drawrate: float,
    away_drawrate: float,
) -> float:
    base = 0.5 * (home_drawrate + away_drawrate)
    blended = 0.6 * base + 0.4 * league_draw_rate
    return min(max(blended, 0.08), 0.38)


def _compute_form(rows: pd.DataFrame) -> dict | None:
    if len(rows) < N_FORM:
        return None
    recent = rows.tail(N_FORM)
    pts = (3 * recent["win"] + 1 * recent["draw"]).sum()
    return {
        "avg_gf_5": float(recent["gf"].mean()),
        "avg_ga_5": float(recent["ga"].mean()),
        "pts_5": float(pts),
        "winrate_5": float(recent["win"].mean()),
        "drawrate_5": float(recent["draw"].mean()),
    }


def _match_features(
    team_history: pd.DataFrame,
    home_team_id: int,
    away_team_id: int,
    match_date: pd.Timestamp,
) -> dict | None:
    home_rows = team_history[
        (team_history["team_id"] == home_team_id) & (team_history["match_date"] < match_date)
    ]
    away_rows = team_history[
        (team_history["team_id"] == away_team_id) & (team_history["match_date"] < match_date)
    ]

    home_form = _compute_form(home_rows)
    away_form = _compute_form(away_rows)
    if not home_form or not away_form:
        return None

    feats = {
        "home_avg_gf_5": home_form["avg_gf_5"],
        "home_avg_ga_5": home_form["avg_ga_5"],
        "home_pts_5": home_form["pts_5"],
        "home_winrate_5": home_form["winrate_5"],
        "home_drawrate_5": home_form["drawrate_5"],
        "away_avg_gf_5": away_form["avg_gf_5"],
        "away_avg_ga_5": away_form["avg_ga_5"],
        "away_pts_5": away_form["pts_5"],
        "away_winrate_5": away_form["winrate_5"],
        "away_drawrate_5": away_form["drawrate_5"],
    }
    feats["diff_pts_5"] = feats["home_pts_5"] - feats["away_pts_5"]
    feats["diff_avg_gf_5"] = feats["home_avg_gf_5"] - feats["away_avg_gf_5"]
    feats["diff_avg_ga_5"] = feats["home_avg_ga_5"] - feats["away_avg_ga_5"]
    feats["diff_winrate_5"] = feats["home_winrate_5"] - feats["away_winrate_5"]
    feats["diff_drawrate_5"] = feats["home_drawrate_5"] - feats["away_drawrate_5"]
    return feats


def get_predictions_for_date(target_date: date, min_prob: float = 0.6) -> list[MatchPrediction]:
    scheduled = _fetch_scheduled_matches(target_date)
    if not scheduled:
        return []

    finished = _fetch_finished_matches_before(target_date)
    if finished.empty:
        return []

    team_history = _build_team_history(finished)
    league_draw_rates = _league_draw_rates(finished)
    model = _load_or_train_model()

    rows = []
    meta = []
    for m in scheduled:
        match_date = pd.to_datetime(m["match_date"])
        feats = _match_features(
            team_history,
            int(m["home_team_id"]),
            int(m["away_team_id"]),
            match_date,
        )
        if feats is None:
            continue
        odds = _fetch_and_store_odds(m)
        rows.append(feats)
        meta.append((m, feats, odds))

    if not rows:
        return []

    X = pd.DataFrame(rows)[FEATURES]
    probs = model.predict_proba(X)[:, 1]

    predictions: list[MatchPrediction] = []
    for (m, feats, odds), p in zip(meta, probs):
        home_prob = float(p)
        league_draw_rate = float(league_draw_rates.get(int(m["league_id"]), 0.26))
        draw_prob = _estimate_draw_prob(
            league_draw_rate=league_draw_rate,
            home_drawrate=feats["home_drawrate_5"],
            away_drawrate=feats["away_drawrate_5"],
        )
        away_prob = 1.0 - home_prob - draw_prob
        if away_prob < 0:
            draw_prob = max(0.05, 1.0 - home_prob)
            away_prob = max(0.0, 1.0 - home_prob - draw_prob)
        side, side_prob, side_edge = _best_value_side(
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            home_odds=odds.get("home_odds"),
            draw_odds=odds.get("draw_odds"),
            away_odds=odds.get("away_odds"),
        )
        candidate = {
            "prob": side_prob,
            "home_winrate_5": feats["home_winrate_5"],
            "away_winrate_5": feats["away_winrate_5"],
            "diff_pts_5": feats["diff_pts_5"],
            "diff_winrate_5": feats["diff_winrate_5"],
            "drawrate_avg_5": 0.5 * (feats["home_drawrate_5"] + feats["away_drawrate_5"]),
            "league_draw_rate": league_draw_rate,
            "value_edge": side_edge,
            "side": side,
        }
        if not _passes_strategy(candidate, min_prob):
            continue
        score = _strategy_score(candidate)
        predictions.append(
            MatchPrediction(
                match_id=int(m["match_id"]),
                league=str(m["league_name"]),
                home=str(m["home_name"]),
                away=str(m["away_name"]),
                kickoff=str(m["match_date"]),
                home_win_prob=home_prob,
                score=float(score),
                value_edge=side_edge,
                bet_side=side,
                bet_prob=side_prob,
            )
        )
    predictions.sort(key=lambda x: x.score, reverse=True)
    return predictions


def _fallback_answer(predictions: Iterable[MatchPrediction], target_date: date, min_prob: float) -> str:
    lines = [
        "[PARTIDOS DEL DIA]",
        f"- Fecha: {target_date.isoformat()}",
        f"- Umbral: {min_prob:.0%}",
    ]
    if not predictions:
        lines.append("- No hay partidos para hoy.")
        lines.append("")
        lines.append("[COMO PREGUNTAR]")
        lines.append("- \"que partidos estan bien para hoy?\"")
        lines.append("- \"partidos para apostar hoy\"")
        return "\n".join(lines)

    for p in predictions:
        star = " *" if p.home_win_prob >= 0.6 else ""
        lines.append(
            f"- {p.league}: {p.home} vs {p.away} | "
            f"prob local {p.home_win_prob:.1%}{star} | {p.kickoff}"
        )
    lines.append("")
    lines.append("[COMO PREGUNTAR]")
    lines.append("- \"partidos para apostar hoy\"")
    return "\n".join(lines)


def answer_question(question: str, target_date: date | None = None, min_prob: float = 0.6) -> str:
    if target_date is None:
        target_date = _today()

    # Solo responder a comandos. Para texto libre, devolver menu.
    if question.strip().startswith("/"):
        return answer_command(question.strip(), target_date=target_date, min_prob=min_prob)

    lowered = question.strip().lower()
    if "semana" in lowered:
        dates = _week_dates(target_date, days=7)
        preds_by_date = [(d, get_predictions_for_date(d, min_prob=min_prob)) for d in dates]
        flat = [p for _, preds in preds_by_date for p in preds]
        flat = _apply_gemini_selection(
            flat,
            key=f"gemini_select:semana:{dates[0].isoformat()}_{dates[-1].isoformat()}",
            label=f"{dates[0].isoformat()}_{dates[-1].isoformat()}",
        )
        filtered_ids = {p.match_id for p in flat}
        preds_by_date = [(d, [p for p in preds if p.match_id in filtered_ids]) for d, preds in preds_by_date]
        return _template_partidos_semana(preds_by_date)

    return (
        "Hola! Usa los comandos del menu:\n"
        "- /partidos_dia\n"
        "- /partidos_finde\n"
        "- /partidos_semana\n"
        "- /partidos_apostar"
    )


def answer_command(command: str, target_date: date | None = None, min_prob: float = 0.6) -> str:
    if target_date is None:
        target_date = _today()

    cmd = command.lower()

    if not API_FOOTBALL_KEY:
        return "Falta API_FOOTBALL_KEY en .env"

    if cmd.startswith("/partidos_dia"):
        preds = get_predictions_for_date(target_date, min_prob=min_prob)
        preds = _apply_gemini_selection(
            preds,
            key=f"gemini_select:dia:{target_date.isoformat()}",
            label=target_date.isoformat(),
        )
        return _template_partidos_dia(preds, target_date)

    if cmd.startswith("/partidos_finde"):
        dates = _weekend_dates(target_date)
        preds_by_date = [(d, get_predictions_for_date(d, min_prob=min_prob)) for d in dates]
        flat = [p for _, preds in preds_by_date for p in preds]
        flat = _apply_gemini_selection(
            flat,
            key=f"gemini_select:finde:{dates[0].isoformat()}_{dates[-1].isoformat()}",
            label=f"{dates[0].isoformat()}_{dates[-1].isoformat()}",
        )
        filtered_ids = {p.match_id for p in flat}
        preds_by_date = [(d, [p for p in preds if p.match_id in filtered_ids]) for d, preds in preds_by_date]
        return _template_partidos_finde(preds_by_date)

    if cmd.startswith("/partidos_semana"):
        dates = _week_dates(target_date, days=7)
        preds_by_date = [(d, get_predictions_for_date(d, min_prob=min_prob)) for d in dates]
        flat = [p for _, preds in preds_by_date for p in preds]
        flat = _apply_gemini_selection(
            flat,
            key=f"gemini_select:semana:{dates[0].isoformat()}_{dates[-1].isoformat()}",
            label=f"{dates[0].isoformat()}_{dates[-1].isoformat()}",
        )
        filtered_ids = {p.match_id for p in flat}
        preds_by_date = [(d, [p for p in preds if p.match_id in filtered_ids]) for d, preds in preds_by_date]
        return _template_partidos_semana(preds_by_date)

    if cmd.startswith("/partidos_apostar"):
        preds = get_predictions_for_date(target_date, min_prob=min_prob)
        preds = _apply_gemini_selection(
            preds,
            key=f"gemini_select:apostar:{target_date.isoformat()}",
            label=target_date.isoformat(),
        )
        return _template_partidos_apostar(preds, target_date)

    if cmd.startswith("/reporte"):
        return _report_accuracy()

    return (
        "Comandos disponibles:\n"
        "- /partidos_dia\n"
        "- /partidos_finde\n"
        "- /partidos_semana\n"
        "- /partidos_apostar\n"
        "- /reporte"
    )


def _weekend_dates(start: date) -> list[date]:
    wd = start.weekday()  # Mon=0 ... Sun=6
    if wd == 5:
        return [start, start + timedelta(days=1)]
    if wd == 6:
        return [start]
    saturday = start + timedelta(days=(5 - wd) % 7)
    sunday = saturday + timedelta(days=1)
    return [saturday, sunday]


def _week_dates(start: date, days: int = 7) -> list[date]:
    out = []
    d = start
    for _ in range(max(days, 1)):
        out.append(d)
        d += timedelta(days=1)
    return out


def _fallback_answer_multi(preds_by_date: list[tuple[date, list[MatchPrediction]]], min_prob: float) -> str:
    lines = ["[PARTIDOS DEL FIN DE SEMANA]"]
    for d, preds in preds_by_date:
        lines.append(f"- Fecha: {d.isoformat()} (umbral {min_prob:.0%})")
        if not preds:
            lines.append("  - No hay partidos para hoy.")
            continue
        for p in preds:
            star = " *" if p.home_win_prob >= 0.6 else ""
            lines.append(
                f"  - {p.league}: {p.home} vs {p.away} | "
                f"prob local {p.home_win_prob:.1%}{star} | {p.kickoff}"
            )
    return "\n".join(lines)


def _expand_date_range(start_iso: str, end_iso: str, max_days: int = 14) -> list[date]:
    try:
        start = pd.to_datetime(start_iso).date()
        end = pd.to_datetime(end_iso).date()
    except Exception:
        return []
    if end < start:
        start, end = end, start
    days = (end - start).days
    if days >= max_days:
        end = start + timedelta(days=max_days - 1)
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _passes_strategy(candidate: dict, min_prob: float) -> bool:
    """
    Estrategia de valor:
    - prob >= min_prob
    - si hay odds, exige edge positivo
    - si no hay odds, exige prob muy alta y forma clara
    """
    if candidate["prob"] < min_prob:
        return False
    edge = candidate.get("value_edge")
    if edge is None:
        if candidate["prob"] < max(min_prob, 0.68):
            return False
        if abs(candidate["diff_pts_5"]) < 2:
            return False
        if abs(candidate["diff_winrate_5"]) < 0.1:
            return False
    else:
        if edge < 0.015:
            return False
    if candidate.get("side") == "home":
        if candidate["diff_pts_5"] < 0:
            return False
        if candidate["home_winrate_5"] < 0.45:
            return False
    if candidate.get("side") == "away":
        if candidate["diff_pts_5"] > 0:
            return False
        if candidate["away_winrate_5"] < 0.45:
            return False
    if candidate.get("side") == "draw":
        # para empate, exigimos forma pareja y alta tasa de empate
        if abs(candidate["diff_pts_5"]) > 1.5:
            return False
        if candidate.get("drawrate_avg_5", 0) < 0.28:
            return False
    return True


def _strategy_score(candidate: dict) -> float:
    """
    Score simple para ordenar:
    - prioriza probabilidad
    - bonifica diferencia de puntos y winrate
    """
    edge = candidate.get("value_edge") or 0
    drawrate = candidate.get("drawrate_avg_5", 0)
    side = candidate.get("side")
    form_pts = candidate["diff_pts_5"]
    form_win = candidate["diff_winrate_5"]
    if side == "away":
        form_pts = -form_pts
        form_win = -form_win
    if side == "draw":
        form_pts = 0
        form_win = 0
    return (
        candidate["prob"] * 0.6
        + max(form_pts, 0) * 0.03
        + max(form_win, 0) * 0.25
        + drawrate * 0.15
        + edge * 0.7
    )


def _best_value_side(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    home_odds: float | None,
    draw_odds: float | None,
    away_odds: float | None,
) -> tuple[str, float, float | None]:
    home_edge = _value_edge(home_prob, home_odds)
    draw_edge = _value_edge(draw_prob, draw_odds)
    away_edge = _value_edge(away_prob, away_odds)

    candidates: list[tuple[str, float, float | None]] = [
        ("home", home_prob, home_edge),
        ("draw", draw_prob, draw_edge),
        ("away", away_prob, away_edge),
    ]
    # elegir el mayor edge válido
    best = None
    for side, prob, edge in candidates:
        if edge is None:
            continue
        if best is None or edge > best[2]:
            best = (side, prob, edge)
    if best is not None:
        return best
    return max(candidates, key=lambda x: x[1])


def _value_edge(model_prob: float, home_odds: float | None) -> float | None:
    if not home_odds or home_odds <= 1:
        return None
    implied = 1.0 / home_odds
    return model_prob - implied


def _fetch_and_store_odds(match_row: dict) -> dict:
    api_fixture_id = match_row.get("api_fixture_id")
    if not api_fixture_id:
        return {}

    try:
        odds_resp = get_odds(int(api_fixture_id))
    except Exception:
        return {}

    market = "1X2"
    home_odds = draw_odds = away_odds = None

    try:
        if odds_resp:
            bookmakers = odds_resp[0].get("bookmakers") or []
            for b in bookmakers:
                bets = b.get("bets") or []
                for bet in bets:
                    name = bet.get("name") or ""
                    if name in {"Match Winner", "1X2"}:
                        values = bet.get("values") or []
                        for v in values:
                            val = v.get("value")
                            if val in {"Home", "1"}:
                                home_odds = float(v.get("odd"))
                            elif val in {"Draw", "X"}:
                                draw_odds = float(v.get("odd"))
                            elif val in {"Away", "2"}:
                                away_odds = float(v.get("odd"))
                        break
                if home_odds:
                    break
    except Exception:
        return {}

    if home_odds is None and draw_odds is None and away_odds is None:
        return {}

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            IF NOT EXISTS (
                SELECT 1 FROM core.Odds WHERE match_id = ? AND market = ?
            )
            INSERT INTO core.Odds (match_id, market, home_odds, draw_odds, away_odds)
            VALUES (?, ?, ?, ?, ?)
            """,
            match_row["match_id"], market,
            match_row["match_id"], market, home_odds, draw_odds, away_odds
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {"home_odds": home_odds, "draw_odds": draw_odds, "away_odds": away_odds}


def _format_date_with_weekday(d: date) -> str:
    weekdays = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    return f"{weekdays[d.weekday()]} {d.isoformat()}"


def _side_label(side: str) -> str:
    if side == "away":
        return "Visitante"
    if side == "draw":
        return "Empate"
    return "Local"


def _apply_gemini_selection(preds: list[MatchPrediction], key: str, label: str) -> list[MatchPrediction]:
    cache = _load_cache()
    cached = cache.get(key)
    if _is_cache_valid(cached):
        ids = set(cached["value"])
        return [p for p in preds if p.match_id in ids]

    payload = [
        {
            "match_id": p.match_id,
            "league": p.league,
            "home": p.home,
            "away": p.away,
            "home_win_prob": p.home_win_prob,
            "bet_side": p.bet_side,
            "bet_prob": p.bet_prob,
            "value_edge": p.value_edge,
        }
        for p in preds
    ]
    ids = select_value_picks(payload, label)
    cache[key] = {"value": ids, "ts": date.today().isoformat()}
    _save_cache(cache)
    if ids:
        idset = set(ids)
        return [p for p in preds if p.match_id in idset]
    return preds


def _record_pick(p: MatchPrediction, target_date: date) -> None:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            IF NOT EXISTS (
                SELECT 1 FROM analytics.Predictions
                WHERE match_id = ? AND model_name = ?
            )
            INSERT INTO analytics.Predictions (
                match_id, model_name, created_at, probability
            )
            VALUES (?, ?, GETDATE(), ?)
            """,
            p.match_id, "value_side_v1",
            p.match_id, "value_side_v1", float(p.bet_prob)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _report_accuracy() -> str:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN m.home_goals > m.away_goals THEN 1 ELSE 0 END) AS wins
            FROM analytics.Predictions p
            JOIN core.Matches m ON m.match_id = p.match_id
            WHERE m.status = 'finished'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
                AND p.model_name = ?
            """,
            "value_side_v1",
        )
        row = cur.fetchone()
        total = int(row[0] or 0)
        wins = int(row[1] or 0)
        acc = (wins / total) if total else 0.0

        cur.execute(
            """
            SELECT
                SUM(CASE
                        WHEN m.home_goals > m.away_goals THEN (o.home_odds - 1.0)
                        ELSE -1.0
                    END) AS profit_units,
                COUNT(*) AS bets
            FROM analytics.Predictions p
            JOIN core.Matches m ON m.match_id = p.match_id
            LEFT JOIN core.Odds o ON o.match_id = p.match_id AND o.market = '1X2'
            WHERE m.status = 'finished'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND p.model_name = ?
              AND o.home_odds IS NOT NULL
            """,
            "value_side_v1",
        )
        row2 = cur.fetchone()
        profit_units = float(row2[0] or 0.0)
        bets = int(row2[1] or 0)
        roi = (profit_units / bets) if bets else 0.0

        cur.execute(
            """
            SELECT COUNT(*)
            FROM analytics.Predictions p
            JOIN core.Matches m ON m.match_id = p.match_id
            WHERE m.status = 'finished'
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND p.model_name = ?
              AND m.match_date >= DATEADD(day, -30, GETDATE())
            """,
            "value_side_v1",
        )
        total_30 = int(cur.fetchone()[0] or 0)

        conn.close()
    except Exception:
        return "No pude calcular el reporte de aciertos."

    lines = [
        "REPORTE DE ACIERTOS",
        f"- Picks cerrados: {total}",
        f"- Aciertos: {wins}",
        f"- Accuracy: {acc:.1%}",
        f"- ROI (unidades): {roi:.2%} sobre {bets} picks con odds",
        f"- Picks ultimos 30 dias: {total_30}",
    ]
    return "\n".join(lines)


def _cached_gemini_or_template(
    key: str,
    question: str,
    matches: list[MatchPrediction],
    target_label: str,
    min_prob: float,
    fallback,
) -> str:
    cache = _load_cache()
    cached = cache.get(key)
    if _is_cache_valid(cached):
        return cached["value"]

    matches_payload = [
        {
            "match_id": p.match_id,
            "league": p.league,
            "home": p.home,
            "away": p.away,
            "kickoff": p.kickoff,
            "home_win_prob": p.home_win_prob,
        }
        for p in matches
    ]
    response = generate_answer(question, matches_payload, target_label, min_prob)
    if response:
        cache[key] = {"value": response, "ts": date.today().isoformat()}
        _save_cache(cache)
        return response

    text = fallback()
    cache[key] = {"value": text, "ts": date.today().isoformat()}
    _save_cache(cache)
    return text


def _cache_path() -> Path:
    return Path("cache/telegram_responses.json")


def _load_cache() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import json
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _is_cache_valid(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    value = entry.get("value")
    ts = entry.get("ts")
    if not value or not ts:
        return False
    try:
        cache_day = pd.to_datetime(ts).date()
        return cache_day == _today()
    except Exception:
        return False


def _today() -> date:
    try:
        tz = ZoneInfo(APP_TIMEZONE)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).date()


def _now() -> datetime:
    try:
        tz = ZoneInfo(APP_TIMEZONE)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).replace(tzinfo=None)


def _template_partidos_dia(predictions: list[MatchPrediction], target_date: date) -> str:
    lines = [
        "[OK] PARTIDOS DEL DIA",
        f"Fecha: {_format_date_with_weekday(target_date)}",
        "Filtro: solo picks de valor",
        "Base: ultimos 10 partidos",
        "",
    ]
    if not predictions:
        lines.append("No hay partidos de valor hoy.")
        return "\n".join(lines)
    for p in predictions:
        star = " *" if p.home_win_prob >= 0.6 else ""
        edge = f" | Valor +{p.value_edge:.1%}" if p.value_edge is not None else ""
        _record_pick(p, target_date)
        lines.append(
            f"- {p.league}: {p.home} vs {p.away} | "
            f"Prob {p.bet_prob:.1%}{star}{edge} | {p.kickoff} | Apuesta: {_side_label(p.bet_side)}"
        )
    return "\n".join(lines)

def _template_partidos_finde(preds_by_date: list[tuple[date, list[MatchPrediction]]]) -> str:
    lines = ["[OK] PARTIDOS DEL FINDE", "Filtro: solo picks de valor", "Base: ultimos 10 partidos", ""]
    for d, preds in preds_by_date:
        lines.append(f"Fecha: {_format_date_with_weekday(d)}")
        if not preds:
            lines.append("  No hay picks de valor.")
            continue
        for p in preds:
            star = " *" if p.home_win_prob >= 0.6 else ""
            edge = f" | Valor +{p.value_edge:.1%}" if p.value_edge is not None else ""
            _record_pick(p, d)
            lines.append(
                f"  - {p.league}: {p.home} vs {p.away} | "
                f"Prob {p.bet_prob:.1%}{star}{edge} | {p.kickoff} | Apuesta: {_side_label(p.bet_side)}"
            )
    return "\n".join(lines)


def _template_partidos_semana(preds_by_date: list[tuple[date, list[MatchPrediction]]]) -> str:
    lines = ["📅 PARTIDOS DE LA SEMANA", "🔎 Solo picks de valor", "📊 Base: ultimos 10 partidos", ""]
    for d, preds in preds_by_date:
        lines.append(f"🗓️ { _format_date_with_weekday(d)}")
        if not preds:
            lines.append("  • (sin picks de valor)")
            continue
        for p in preds:
            star = " ⭐" if p.home_win_prob >= 0.6 else ""
            edge = f" | 💎 Valor +{p.value_edge:.1%}" if p.value_edge is not None else ""
            _record_pick(p, d)
            lines.append(
                f"  ⚽ {p.league}: {p.home} vs {p.away}\n"
                f"     🎯 Prob {p.bet_prob:.1%}{star}{edge}\n"
                f"     🕒 {p.kickoff} | 🧾 Apuesta: {_side_label(p.bet_side)}"
            )
    return "\n".join(lines)

def _template_partidos_apostar(predictions: list[MatchPrediction], target_date: date) -> str:
    lines = [
        "[OK] PARTIDOS PARA APOSTAR",
        f"Fecha: {_format_date_with_weekday(target_date)}",
        "Filtro: valor (prob >= 60%, forma positiva)",
        "Base: ultimos 10 partidos",
        "",
    ]
    if not predictions:
        lines.append("No hay picks de valor.")
        return "\n".join(lines)
    for p in predictions:
        star = " *" if p.home_win_prob >= 0.6 else ""
        edge = f" | Valor +{p.value_edge:.1%}" if p.value_edge is not None else ""
        _record_pick(p, target_date)
        lines.append(
            f"- {p.league}: {p.home} vs {p.away} | "
            f"Prob {p.bet_prob:.1%}{star}{edge} | {p.kickoff} | Apuesta: {_side_label(p.bet_side)}"
        )
    return "\n".join(lines)

def answer_and_notify(question: str, target_date: date | None = None, min_prob: float = 0.6) -> str:
    response = answer_question(question, target_date=target_date, min_prob=min_prob)
    send_message(response)
    return response
