from __future__ import annotations

import json
import os
from typing import Any, Iterable

from google import genai
from google.genai import types

from src.utils.config import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.leagues import PRIMARY_LEAGUES


_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    return ""


def generate_answer(question: str, matches: list[dict], target_date: str, min_prob: float) -> str | None:
    if _client is None:
        return None

    matches_lines = []
    for m in matches:
        line = (
            f"{m['league']} | {m['home']} vs {m['away']} | "
            f"home_win_prob={m['home_win_prob']:.1%} | {m['kickoff']}"
        )
        matches_lines.append(line)

    data_block = "\n".join(matches_lines) if matches_lines else "(sin partidos o sin datos)"

    system_msg = (
        "Eres un analista de futbol. Responde en espanol natural y conversacional, "
        "sin formato rigido. Interpreta la intencion del usuario y responde lo mejor posible. "
        "Solo puedes usar los partidos entregados en el bloque de datos (no inventes). "
        "Usa fechas exactas en formato YYYY-MM-DD (no inventes el dia de la semana). "
        "Si no hay datos suficientes o la consulta no es de partidos, dilo y ofrece ayuda."
    )
    user_msg = (
        f"Fecha objetivo: {target_date}\n"
        f"Umbral minimo: {min_prob:.0%}\n"
        f"Pregunta: {question}\n"
        "Datos de partidos (no inventar):\n"
        f"{data_block}\n\n"
        "Devuelve la lista de partidos y su probabilidad de ganar en casa. "
        "Marca con una estrella (*) los partidos con probabilidad >= 60%."
    )

    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_msg}\n\n{user_msg}",
        )
        text = _extract_output_text(response)
        text = (text or "").strip()
        if not text or text in {"[]", "{}"}:
            return None
        return text
    except Exception:
        return None


def parse_date_range(question: str, today_iso: str) -> dict | None:
    """
    Devuelve un dict con start_date y end_date (YYYY-MM-DD).
    Si no se puede inferir, devuelve None.
    """
    if _client is None:
        return None

    system_msg = (
        "Eres un parser de fechas. Devuelve SOLO JSON valido. "
        "Formato: {\"start_date\":\"YYYY-MM-DD\",\"end_date\":\"YYYY-MM-DD\"}. "
        "Si el usuario pide un solo dia, start_date=end_date. "
        "Si no puedes inferir fecha, devuelve {}."
    )
    user_msg = (
        f"Hoy es {today_iso}.\n"
        f"Texto del usuario: {question}\n"
        "Interpreta frases como 'manana', 'la otra semana', 'el finde', "
        "'este sabado', 'proximo domingo', etc."
    )
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_msg}\n\n{user_msg}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        text = _extract_output_text(response)
        if not text:
            return None
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        start_date = str(data.get("start_date") or "").strip()
        end_date = str(data.get("end_date") or "").strip()
        if not start_date or not end_date:
            return None
        return {"start_date": start_date, "end_date": end_date}
    except Exception:
        return None


def select_value_picks(
    matches: list[dict],
    target_label: str,
) -> list[int]:
    """
    Devuelve una lista de match_id seleccionados como valor.
    """
    if _client is None:
        return []

    system_msg = (
        "Eres un analista de apuestas. Devuelve SOLO JSON valido: "
        "{\"match_ids\":[1,2,3]}. Si ninguno tiene valor, devuelve {\"match_ids\":[]}."
    )
    lines = []
    for m in matches:
        lines.append(
            f"{m['match_id']} | {m['league']} | {m['home']} vs {m['away']} | "
            f"side={m.get('bet_side')} | prob={m.get('bet_prob')} | valor={m.get('value_edge')}"
        )
    data_block = "\n".join(lines) if lines else "(sin partidos)"
    user_msg = (
        f"Rango: {target_label}\n"
        "Selecciona solo los partidos de mayor valor esperado. "
        "Prioriza value_edge positivo y probabilidad alta del lado sugerido. "
        "Evita picks sin odds salvo probabilidad muy alta.\n"
        f"Datos:\n{data_block}"
    )
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_msg}\n\n{user_msg}",
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        text = _extract_output_text(response)
        if not text:
            return []
        data = json.loads(text)
        ids = data.get("match_ids")
        if isinstance(ids, list):
            return [int(x) for x in ids if isinstance(x, (int, float, str)) and str(x).isdigit()]
        return []
    except Exception:
        return []


def get_top5_fixtures(target_date: str) -> list[dict]:
    """
    Devuelve fixtures de las ligas principales via Gemini.
    Formato esperado:
    [
      {"league":"Premier League","home":"Team A","away":"Team B","kickoff":"YYYY-MM-DDTHH:MM:SS"},
      ...
    ]
    """
    if _client is None:
        return []

    leagues_str = ", ".join(PRIMARY_LEAGUES)
    system_msg = (
        "Devuelve SOLO un JSON valido (sin texto extra). "
        f"Solo incluye estas ligas: {leagues_str}. "
        "Si no sabes la hora exacta, deja kickoff vacio."
    )
    user_msg = (
        f"Fecha: {target_date}\n"
        "Necesito la lista de partidos de estas ligas para esa fecha. "
        "Cada item debe tener: league, home, away, kickoff (ISO 8601 o vacio). "
        "Si hay partidos, devuelvelos aunque no tengas la hora exacta."
    )
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_msg}\n\n{user_msg}",
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        text = _extract_output_text(response)
        if os.getenv("GEMINI_DEBUG") == "1":
            with open("gemini_fixtures_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\n=== {target_date} ===\n")
                f.write(text or "<empty>\n")
        if not text:
            return []
        data = _safe_json_list(text)
        return _validate_fixtures(data)
    except Exception:
        return []


def _safe_json_list(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("fixtures"), list):
            return data["fixtures"]
    except Exception:
        pass
    # Try to extract first JSON array from text
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start:end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("fixtures"), list):
                return data["fixtures"]
        except Exception:
            return []
    return []


def _validate_fixtures(items: list[dict]) -> list[dict]:
    if not items:
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        league = str(item.get("league") or "").strip()
        home = str(item.get("home") or "").strip()
        away = str(item.get("away") or "").strip()
        kickoff = str(item.get("kickoff") or "").strip()
        if not league or not home or not away:
            continue
        out.append(
            {
                "league": league,
                "home": home,
                "away": away,
                "kickoff": kickoff,
            }
        )
    return out
