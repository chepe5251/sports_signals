from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.api_telegram import get_updates, send_message
from src.services.signals_service import answer_question, answer_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=0, help="Minutos activos escuchando (0 = infinito)")
    parser.add_argument("--chat-id", type=str, default=None, help="Responder solo a este chat_id")
    args = parser.parse_args()

    end_time = None if args.minutes <= 0 else time.time() + max(args.minutes, 1) * 60
    offset = None

    if end_time is None:
        print("Escuchando Telegram en modo infinito...")
    else:
        print(f"Escuchando Telegram por {args.minutes} minutos...")

    while True:
        if end_time is not None and time.time() >= end_time:
            break
        try:
            updates = get_updates(offset=offset, timeout=25)
        except Exception:
            time.sleep(2)
            continue
        for upd in updates:
            update_id = upd.get("update_id")
            if update_id is not None:
                offset = update_id + 1

            msg = upd.get("message") or {}
            chat = msg.get("chat") or {}
            text = msg.get("text")
            chat_id = chat.get("id")
            if not text or chat_id is None:
                continue

            if args.chat_id and str(chat_id) != str(args.chat_id):
                continue

            if text.strip().startswith("/"):
                reply = answer_command(text.strip())
            else:
                reply = answer_question(text)
            send_message(reply, chat_id=str(chat_id))

    print("Listo. Listener detenido.")


if __name__ == "__main__":
    main()
