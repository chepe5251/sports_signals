import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.signals_service import answer_and_notify


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Pregunta: ").strip()
    if not question:
        print("Necesito una pregunta.")
        return

    response = answer_and_notify(question)
    print(response)


if __name__ == "__main__":
    main()
