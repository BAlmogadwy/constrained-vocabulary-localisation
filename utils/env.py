"""Minimal stub for the deleted utils package (HEAD commit removed it).
run_oracle_localisation imports load_environment to read API keys from .env;
for offline analyses a no-op (optional dotenv) suffices.
"""
from pathlib import Path


def load_environment() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass
