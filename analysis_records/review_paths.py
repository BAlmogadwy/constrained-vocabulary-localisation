"""Portable paths for optional offline reruns; importing this makes no changes."""
import os
from pathlib import Path

RECORDS = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CVL_REPO_ROOT', str(RECORDS.parent))).resolve()
OUTPUT = Path(os.environ.get('CVL_RESULTS_DIR', str(RECORDS / 'recomputed'))).resolve()

def output_path(name):
    if OUTPUT == RECORDS:
        raise ValueError('CVL_RESULTS_DIR must not overwrite the retained analysis_records directory')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT / name
