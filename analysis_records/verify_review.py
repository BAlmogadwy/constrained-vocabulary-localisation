"""Check retained files and display saved results, using only Python's standard library.

No imports of evaluation/model code, API calls, downloads or metric calculations.
Run from the repository root: python analysis_records/verify_review.py
Use --all-inputs after prepare_review_inputs.py to check external COCO annotations too.
"""
import argparse
import hashlib
import json
from pathlib import Path
from review_paths import ROOT, RECORDS

def digest(data):
    return hashlib.sha256(data).hexdigest()

def matches(data, entry):
    return digest(data) == entry['sha256'] or (entry.get('lf_sha256') is not None and digest(data.replace(b'\r\n', b'\n')) == entry['lf_sha256'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-inputs', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((RECORDS / 'review_manifest.json').read_text(encoding='utf-8'))
    failures = []
    checked = 0
    for section in ['records', 'archives', 'inputs']:
        for rel, entry in manifest[section].items():
            if section == 'inputs' and entry.get('archive') and not args.all_inputs:
                continue
            if entry.get('external') and not args.all_inputs:
                continue
            path = ROOT / rel
            if not path.is_file():
                failures.append(f'Missing: {rel}')
            elif not matches(path.read_bytes(), entry):
                failures.append(f'Checksum mismatch: {rel}')
            else:
                checked += 1
    if failures:
        for failure in failures[:20]:
            print(failure)
        print(f'{len(failures)} problem(s). See REVIEWER_GUIDE.md.')
        return 1
    print(f'PASS: {checked} files match the retained evidence (LF/CRLF differences allowed).')
    if not args.all_inputs:
        print('Compressed detector inputs checked through their archive checksum. External COCO annotations not checked.')
    replay = json.loads((RECORDS / 'replay_results.json').read_text(encoding='utf-8'))
    lvis = json.loads((RECORDS / 'lvis_semantics_sensitivity.json').read_text(encoding='utf-8'))
    print('Saved official LVIS AP:')
    for model, entry in lvis['models'].items():
        print(f"  {model}: {entry['official_lvis']['metrics']['AP']:.6f}")
    print('Saved COCO router results:')
    for key in ['coco_original', 'coco_exclude_lvis_overlap']:
        result = replay['router'][key]
        print(f"  {result['images']} images: detector {result['detector']['ap']:.6f}; router {result['fused']['ap']:.6f}")
    print('Read saved results only. No AP or bootstrap was recomputed.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
