"""Unpack saved detector outputs and optionally copy an existing COCO annotation file.

This uses the standard library only; it never downloads data or runs inference/evaluation.
Existing files with different contents are not overwritten.
"""
import argparse
import json
import zipfile
from pathlib import Path
from review_paths import ROOT, RECORDS
from verify_review import matches

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-annotations', type=Path, help='Existing instances_val2017.json from COCO')
    args = parser.parse_args()
    manifest = json.loads((RECORDS / 'review_manifest.json').read_text(encoding='utf-8'))
    archive_rel = 'results/reviewer_detector_outputs.zip'
    archive = ROOT / archive_rel
    if not matches(archive.read_bytes(), manifest['archives'][archive_rel]):
        raise ValueError('Detector archive checksum mismatch')
    expected = {p:e for p,e in manifest['inputs'].items() if e.get('archive') == archive_rel}
    writes = []
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != set(expected) or len(z.namelist()) != len(expected):
            raise ValueError('Archive members do not match the manifest')
        for rel, entry in expected.items():
            path = (ROOT / rel).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Archive member escapes repository')
            data = z.read(rel)
            if not matches(data, entry):
                raise ValueError(f'Archive member checksum mismatch: {rel}')
            writes.append((path, data, entry))
    if args.coco_annotations:
        rel = 'data/raw/coco2017/annotations/instances_val2017.json'
        data = args.coco_annotations.read_bytes()
        entry = manifest['inputs'][rel]
        if not matches(data, entry):
            raise ValueError('COCO annotation checksum differs from the retained evaluation input')
        writes.append((ROOT / rel, data, entry))
    # Validate all destinations before changing any files.
    for path, data, entry in writes:
        if path.exists() and (not path.is_file() or not matches(path.read_bytes(), entry)):
            raise ValueError(f'Refusing to overwrite a different file: {path}')
    count = 0
    for path, data, entry in writes:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            count += 1
    print(f'Prepared {len(expected)} saved detector records; wrote {count} missing files.')
    print('No inference, API requests or metric calculations were run.')

if __name__ == '__main__':
    main()
