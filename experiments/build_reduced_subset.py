"""Build a reproducible reduced LVIS-unseen subset for the oracle + ablation
re-runs. Selects N images covering all 10 unseen categories (deterministic),
writes the filename list, and creates a symlinked images dir that the existing
run_vlm.py can consume via a config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_oracle_localisation import (  # same dir
    LVIS_IMAGES,
    PROJECT_ROOT,
    load_data,
    select_subset,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--name", default="lvis_v1_val_unseen_reduced")
    args = ap.parse_args()

    cats, id2file, by_image, lpi = load_data()
    image_ids = select_subset(cats, id2file, by_image, lpi, args.n)
    files = [id2file[i] for i in image_ids]

    out_dir = PROJECT_ROOT / "data/subsets" / args.name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # refresh symlinks
    for old in img_dir.glob("*.jpg"):
        old.unlink()
    n_linked = 0
    for fn in files:
        src = (LVIS_IMAGES / fn).resolve()
        dst = img_dir / fn
        if src.exists():
            dst.symlink_to(src)
            n_linked += 1

    # category coverage
    from collections import Counter
    cov = Counter()
    for i in image_ids:
        for a in by_image[i]:
            cov[cats[a["category_id"]]] += 1

    (out_dir / "subset_manifest.json").write_text(json.dumps({
        "name": args.name, "n_images": len(files), "n_linked": n_linked,
        "files": files, "category_ann_counts": dict(cov),
    }, indent=2))
    print(f"[subset] {args.name}: {n_linked}/{len(files)} images linked at {img_dir}")
    print(f"[subset] categories covered ({len(cov)}): {dict(cov)}")


if __name__ == "__main__":
    main()
