import json
from pathlib import Path

root = Path("data")
raw = root / "raw" / "coco2017" / "val2017"
metadata = root / "processed" / "lvis_v1" / "val" / "lvis_v1_val_unseen.json"
subset = root / "subsets" / "lvis_v1_val_unseen" / "images"
subset.mkdir(parents=True, exist_ok=True)

with metadata.open(encoding="utf-8") as fh:
    data = json.load(fh)

for img in data["images"]:
    filename = img["coco_url"].split("/")[-1]
    src = raw / filename
    dst = subset / filename
    if not src.exists():
        raise FileNotFoundError(f"Missing COCO val image: {src}")
    if not dst.exists():
        dst.symlink_to(src.resolve())

print(f"{len(list(subset.iterdir()))} images symlinked into {subset}")
