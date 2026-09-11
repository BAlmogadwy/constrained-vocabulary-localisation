"""Oracle-Localisation diagnosis for the 2026 VLM roster.

For each ground-truth box in the (reduced) LVIS-unseen subset, crop the object
and ask each VLM to classify it into exactly one of that image's candidate
categories. This removes the localisation bottleneck and measures pure semantic
recognition (the "recognition upper bound"), and also surfaces JSON-interface
brittleness (models that cannot answer under a strict classification format).

API/serverless only. Per-(model, crop) results are cached so reruns do not
re-spend. Outputs:
  results/raw/oracle/<run>/crops/<image_stem>/<ann_id>.jpg          (cached crops)
  results/raw/oracle/<run>/per_crop/<model>/<image_stem>__<ann>.json (cached calls)
  results/raw/oracle/<run>/<model>_oracle.json                       (per-model aggregate)
  results/raw/oracle/<run>/oracle_summary.json                       (all models)
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

import os  # noqa: E402

LVIS_ANN = PROJECT_ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
LVIS_DYNAMIC = PROJECT_ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
LVIS_IMAGES = PROJECT_ROOT / "data/subsets/lvis_v1_val_unseen/images"

# Roster: model -> (family, default_model_id, key_env, base_url_or_None)
ROSTER: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    "gpt-5.5": ("openai", os.getenv("OPENAI_GPT55_MODEL", "gpt-5.5"), "OPENAI_API_KEY", None),
    "gpt-5.4-mini": ("openai", os.getenv("OPENAI_GPT54_MINI_MODEL", "gpt-5.4-mini"), "OPENAI_API_KEY", None),
    "claude-opus-4.8": ("anthropic", os.getenv("ANTHROPIC_CLAUDE_OPUS_48_MODEL", "claude-opus-4-8"), "ANTHROPIC_API_KEY", None),
    "claude-haiku-4.5": ("anthropic", os.getenv("ANTHROPIC_CLAUDE_HAIKU_45_MODEL", "claude-haiku-4-5"), "ANTHROPIC_API_KEY", None),
    "gemini-3.1-pro": ("gemini", os.getenv("GEMINI_31_PRO_MODEL", "gemini-3.1-pro-preview"), "GOOGLE_API_KEY", None),
    "gemini-3.5-flash": ("gemini", os.getenv("GEMINI_35_FLASH_MODEL", "gemini-3.5-flash"), "GOOGLE_API_KEY", None),
    "qwen-vl-max": ("compat", os.getenv("QWEN_VL_MAX_MODEL", "qwen-vl-max"), "QWEN_API_KEY",
                    os.getenv("QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")),
    "qwen3-vl-235b-openrouter": ("compat", os.getenv("OPENROUTER_QWEN3_VL_235B_MODEL", "qwen/qwen3-vl-235b-a22b-instruct"), "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "qwen2.5-vl-72b-openrouter": ("compat", os.getenv("OPENROUTER_QWEN25_VL_72B_MODEL", "qwen/qwen2.5-vl-72b-instruct"), "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "mistral-large-openrouter": ("compat", os.getenv("OPENROUTER_MISTRAL_LARGE_MODEL", "mistralai/mistral-large-2512"), "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "llama-4-maverick-openrouter": ("compat", os.getenv("OPENROUTER_LLAMA4_MAVERICK_MODEL", "meta-llama/llama-4-maverick"), "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "gemma-3-27b-openrouter": ("compat", os.getenv("OPENROUTER_GEMMA3_27B_MODEL", "google/gemma-3-27b-it"), "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
}
DEFAULT_MODELS = list(ROSTER.keys())


def norm_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)            # drop parenthetical, e.g. stapler_(stapling_machine)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def classify_prompt(options: List[str]) -> str:
    opts = ", ".join(options)
    return (
        "The image shows exactly one main object, tightly cropped. "
        f"Classify it into exactly one of these categories: {opts}. "
        'Respond with ONLY a JSON object of the form {"label": "<one category from the list>"} '
        "and nothing else."
    )


def parse_predicted(raw_text: Optional[str], options: List[str]) -> Tuple[Optional[str], bool]:
    """Return (predicted_option_or_None, format_ok)."""
    if not raw_text:
        return None, False
    opt_norm = {norm_label(o): o for o in options}
    # 1) strict JSON object with a label key
    m = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            lab = obj.get("label")
            if isinstance(lab, str):
                key = norm_label(lab)
                if key in opt_norm:
                    return opt_norm[key], True
                # label present & well-formed but not in option set
                return None, True
        except (json.JSONDecodeError, AttributeError):
            pass
    # 2) fallback: an option name appears verbatim in the text
    text_norm = norm_label(raw_text)
    for key, orig in opt_norm.items():
        if key and re.search(rf"\b{re.escape(key)}\b", text_norm):
            return orig, False  # recovered, but not via the requested JSON format
    return None, False


# ---------------------------------------------------------------------------
# Provider classification calls. Each returns (raw_text, raw_obj).
# ---------------------------------------------------------------------------
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_openai(model_id: str, key_env: str, base: Optional[str], image: Path, prompt: str) -> Tuple[Optional[str], Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv(key_env))
    resp = client.responses.create(
        model=model_id,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{_b64(image)}"},
        ]}],
        max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2048")),
    )
    text = getattr(resp, "output_text", None)
    if not text:
        pieces = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message" or (isinstance(item, dict) and item.get("type") == "message"):
                contents = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
                for c in contents or []:
                    t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                    if t:
                        pieces.append(t)
        text = "\n".join(pieces) if pieces else None
    raw = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
    return text, raw


def call_anthropic(model_id: str, key_env: str, base: Optional[str], image: Path, prompt: str) -> Tuple[Optional[str], Any]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv(key_env))
    msg = client.messages.create(
        model=model_id,
        max_tokens=256,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(image)}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = None
    try:
        text = msg.content[0].text
    except (AttributeError, IndexError):
        pass
    raw = msg.model_dump() if hasattr(msg, "model_dump") else str(msg)
    return text, raw


def call_gemini(model_id: str, key_env: str, base: Optional[str], image: Path, prompt: str) -> Tuple[Optional[str], Any]:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv(key_env))
    model = genai.GenerativeModel(model_id)
    img = Image.open(image)
    resp = model.generate_content([prompt, img], stream=True)
    resp.resolve()
    text = getattr(resp, "text", "") or ""
    return text, text


def call_compat(model_id: str, key_env: str, base: Optional[str], image: Path, prompt: str) -> Tuple[Optional[str], Any]:
    import requests
    headers = {"Authorization": f"Bearer {os.getenv(key_env)}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image)}"}},
        ]}],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    delay = float(os.getenv("OPENAI_COMPATIBLE_RETRY_DELAY", "30"))
    retries = int(os.getenv("OPENAI_COMPATIBLE_MAX_RETRIES", "4"))
    for attempt in range(retries + 1):
        r = requests.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=120)
        if r.status_code == 429 and attempt < retries:
            time.sleep(delay)
            continue
        r.raise_for_status()
        data = r.json()
        break
    text = None
    try:
        text = data["choices"][0]["message"]["content"]
        if isinstance(text, list):  # some providers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    except (KeyError, IndexError, TypeError):
        pass
    return text, data


CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "gemini": call_gemini, "compat": call_compat}


def load_data():
    ann = json.loads(LVIS_ANN.read_text())
    dyn = json.loads(LVIS_DYNAMIC.read_text())
    lpi = dyn.get("labels_per_image", {})
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    id2file = {im["id"]: f"{im['id']:012d}.jpg" for im in ann["images"]}
    by_image: Dict[int, List[dict]] = defaultdict(list)
    for a in ann["annotations"]:
        by_image[a["image_id"]].append(a)
    return cats, id2file, by_image, lpi


def select_subset(cats, id2file, by_image, lpi, target: int) -> List[int]:
    """Pick image_ids covering all categories, capped near `target`, deterministic."""
    # categories present per image
    img_cats: Dict[int, set] = {}
    for iid, anns in by_image.items():
        fn = id2file[iid]
        if fn not in lpi:
            continue
        img_cats[iid] = {cats[a["category_id"]] for a in anns}
    all_cats = set(cats.values())
    chosen: List[int] = []
    covered: set = set()
    # greedy: rarest categories first by selecting images that add new coverage
    order = sorted(img_cats, key=lambda i: id2file[i])
    for iid in order:
        if not (img_cats[iid] - covered):
            continue
        chosen.append(iid)
        covered |= img_cats[iid]
        if covered >= all_cats:
            break
    # top up to target with additional images (sorted), avoiding dups
    for iid in order:
        if len(chosen) >= target:
            break
        if iid not in chosen:
            chosen.append(iid)
    return sorted(chosen, key=lambda i: id2file[i])


def make_crop(image_path: Path, bbox, out_path: Path, pad: float = 0.08) -> bool:
    if out_path.exists():
        return True
    try:
        im = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return False
    W, H = im.size
    x, y, w, h = bbox
    px, py = w * pad, h * pad
    x1 = max(0, int(x - px)); y1 = max(0, int(y - py))
    x2 = min(W, int(x + w + px)); y2 = min(H, int(y + h + py))
    if x2 <= x1 or y2 <= y1:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.crop((x1, y1, x2, y2)).save(out_path, "JPEG", quality=92)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="oracle_lvis_2026")
    ap.add_argument("--limit-images", type=int, default=50)
    ap.add_argument("--max-crops", type=int, default=0, help="0 = no cap")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--test", action="store_true", help="2 images, first 2 models")
    args = ap.parse_args()

    cats, id2file, by_image, lpi = load_data()
    out_root = PROJECT_ROOT / "results/raw/oracle" / args.run
    crops_dir = out_root / "crops"

    target = 2 if args.test else args.limit_images
    image_ids = select_subset(cats, id2file, by_image, lpi, target)
    if args.test:
        image_ids = image_ids[:2]

    # Build crop list: (image_stem, ann_id, crop_path, true_label, options)
    crop_items: List[Tuple[str, int, Path, str, List[str]]] = []
    for iid in image_ids:
        fn = id2file[iid]
        stem = Path(fn).stem
        options = lpi[fn]["labels"]
        img_path = LVIS_IMAGES / fn
        for a in by_image[iid]:
            true_label = cats[a["category_id"]]
            if true_label not in options:
                continue
            cp = crops_dir / stem / f"{a['id']}.jpg"
            if make_crop(img_path, a["bbox"], cp):
                crop_items.append((stem, a["id"], cp, true_label, options))
    if args.max_crops:
        crop_items = crop_items[: args.max_crops]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.test:
        models = models[:2]
    print(f"[oracle] images={len(image_ids)} crops={len(crop_items)} models={models}")
    cat_cov = sorted({tl for *_, tl, _ in crop_items})
    print(f"[oracle] category coverage ({len(cat_cov)}): {cat_cov}")

    summary = {}
    for model in models:
        if model not in ROSTER:
            print(f"[skip] unknown model {model}")
            continue
        family, model_id, key_env, base = ROSTER[model]
        caller = CALLERS[family]
        per_crop_dir = out_root / "per_crop" / model
        per_crop_dir.mkdir(parents=True, exist_ok=True)
        records = []
        t0 = time.perf_counter()
        for stem, ann_id, cp, true_label, options in crop_items:
            cache = per_crop_dir / f"{stem}__{ann_id}.json"
            if cache.exists():
                records.append(json.loads(cache.read_text()))
                continue
            prompt = classify_prompt(options)
            start = time.perf_counter()
            try:
                raw_text, raw_obj = caller(model_id, key_env, base, cp, prompt)
                err = None
            except Exception as exc:  # noqa: BLE001
                raw_text, raw_obj, err = None, {"error": str(exc)}, str(exc)
            lat = time.perf_counter() - start
            pred, fmt_ok = parse_predicted(raw_text, options)
            rec = {
                "image": stem, "ann_id": ann_id, "true_label": true_label,
                "options": options, "predicted": pred,
                "correct": bool(pred and norm_label(pred) == norm_label(true_label)),
                "format_ok": fmt_ok, "error": err,
                "latency_sec": lat, "raw_text": raw_text, "raw_response": raw_obj,
            }
            cache.write_text(json.dumps(rec, indent=2))
            records.append(rec)
            tag = "ok" if rec["correct"] else ("fmt!" if not fmt_ok else "miss")
            print(f"  [{model}] {stem}/{ann_id} true={true_label} pred={pred} {tag} ({lat:.2f}s)")
        n = len(records)
        n_correct = sum(r["correct"] for r in records)
        n_fmt_fail = sum((not r["format_ok"]) for r in records)
        n_err = sum(r["error"] is not None for r in records)
        agg = {
            "model": model, "model_id": model_id, "n_crops": n,
            "recognition_accuracy": (n_correct / n) if n else 0.0,
            "format_failure_rate": (n_fmt_fail / n) if n else 0.0,
            "n_correct": n_correct, "n_format_fail": n_fmt_fail, "n_error": n_err,
            "wall_time_sec": time.perf_counter() - t0,
            "records": records,
        }
        (out_root / f"{model.replace('/', '_')}_oracle.json").write_text(json.dumps(agg, indent=2))
        summary[model] = {k: agg[k] for k in ("model_id", "n_crops", "recognition_accuracy", "format_failure_rate", "n_correct", "n_error")}
        print(f"[oracle] {model}: acc={agg['recognition_accuracy']:.3f} fmt_fail={agg['format_failure_rate']:.3f} (n={n})")

    (out_root / "oracle_summary.json").write_text(json.dumps({
        "run": args.run, "n_images": len(image_ids), "n_crops": len(crop_items),
        "category_coverage": cat_cov, "models": summary,
    }, indent=2))
    print(f"[oracle] summary -> {out_root / 'oracle_summary.json'}")


if __name__ == "__main__":
    main()
