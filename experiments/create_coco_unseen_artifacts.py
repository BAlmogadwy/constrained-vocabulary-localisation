#!/usr/bin/env python3
"""Create COCO-unseen metric tables, figures, and cost ledgers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON = PROJECT_ROOT / "results" / "coco_unseen_metrics_summary.json"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "experiments" / "results" / "figures"

BATCH_RESULTS_ROOT = PROJECT_ROOT / "results" / "raw" / "vlm" / "coco_unseen_refresh_2026"
OPENROUTER_RESULTS_ROOT = PROJECT_ROOT / "results" / "raw" / "vlm" / "coco_serverless_refresh_2026"

METRICS_CSV = TABLE_DIR / "coco_unseen_metrics_api_refresh_2026.csv"
METRICS_MD = TABLE_DIR / "coco_unseen_metrics_api_refresh_2026.md"
COSTS_CSV = TABLE_DIR / "coco_unseen_costs_api_refresh_2026.csv"
COSTS_MD = TABLE_DIR / "coco_unseen_costs_api_refresh_2026.md"
ACCOUNT_COSTS_CSV = TABLE_DIR / "coco_unseen_costs_by_account_api_refresh_2026.csv"
ACCOUNT_COSTS_MD = TABLE_DIR / "coco_unseen_costs_by_account_api_refresh_2026.md"

# Prices are the user's planning prices in USD per 1M input/output tokens.
# OpenAI, Anthropic, and Gemini used batch APIs, so the 50% batch discount is applied.
BATCH_PRICES: Dict[str, Dict[str, Any]] = {
    "gpt-5.5": {"account": "OpenAI Batch", "input": 5.0, "output": 30.0, "discount": 0.5},
    "gpt-5.4-mini": {"account": "OpenAI Batch", "input": 0.75, "output": 4.5, "discount": 0.5},
    "claude-opus-4.8": {"account": "Anthropic Batch", "input": 5.0, "output": 25.0, "discount": 0.5},
    "claude-haiku-4.5": {"account": "Anthropic Batch", "input": 1.0, "output": 5.0, "discount": 0.5},
    "gemini-3.1-pro": {"account": "Gemini Batch", "input": 2.0, "output": 12.0, "discount": 0.5},
    "gemini-3.5-flash": {"account": "Gemini Batch", "input": 1.5, "output": 9.0, "discount": 0.5},
}

MODEL_ORDER = {
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "qwen3-vl-235b-openrouter": "Qwen3-VL-235B",
    "gpt-5.5": "GPT-5.5",
    "qwen2.5-vl-72b-openrouter": "Qwen2.5-VL-72B",
    "claude-opus-4.8": "Claude Opus 4.8",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "llama-4-maverick-openrouter": "Llama 4 Maverick",
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "gemma-3-27b-openrouter": "Gemma 3 27B",
    "mistral-large-openrouter": "Mistral Large",
}


def display_name(model: str) -> str:
    return MODEL_ORDER.get(model, model)


def fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def markdown_table(rows: List[Dict[str, Any]], columns: List[str], decimals: Optional[Dict[str, int]] = None) -> str:
    decimals = decimals or {}
    rendered = []
    for row in rows:
        rendered.append([fmt(row.get(col), decimals.get(col, 4)) for col in columns])

    widths = [len(col) for col in columns]
    for row in rendered:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header = "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx, _ in enumerate(columns)) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rendered
    ]
    return "\n".join([header, sep, *body]) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def load_metric_rows() -> List[Dict[str, Any]]:
    data = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    rows = data.get("coco2017-val", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected metrics JSON shape in {METRICS_JSON}")
    for row in rows:
        row["dataset"] = "coco-unseen-901"
        row["display_model"] = display_name(row["model"])
    return sorted(rows, key=lambda row: (row.get("ap") or 0.0), reverse=True)


def usage_from_openai(raw: Dict[str, Any]) -> Dict[str, int]:
    usage = raw.get("response", {}).get("body", {}).get("usage", {})
    details = usage.get("output_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def usage_from_anthropic(raw: Dict[str, Any]) -> Dict[str, int]:
    usage = raw.get("result", {}).get("message", {}).get("usage", {})
    input_tokens = int(usage.get("input_tokens") or 0)
    input_tokens += int(usage.get("cache_creation_input_tokens") or 0)
    input_tokens += int(usage.get("cache_read_input_tokens") or 0)
    details = usage.get("output_tokens_details") or {}
    output_tokens = int(usage.get("output_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": int(details.get("thinking_tokens") or 0),
        "total_tokens": input_tokens + output_tokens,
    }


def usage_from_gemini(raw: Dict[str, Any]) -> Dict[str, int]:
    usage = raw.get("response", {}).get("usageMetadata", {})
    reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)
    candidate_tokens = int(usage.get("candidatesTokenCount") or 0)
    return {
        "input_tokens": int(usage.get("promptTokenCount") or 0),
        "output_tokens": candidate_tokens + reasoning_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": int(usage.get("totalTokenCount") or 0),
    }


def usage_from_openrouter(raw: Dict[str, Any]) -> Dict[str, float]:
    usage = raw.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": float(usage.get("prompt_tokens") or 0),
        "output_tokens": float(usage.get("completion_tokens") or 0),
        "reasoning_tokens": float(details.get("reasoning_tokens") or 0),
        "total_tokens": float(usage.get("total_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0.0),
    }


def iter_json_files(model_dir: Path) -> Iterable[Path]:
    return sorted(path for path in model_dir.glob("*.json") if path.is_file())


def cost_for_batch_model(model_dir: Path) -> Dict[str, Any]:
    model = model_dir.name
    pricing = BATCH_PRICES[model]
    totals: Dict[str, Any] = {
        "model": model,
        "display_model": display_name(model),
        "account": pricing["account"],
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "response_cost_usd": None,
    }

    if model.startswith("gpt-"):
        usage_reader = usage_from_openai
    elif model.startswith("claude-"):
        usage_reader = usage_from_anthropic
    elif model.startswith("gemini-"):
        usage_reader = usage_from_gemini
    else:
        raise ValueError(f"Unsupported batch model for cost extraction: {model}")

    for path in iter_json_files(model_dir):
        raw = json.loads(path.read_text(encoding="utf-8")).get("raw_response") or {}
        usage = usage_reader(raw)
        totals["requests"] += 1
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            totals[key] += usage[key]

    undiscounted = (
        totals["input_tokens"] / 1_000_000 * pricing["input"]
        + totals["output_tokens"] / 1_000_000 * pricing["output"]
    )
    totals["estimated_cost_usd"] = undiscounted * pricing["discount"]
    totals["discount_applied"] = pricing["discount"]
    totals["input_usd_per_1m"] = pricing["input"]
    totals["output_usd_per_1m"] = pricing["output"]
    return totals


def cost_for_openrouter_model(model_dir: Path) -> Dict[str, Any]:
    model = model_dir.name
    totals: Dict[str, Any] = {
        "model": model,
        "display_model": display_name(model),
        "account": "OpenRouter",
        "requests": 0,
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "reasoning_tokens": 0.0,
        "total_tokens": 0.0,
        "response_cost_usd": 0.0,
        "discount_applied": 1.0,
        "input_usd_per_1m": None,
        "output_usd_per_1m": None,
    }
    for path in iter_json_files(model_dir):
        raw = json.loads(path.read_text(encoding="utf-8")).get("raw_response") or {}
        usage = usage_from_openrouter(raw)
        totals["requests"] += 1
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            totals[key] += usage[key]
        totals["response_cost_usd"] += usage["cost_usd"]

    totals["estimated_cost_usd"] = totals["response_cost_usd"]
    return totals


def load_cost_rows() -> List[Dict[str, Any]]:
    rows = []
    for model_dir in sorted((BATCH_RESULTS_ROOT / "per_image").iterdir()):
        if model_dir.is_dir() and model_dir.name in BATCH_PRICES:
            rows.append(cost_for_batch_model(model_dir))
    for model_dir in sorted((OPENROUTER_RESULTS_ROOT / "per_image").iterdir()):
        if model_dir.is_dir():
            rows.append(cost_for_openrouter_model(model_dir))
    return sorted(rows, key=lambda row: row["estimated_cost_usd"], reverse=True)


def plot_metric_bars(rows: List[Dict[str, Any]], metric: str, title: str, path_stem: str) -> None:
    plot_rows = sorted(rows, key=lambda row: row.get(metric) or 0.0)
    labels = [row["display_model"] for row in plot_rows]
    values = [row.get(metric) or 0.0 for row in plot_rows]

    fig_height = max(5.5, 0.42 * len(labels))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    ax.barh(labels, values, color="#3b82f6")
    ax.set_xlabel(metric.upper() if metric != "label_accuracy" else "Label accuracy")
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.12 if values else 1)
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(value + max(values) * 0.015, index, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{path_stem}.{suffix}", dpi=180)
    plt.close(fig)


def plot_cost_vs_ap(metric_rows: List[Dict[str, Any]], cost_rows: List[Dict[str, Any]]) -> None:
    costs_by_model = {row["model"]: row for row in cost_rows}
    rows = [row for row in metric_rows if row["model"] in costs_by_model]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for row in rows:
        cost = costs_by_model[row["model"]]["estimated_cost_usd"]
        ap = row.get("ap") or 0.0
        account = costs_by_model[row["model"]]["account"]
        color = "#16a34a" if "Batch" in account else "#f97316"
        ax.scatter(cost, ap, s=54, color=color, alpha=0.85)
        ax.text(cost, ap, " " + display_name(row["model"]), fontsize=7, va="center")
    ax.set_xlabel("Estimated cost for 901 images (USD)")
    ax.set_ylabel("AP")
    ax.set_title("COCO-unseen API refresh: cost vs AP")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"coco_unseen_api_cost_vs_ap.{suffix}", dpi=180)
    plt.close(fig)


def summarize_costs_by_account(cost_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in cost_rows:
        account = row["account"]
        current = grouped.setdefault(
            account,
            {
                "account": account,
                "models": 0,
                "requests": 0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "reasoning_tokens": 0.0,
                "total_tokens": 0.0,
                "estimated_cost_usd": 0.0,
            },
        )
        current["models"] += 1
        current["requests"] += int(row["requests"])
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "estimated_cost_usd"):
            current[key] += float(row[key] or 0.0)
    return sorted(grouped.values(), key=lambda row: row["estimated_cost_usd"], reverse=True)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = load_metric_rows()
    metric_columns = [
        "dataset",
        "model",
        "display_model",
        "family",
        "eval_images",
        "eval_categories",
        "detections",
        "ap",
        "ap50",
        "label_accuracy",
        "latency_mean",
        "latency_median",
        "latency_p95",
    ]
    write_csv(METRICS_CSV, metric_rows, metric_columns)
    METRICS_MD.write_text(
        markdown_table(
            metric_rows,
            metric_columns,
            decimals={"ap": 4, "ap50": 4, "label_accuracy": 4, "latency_mean": 3, "latency_median": 3, "latency_p95": 3},
        ),
        encoding="utf-8",
    )

    cost_rows = load_cost_rows()
    cost_columns = [
        "model",
        "display_model",
        "account",
        "requests",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "input_usd_per_1m",
        "output_usd_per_1m",
        "discount_applied",
        "estimated_cost_usd",
        "response_cost_usd",
    ]
    write_csv(COSTS_CSV, cost_rows, cost_columns)
    cost_decimals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "input_usd_per_1m": 3,
        "output_usd_per_1m": 3,
        "discount_applied": 2,
        "estimated_cost_usd": 6,
        "response_cost_usd": 6,
    }
    COSTS_MD.write_text(
        markdown_table(cost_rows, cost_columns, decimals=cost_decimals),
        encoding="utf-8",
    )

    account_rows = summarize_costs_by_account(cost_rows)
    account_columns = [
        "account",
        "models",
        "requests",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "estimated_cost_usd",
    ]
    write_csv(ACCOUNT_COSTS_CSV, account_rows, account_columns)
    account_total = sum(float(row["estimated_cost_usd"]) for row in account_rows)
    account_markdown = markdown_table(account_rows, account_columns, decimals=cost_decimals)
    account_markdown += f"\nTotal estimated COCO-unseen API spend: ${account_total:.6f}\n"
    ACCOUNT_COSTS_MD.write_text(account_markdown, encoding="utf-8")

    plot_metric_bars(metric_rows, "ap", "COCO-unseen API refresh: AP", "coco_unseen_api_vlm_ap")
    plot_metric_bars(metric_rows, "ap50", "COCO-unseen API refresh: AP@0.5", "coco_unseen_api_vlm_ap50")
    plot_metric_bars(
        metric_rows,
        "label_accuracy",
        "COCO-unseen API refresh: dynamic-label accuracy",
        "coco_unseen_api_vlm_label_accuracy",
    )
    plot_cost_vs_ap(metric_rows, cost_rows)

    provider_totals: Dict[str, float] = {}
    for row in cost_rows:
        provider_totals[row["account"]] = provider_totals.get(row["account"], 0.0) + float(row["estimated_cost_usd"])

    print(f"wrote {METRICS_CSV}")
    print(f"wrote {METRICS_MD}")
    print(f"wrote {COSTS_CSV}")
    print(f"wrote {COSTS_MD}")
    print(f"wrote {ACCOUNT_COSTS_CSV}")
    print(f"wrote {ACCOUNT_COSTS_MD}")
    print("provider totals:")
    for provider, total in sorted(provider_totals.items()):
        print(f"  {provider}: ${total:.6f}")


if __name__ == "__main__":
    main()
