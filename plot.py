#!/usr/bin/env python3
"""results/ の JSON からマシン横断の比較グラフ (PNG) を生成する。

使い方:
  python plot.py                       # results/ を読み charts/ に PNG 出力
  python plot.py --results-dir results --out-dir charts

モデル別のグループ化棒グラフ (系列=マシン) を 4 指標分、1 枚の図にまとめる。
日本語フォントが無い環境でも文字化けしないよう、ラベルは ASCII で統一する。
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = {"tiny": 0, "base": 1, "small": 2, "medium": 3, "large": 4}

# (benchmark, metric_key, title, ylabel, log_y)
PANELS = [
    ("infer",    "rtf_avg",      "Inference: RTF (avg, lower is faster)", "RTF",            False),
    ("infer",    "speedup_avg",  "Inference: realtime speedup (higher is faster)", "x realtime", False),
    ("finetune", "avg_step_sec", "Fine-tuning: 1-step time (lower is faster)", "seconds/step", True),
    ("finetune", "rtf_30s_avg",  "Fine-tuning: RTF (30s-window basis)", "RTF",            True),
]


def load_records(results_dir):
    recs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            recs.append(json.load(f))
    return recs


def model_key(m):
    return (MODEL_ORDER.get(m, 99), m)


def draw_panel(ax, records, benchmark, metric_key, title, ylabel, log_y):
    recs = [r for r in records if r["benchmark"] == benchmark]
    machines = sorted({r["machine"]["label"] for r in recs})
    models = sorted({r["config"]["model"] for r in recs}, key=model_key)

    cell = {}
    for r in recs:
        cell[(r["machine"]["label"], r["config"]["model"])] = r["metrics"].get(metric_key)

    n = len(machines)
    width = 0.8 / max(n, 1)
    x = range(len(models))
    for i, mc in enumerate(machines):
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        vals = [cell.get((mc, m)) or 0 for m in models]
        bars = ax.bar(offsets, vals, width=width, label=mc)
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)


QUANT_ORDER = {"fp16": 0, "bf16": 1, "int8": 2, "int4": 3}

# Gemma 用パネル (benchmark=gemma): (metric_key, title, ylabel, log_y)
GEMMA_PANELS = [
    ("decode_tps", "Gemma: decode throughput (higher is faster)", "tok/s", False),
    ("ttft_ms",    "Gemma: TTFT (lower is faster)",               "ms",    False),
    ("vram_gb",    "Gemma: peak VRAM",                            "GB",    False),
]


def draw_gemma_panel(ax, records, metric_key, title, ylabel, log_y):
    recs = [r for r in records if r["benchmark"] == "gemma"]
    machines = sorted({r["machine"]["label"] for r in recs})

    def mq_key(t):
        model, quant = t
        return (model, QUANT_ORDER.get(quant, 99), quant)

    combos = sorted({(r["config"]["model"], r["config"]["quant"]) for r in recs}, key=mq_key)
    labels = [f"{m.replace('gemma-3-','').replace('-it','')}\n{q}" for m, q in combos]

    cell = {}
    for r in recs:
        cell[(r["machine"]["label"], r["config"]["model"], r["config"]["quant"])] = \
            r["metrics"].get(metric_key)

    n = len(machines)
    width = 0.8 / max(n, 1)
    x = range(len(combos))
    for i, mc in enumerate(machines):
        offsets = [xi + (i - (n - 1) / 2) * width for xi in x]
        vals = [cell.get((mc, m, q)) or 0 for m, q in combos]
        bars = ax.bar(offsets, vals, width=width, label=mc)
        ax.bar_label(bars, fmt="%.3g", fontsize=7, padding=2)

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser(description="ベンチ結果のマシン横断比較グラフ")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="charts")
    args = ap.parse_args()

    records = load_records(args.results_dir)
    if not records:
        print(f"no results in {args.results_dir}/")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Whisper (推論 / fine-tuning)
    if any(r["benchmark"] in ("infer", "finetune") for r in records):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (bench, key, title, ylabel, log_y) in zip(axes.flat, PANELS):
            draw_panel(ax, records, bench, key, title, ylabel, log_y)
        fig.suptitle("Whisper GPU Benchmark — machine comparison", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(args.out_dir, "benchmark_comparison.png")
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")

    # Gemma (テキスト生成 LLM, 条件別)
    if any(r["benchmark"] == "gemma" for r in records):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (key, title, ylabel, log_y) in zip(axes.flat, GEMMA_PANELS):
            draw_gemma_panel(ax, records, key, title, ylabel, log_y)
        fig.suptitle("Gemma inference Benchmark — by model/quant, machine comparison", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = os.path.join(args.out_dir, "gemma_comparison.png")
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
