#!/usr/bin/env python3
"""results/ 内の JSON を集計し、マシン横断の比較表 (Markdown) を出力する。

使い方:
  python compare.py                 # results/ を読み標準出力に表示
  python compare.py --out COMPARISON.md
  python compare.py --results-dir results
"""
import argparse
import glob
import json
import os

MODEL_ORDER = {"tiny": 0, "base": 1, "small": 2, "medium": 3, "large": 4}


def load_records(results_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    return records


def model_key(m):
    return (MODEL_ORDER.get(m, 99), m)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def machine_header(records):
    """machine label -> 1行サマリ (GPU/arch/torch)。"""
    info = {}
    for r in records:
        lbl = r["machine"]["label"]
        if lbl not in info:
            g, m = r["gpu"], r["machine"]
            info[lbl] = (f"{g.get('name')} · cap {g.get('capability')} · "
                         f"{m.get('arch')} · torch {g.get('torch_version')} · "
                         f"CUDA(drv) {g.get('driver_cuda_version')}")
    return info


def build(records, benchmark, metric_key, metric_label):
    recs = [r for r in records if r["benchmark"] == benchmark]
    if not recs:
        return None
    machines = sorted({r["machine"]["label"] for r in recs})
    models = sorted({r["config"]["model"] for r in recs}, key=model_key)

    # (machine, model) -> value
    cell = {}
    for r in recs:
        cell[(r["machine"]["label"], r["config"]["model"])] = r["metrics"].get(metric_key)

    headers = ["model"] + machines
    rows = []
    for model in models:
        row = [model]
        for mc in machines:
            v = cell.get((mc, model))
            row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
        rows.append(row)
    return f"### {benchmark}: {metric_label}\n\n" + md_table(headers, rows)


QUANT_ORDER = {"fp16": 0, "bf16": 1, "int8": 2, "int4": 3}


def build_gemma(records, metric_key, metric_label):
    """Gemma: 行 = (model, quant), 列 = machine。"""
    recs = [r for r in records if r["benchmark"] == "gemma"]
    if not recs:
        return None
    machines = sorted({r["machine"]["label"] for r in recs})

    def mq_key(t):
        model, quant = t
        return (model, QUANT_ORDER.get(quant, 99), quant)

    mqs = sorted({(r["config"]["model"], r["config"]["quant"]) for r in recs}, key=mq_key)

    cell = {}
    for r in recs:
        cell[(r["machine"]["label"], r["config"]["model"], r["config"]["quant"])] = \
            r["metrics"].get(metric_key)

    headers = ["model", "quant"] + machines
    rows = []
    for model, quant in mqs:
        row = [model, quant]
        for mc in machines:
            v = cell.get((mc, model, quant))
            row.append(f"{v:.2f}" if isinstance(v, (int, float)) else "-")
        rows.append(row)
    return f"### gemma: {metric_label}\n\n" + md_table(headers, rows)


def main():
    ap = argparse.ArgumentParser(description="ベンチ結果のマシン横断比較")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="出力先 Markdown (省略時は標準出力)")
    args = ap.parse_args()

    records = load_records(args.results_dir)
    if not records:
        print(f"no results in {args.results_dir}/")
        return

    sections = ["# ベンチマーク比較\n"]

    # マシン凡例
    info = machine_header(records)
    sections.append("## マシン\n")
    sections.append(md_table(
        ["label", "GPU · capability · arch · torch · CUDA(driver)"],
        [[lbl, desc] for lbl, desc in sorted(info.items())]))
    sections.append("")

    # 推論
    sections.append("## 推論 (RTF = 処理時間 / 音声長, 小さいほど速い)\n")
    for key, lbl in [("rtf_avg", "RTF (avg)"), ("speedup_avg", "実時間比 (x, 大きいほど速い)")]:
        t = build(records, "infer", key, lbl)
        if t:
            sections.append(t + "\n")

    # fine-tuning
    sections.append("## fine-tuning (1step)\n")
    for key, lbl in [("avg_step_sec", "1step 時間 (s)"),
                     ("rtf_30s_avg", "RTF (30秒窓基準)"),
                     ("final_loss", "final_loss")]:
        t = build(records, "finetune", key, lbl)
        if t:
            sections.append(t + "\n")

    # Gemma (テキスト生成 LLM)
    if any(r["benchmark"] == "gemma" for r in records):
        sections.append("## Gemma 推論 (テキスト生成, 条件別)\n")
        for key, lbl in [("decode_tps", "decode throughput (tok/s, 大きいほど速い)"),
                         ("e2e_tps_best", "end-to-end throughput (tok/s)"),
                         ("ttft_ms", "TTFT (ms, 小さいほど速い)"),
                         ("vram_gb", "VRAM (GB)")]:
            t = build_gemma(records, key, lbl)
            if t:
                sections.append(t + "\n")

    text = "\n".join(sections)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
