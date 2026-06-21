"""ベンチマーク共通: 環境メタデータ収集 と JSON 結果保存。

マシン横断 (PRO 6000 Blackwell / DGX Spark / RTX 5070 ...) で結果を比較できるよう、
各実行の環境情報と計測値を 1 つの JSON にまとめて results/ に保存する。
追加の依存は持たない (torch / 標準ライブラリのみ)。
"""
import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime

import torch

SCHEMA_VERSION = 1


def _slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _driver_cuda_version() -> str | None:
    """nvidia-smi のヘッダから "CUDA Version: X.Y" を best-effort で取得。"""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def _driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return out[0].strip() if out else None
    except Exception:
        return None


def collect_env(machine_label: str | None) -> dict:
    """環境メタデータを収集して dict で返す。"""
    use_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None
    cap = torch.cuda.get_device_capability(0) if use_cuda else None

    if machine_label is None:
        machine_label = os.environ.get("BENCH_MACHINE")
    if machine_label is None:
        machine_label = _slug(gpu_name) if gpu_name else _slug(platform.node() or "cpu")

    return {
        "machine": {
            "label": machine_label,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "arch": platform.machine(),          # x86_64 / aarch64 など
            "python": platform.python_version(),
        },
        "gpu": {
            "available": use_cuda,
            "device": "cuda" if use_cuda else "cpu",
            "name": gpu_name,
            "capability": f"{cap[0]}.{cap[1]}" if cap else None,
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "driver_version": _driver_version() if use_cuda else None,
            "driver_cuda_version": _driver_cuda_version() if use_cuda else None,
        },
    }


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """全ベンチ共通の引数。位置引数 wav / model_name / n は呼び出し側で追加する。"""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--machine", default=None,
                   help="マシン識別ラベル (例: pro6000-blackwell, dgx-spark, rtx-5070)。"
                        "未指定なら環境変数 BENCH_MACHINE か GPU 名から自動生成。")
    p.add_argument("--out-dir", default="results", help="JSON 出力先ディレクトリ (default: results)")
    p.add_argument("--no-json", action="store_true", help="JSON を保存しない")
    return p


def save_json(record: dict, benchmark: str, machine_label: str, model: str,
              out_dir: str) -> str:
    """results/<benchmark>_<machine>_<model>.json に保存しパスを返す。

    同一 (benchmark, machine, model) は最新結果で上書き (比較表が一意になる)。
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{benchmark}_{_slug(machine_label)}_{_slug(model)}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
