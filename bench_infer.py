#!/usr/bin/env python3
"""Whisper 推論ベンチマーク。

使い方: python bench_infer.py <wav> <model_name> <n_runs> [--machine LABEL] [--out-dir DIR] [--no-json]
RTF = 処理時間 / 音声長 を avg/best で出力し、環境情報込みで JSON 保存する。
"""
import time

import soundfile as sf
import torch
import whisper

import bench_common


def main():
    parser = bench_common.base_arg_parser("Whisper 推論ベンチマーク")
    parser.add_argument("wav")
    parser.add_argument("model_name")
    parser.add_argument("n_runs", type=int)
    args = parser.parse_args()

    wav, model_name, n_runs = args.wav, args.model_name, args.n_runs

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    fp16 = use_cuda  # CPU では fp16 非対応

    data, sr = sf.read(wav)
    audio_sec = len(data) / sr

    env = bench_common.collect_env(args.machine)
    print(f"=== bench_infer: model={model_name} device={device} fp16={fp16} "
          f"machine={env['machine']['label']} ===")
    print(f"audio: {wav} duration={audio_sec:.3f}s")
    if not use_cuda:
        print("[WARN] CUDA が使えないため CPU で実行します。")

    model = whisper.load_model(model_name, device=device)

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    decode_opts = dict(fp16=fp16, language="en", without_timestamps=True)

    # warmup (計測から除外)
    sync()
    result = model.transcribe(wav, **decode_opts)
    sync()

    times = []
    for i in range(n_runs):
        sync()
        t0 = time.perf_counter()
        result = model.transcribe(wav, **decode_opts)
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  run {i+1}/{n_runs}: {dt:.4f}s  RTF={dt/audio_sec:.4f}")

    avg = sum(times) / len(times)
    best = min(times)
    text = result["text"].strip()
    print("--- result ---")
    print(f"avg time = {avg:.4f}s  RTF(avg) = {avg/audio_sec:.4f}  ({audio_sec/avg:.2f}x realtime)")
    print(f"best time= {best:.4f}s  RTF(best)= {best/audio_sec:.4f}  ({audio_sec/best:.2f}x realtime)")
    print(f"text: {text}")

    if not args.no_json:
        record = {
            "schema_version": bench_common.SCHEMA_VERSION,
            "benchmark": "infer",
            "timestamp": bench_common.now_iso(),
            **env,
            "audio": {"path": wav, "duration_sec": round(audio_sec, 3), "sample_rate": sr},
            "config": {"model": model_name, "fp16": fp16, "n_runs": n_runs},
            "metrics": {
                "times_sec": [round(t, 6) for t in times],
                "avg_sec": round(avg, 6),
                "best_sec": round(best, 6),
                "rtf_avg": round(avg / audio_sec, 6),
                "rtf_best": round(best / audio_sec, 6),
                "speedup_avg": round(audio_sec / avg, 3),
                "speedup_best": round(audio_sec / best, 3),
                "text": text,
            },
        }
        path = bench_common.save_json(record, "infer", env["machine"]["label"],
                                      model_name, args.out_dir)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
