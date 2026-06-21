#!/usr/bin/env python3
"""Whisper 推論ベンチマーク。

使い方: python bench_infer.py <wav> <model_name> <n_runs>
RTF = 処理時間 / 音声長 を avg/best で出力する。
"""
import sys
import time

import soundfile as sf
import torch
import whisper


def main():
    if len(sys.argv) != 4:
        print("usage: python bench_infer.py <wav> <model_name> <n_runs>")
        sys.exit(1)

    wav = sys.argv[1]
    model_name = sys.argv[2]
    n_runs = int(sys.argv[3])

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    fp16 = use_cuda  # CPU では fp16 非対応

    # 音声長
    data, sr = sf.read(wav)
    audio_sec = len(data) / sr

    print(f"=== bench_infer: model={model_name} device={device} fp16={fp16} ===")
    print(f"audio: {wav} duration={audio_sec:.3f}s")
    if not use_cuda:
        print("[WARN] CUDA が使えないため CPU で実行します。")

    # モデルロード
    model = whisper.load_model(model_name, device=device)

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    decode_opts = dict(fp16=fp16, language="en", without_timestamps=True)

    # warmup (計測から除外)
    sync()
    result = model.transcribe(wav, **decode_opts)
    sync()

    # 計測
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
    print("--- result ---")
    print(f"avg time = {avg:.4f}s  RTF(avg) = {avg/audio_sec:.4f}  ({audio_sec/avg:.2f}x realtime)")
    print(f"best time= {best:.4f}s  RTF(best)= {best/audio_sec:.4f}  ({audio_sec/best:.2f}x realtime)")
    print(f"text: {result['text'].strip()}")

    # 機械可読サマリ (後段で集計しやすいよう1行で)
    print(f"SUMMARY infer model={model_name} audio={audio_sec:.3f} "
          f"avg={avg:.4f} best={best:.4f} rtf_avg={avg/audio_sec:.4f} "
          f"rtf_best={best/audio_sec:.4f} speedup_avg={audio_sec/avg:.2f} speedup_best={audio_sec/best:.2f}")


if __name__ == "__main__":
    main()
