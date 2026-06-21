#!/usr/bin/env python3
"""Whisper 軽量 fine-tuning ベンチマーク (openai-whisper を直接学習ループで回す)。

使い方: python bench_finetune.py <wav> <model_name> <n_steps> [--machine LABEL] [--out-dir DIR] [--no-json]
1 step = forward + backward + optimizer 更新。warmup 1step を除外して計測。
RTF = (1stepあたり時間) / 音声長。あわせて 30秒窓基準の RTF も併記。
final_loss で重みが更新されている (loss が下がる) ことを確認する。環境情報込みで JSON 保存。
"""
import time

import soundfile as sf
import torch
import torch.nn.functional as F
import whisper
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import get_tokenizer

import bench_common

LR = 1e-5


def main():
    parser = bench_common.base_arg_parser("Whisper 軽量 fine-tuning ベンチマーク")
    parser.add_argument("wav")
    parser.add_argument("model_name")
    parser.add_argument("n_steps", type=int)
    args = parser.parse_args()

    wav, model_name, n_steps = args.wav, args.model_name, args.n_steps

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    amp = use_cuda  # AMP(fp16) は CUDA のみ

    data, sr = sf.read(wav)
    audio_sec = len(data) / sr

    env = bench_common.collect_env(args.machine)
    print(f"=== bench_finetune: model={model_name} device={device} amp(fp16)={amp} "
          f"machine={env['machine']['label']} ===")
    print(f"audio: {wav} duration={audio_sec:.3f}s")
    if not use_cuda:
        print("[WARN] CUDA が使えないため CPU で実行します。")

    # モデルは fp32 で読み込み (AMP は autocast 側で fp16 化、重みは fp32 維持)
    model = whisper.load_model(model_name, device=device)
    model.train()

    # --- 参照文を用意 (単一サンプルへ過学習させるためのターゲット) ---
    with torch.no_grad():
        ref_text = model.transcribe(wav, fp16=amp, language="en",
                                    without_timestamps=True)["text"].strip()
    model.train()
    print(f"reference text: {ref_text}")

    # --- メル作成 ---
    n_mels = model.dims.n_mels
    mel = log_mel_spectrogram(wav, n_mels=n_mels)
    mel = pad_or_trim(mel, N_FRAMES)            # (n_mels, 3000)
    mel = mel.unsqueeze(0).to(device)           # (1, n_mels, 3000)

    # --- ターゲットトークン ---
    tokenizer = get_tokenizer(
        multilingual=model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )
    text_tokens = tokenizer.encode(ref_text)
    tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens + [tokenizer.eot]
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    dec_in = tokens[:, :-1]
    target = tokens[:, 1:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(mel, dec_in)                       # (1, T-1, vocab)
            loss = F.cross_entropy(logits.transpose(1, 2), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    # warmup (計測から除外)
    first_loss = step()
    sync()

    times = []
    losses = []
    last_loss = first_loss
    for i in range(n_steps):
        sync()
        t0 = time.perf_counter()
        last_loss = step()
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(last_loss)
        print(f"  step {i+1}/{n_steps}: {dt:.4f}s  loss={last_loss:.4f}")

    avg = sum(times) / len(times)
    best = min(times)
    print("--- result ---")
    print(f"warmup_loss(=step0) = {first_loss:.4f}  final_loss = {last_loss:.4f}")
    print(f"avg step time = {avg:.4f}s  best = {best:.4f}s")
    print(f"RTF(audio {audio_sec:.1f}s 基準) avg = {avg/audio_sec:.4f}  best = {best/audio_sec:.4f}")
    print(f"RTF(30秒窓 基準)            avg = {avg/30.0:.4f}  best = {best/30.0:.4f}")

    if not args.no_json:
        record = {
            "schema_version": bench_common.SCHEMA_VERSION,
            "benchmark": "finetune",
            "timestamp": bench_common.now_iso(),
            **env,
            "audio": {"path": wav, "duration_sec": round(audio_sec, 3), "sample_rate": sr},
            "config": {"model": model_name, "amp_fp16": amp, "n_steps": n_steps,
                       "optimizer": "AdamW", "lr": LR},
            "metrics": {
                "step_times_sec": [round(t, 6) for t in times],
                "losses": [round(l, 6) for l in losses],
                "avg_step_sec": round(avg, 6),
                "best_step_sec": round(best, 6),
                "rtf_audio_avg": round(avg / audio_sec, 6),
                "rtf_audio_best": round(best / audio_sec, 6),
                "rtf_30s_avg": round(avg / 30.0, 6),
                "rtf_30s_best": round(best / 30.0, 6),
                "warmup_loss": round(first_loss, 6),
                "final_loss": round(last_loss, 6),
            },
        }
        path = bench_common.save_json(record, "finetune", env["machine"]["label"],
                                      model_name, args.out_dir)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
