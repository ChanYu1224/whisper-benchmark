# Whisper GPU ベンチマーク (推論 / 軽量 fine-tuning)

RTF (Real-Time Factor = 処理時間 ÷ 音声長) を計測する。

## 環境
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120, 約97GB VRAM)
- ドライバ CUDA: 13.0 / PyTorch ビルド: cu128
- torch: 2.11.0+cu128 (`torch.cuda.is_available() == True`)
- openai-whisper / soundfile / ffmpeg 6.1.1
- テスト音声: `jfk.wav` (16kHz mono, **11.000 秒**)

## 構築・実行
```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell(sm_120)対応
pip install openai-whisper soundfile
ffmpeg -y -i jfk.flac -ar 16000 -ac 1 jfk.wav                          # テスト音声

python bench_infer.py    jfk.wav <tiny|base|small|large> 10
python bench_finetune.py jfk.wav <tiny|base|small|large> 10
```

## 結果 (音声長 11.000 秒, warmup 除外 / `torch.cuda.synchronize()` で同期)

### 推論 (fp16)
| model | avg 時間 | RTF(avg) | 実時間比 | best RTF |
|-------|---------|----------|---------|----------|
| tiny  | 0.097 s | 0.0088   | 113.8x  | 0.0083   |
| base  | 0.117 s | 0.0107   |  93.8x  | 0.0104   |
| small | 0.172 s | 0.0156   |  64.0x  | 0.0153   |
| large | 0.360 s | 0.0327   |  30.6x  | 0.0321   |

### fine-tuning (AMP fp16, AdamW lr=1e-5, 1 sample 過学習)
| model | 1step avg | RTF(音声11s基準) | RTF(30秒窓基準) | warmup_loss → final_loss |
|-------|-----------|------------------|-----------------|--------------------------|
| tiny  | 0.0193 s  | 0.0018 | 0.0006 | 0.5288 → 0.0725 |
| base  | 0.0253 s  | 0.0023 | 0.0008 | 0.4697 → 0.0231 |
| small | 0.0469 s  | 0.0043 | 0.0016 | 0.4619 → 0.0053 |
| large | 0.1945 s  | 0.0177 | 0.0065 | 0.4530 → 0.0002 |

VRAM が潤沢 (97GB) なためスキップしたモデルは無し。全モデルで loss が単調に低下しており、重みが実際に更新されている。
