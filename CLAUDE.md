# CLAUDE.md

このリポジトリで作業する Claude Code 向けのガイド。

## 目的

Whisper (openai-whisper) の GPU 性能を **RTF (Real-Time Factor = 処理時間 ÷ 音声長)** で
計測するベンチマーク。推論と軽量 fine-tuning の 2 種類を、複数マシン横断で比較できるよう
結果を JSON で `results/` に蓄積する。

## 全体構成

| ファイル | 役割 |
|---|---|
| `bench_common.py` | 共通基盤。環境メタデータ収集 (`collect_env`)、共通引数 (`base_arg_parser`)、JSON 保存 (`save_json`)。追加依存なし (torch + 標準ライブラリのみ)。 |
| `bench_infer.py` | 推論ベンチ。warmup 1 回を除外し `model.transcribe` を n_runs 回計測。fp16 (CUDA時)。 |
| `bench_finetune.py` | 軽量 fine-tuning ベンチ。単一サンプルへの過学習で 1 step (forward+backward+optimizer) を n_steps 回計測。AMP fp16 / AdamW lr=1e-5。loss 低下で重み更新を確認。 |
| `compare.py` | `results/*.json` を集計しマシン横断の比較表 (Markdown) を生成。`--out COMPARISON.md`。 |
| `plot.py` | `results/*.json` からマシン横断の比較グラフ (PNG) を生成。`charts/benchmark_comparison.png`。matplotlib 必要。 |
| `results/` | `<benchmark>_<machine>_<model>.json`。同一 (benchmark, machine, model) は最新で上書き。 |
| `jfk.flac` / `jfk.wav` | テスト音声 (16kHz mono, 11.000 秒)。`jfk.wav` は flac から ffmpeg 変換したもの。 |
| `README.md` | 人間向けの手順・結果サマリ。 |
| `COMPARISON.md` | `compare.py` が生成するマシン横断比較 (生成物)。手で編集しない。 |

## セットアップ

GPU アーキ依存があるため、各マシンで個別に環境を整える。

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU/arch に合わせる
pip install openai-whisper soundfile
```

- **ffmpeg が必須** (whisper の音声ロードが内部で呼ぶ)。システムに無く sudo も使えない場合は
  `pip install imageio-ffmpeg` でバイナリを取得し、`venv/bin/ffmpeg` に名前付きで配置 (シンボリックリンク)
  して PATH を通す回避策がある。
- `jfk.wav` が無ければ `ffmpeg -y -i jfk.flac -ar 16000 -ac 1 jfk.wav` で再生成。

## 実行

```bash
# 推論 / fine-tuning (model は tiny|base|small|medium|large、最後の数値は試行回数)
python bench_infer.py    jfk.wav large 10 --machine <ラベル>
python bench_finetune.py jfk.wav large 10 --machine <ラベル>

# 比較表を更新
python compare.py --out COMPARISON.md

# 比較グラフを更新 (matplotlib 必要: pip install matplotlib)
python plot.py
```

## マシンラベルの規則 (重要)

結果ファイル名・比較表の列はこのラベルで一意化される。**マシンごとに固有のラベルを必ず指定する。**

- 優先順位: `--machine` 引数 > 環境変数 `BENCH_MACHINE` > GPU 名からの自動 slug。
- 既存例: `pro6000-blackwell` (NVIDIA RTX PRO 6000 Blackwell, x86_64)。
- 別マシンで回すときは新しいラベル (例 GB10/DGX Spark 系なら `dgx-spark` など) を使い、
  既存マシンの結果を**上書きしないこと**。

## 計測上の約束ごと

- warmup を必ず除外し、CUDA では各計測の前後で `torch.cuda.synchronize()` して GPU 完了を待つ。
- 推論は fp16 (CUDA時)、CPU は fp16 非対応のため自動で fp32 + 警告。
- RTF が小さいほど速い。`speedup` (実時間比 x) は大きいほど速い。
- fine-tuning は `final_loss` が `warmup_loss` より下がっていれば重みが更新されている証拠。
- JSON には環境メタデータ (GPU 名 / capability / arch / torch・CUDA バージョン / 音声長) が必ず入る。
  異なる環境の数値を比較するときはこのメタデータも併せて確認する。

## 変更時の注意

- JSON スキーマを変えたら `bench_common.SCHEMA_VERSION` を上げ、`compare.py` の参照キー
  (`metrics` 内のキー名) と整合させる。
- 新しいメトリクスを比較表に出すには `compare.py` の `build(...)` 呼び出しに追加する。
- `.gitignore` で `venv/` `__pycache__/` `*.pyc` は除外済み。`results/*.json` はコミット対象 (蓄積する)。
