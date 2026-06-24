# ベンチマーク比較

## マシン

| label | GPU · capability · arch · torch · CUDA(driver) |
|---|---|
| dgx-spark-gb10 | NVIDIA GB10 · cap 12.1 · aarch64 · torch 2.11.0+cu128 · CUDA(drv) 13.0 |
| pro6000-blackwell | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition · cap 12.0 · x86_64 · torch 2.11.0+cu128 · CUDA(drv) 13.0 |

## 推論 (RTF = 処理時間 / 音声長, 小さいほど速い)

### infer: RTF (avg)

| model | dgx-spark-gb10 | pro6000-blackwell |
|---|---|---|
| tiny | 0.0059 | 0.0087 |
| base | 0.0092 | 0.0106 |
| small | 0.0195 | 0.0158 |
| large | 0.0913 | 0.0327 |

### infer: 実時間比 (x, 大きいほど速い)

| model | dgx-spark-gb10 | pro6000-blackwell |
|---|---|---|
| tiny | 169.6270 | 114.5090 |
| base | 108.7530 | 94.7340 |
| small | 51.2340 | 63.1100 |
| large | 10.9580 | 30.6080 |

## fine-tuning (1step)

### finetune: 1step 時間 (s)

| model | dgx-spark-gb10 | pro6000-blackwell |
|---|---|---|
| tiny | 0.0614 | 0.0158 |
| base | 0.0902 | 0.0252 |
| small | 0.2489 | 0.0449 |
| large | 1.5957 | 0.1940 |

### finetune: RTF (30秒窓基準)

| model | dgx-spark-gb10 | pro6000-blackwell |
|---|---|---|
| tiny | 0.0020 | 0.0005 |
| base | 0.0030 | 0.0008 |
| small | 0.0083 | 0.0015 |
| large | 0.0532 | 0.0065 |

### finetune: final_loss

| model | dgx-spark-gb10 | pro6000-blackwell |
|---|---|---|
| tiny | 0.0724 | 0.0730 |
| base | 0.0235 | 0.0231 |
| small | 0.0053 | 0.0053 |
| large | 0.0002 | 0.0002 |

## Gemma 推論 (テキスト生成, 条件別)

### gemma: decode throughput (tok/s, 大きいほど速い)

| model | quant | pro6000-blackwell |
|---|---|---|
| gemma-3-27b-it | fp16 | 18.44 |
| gemma-3-27b-it | bf16 | 18.09 |
| gemma-3-27b-it | int8 | 7.50 |
| gemma-3-27b-it | int4 | 22.96 |
| gemma-3-4b-it | fp16 | 58.80 |
| gemma-3-4b-it | bf16 | 58.29 |
| gemma-3-4b-it | int8 | 14.78 |
| gemma-3-4b-it | int4 | 41.25 |

### gemma: end-to-end throughput (tok/s)

| model | quant | pro6000-blackwell |
|---|---|---|
| gemma-3-27b-it | fp16 | 18.43 |
| gemma-3-27b-it | bf16 | 18.08 |
| gemma-3-27b-it | int8 | 7.49 |
| gemma-3-27b-it | int4 | 22.68 |
| gemma-3-4b-it | fp16 | 58.76 |
| gemma-3-4b-it | bf16 | 58.26 |
| gemma-3-4b-it | int8 | 14.77 |
| gemma-3-4b-it | int4 | 41.18 |

### gemma: TTFT (ms, 小さいほど速い)

| model | quant | pro6000-blackwell |
|---|---|---|
| gemma-3-27b-it | fp16 | 57.97 |
| gemma-3-27b-it | bf16 | 61.73 |
| gemma-3-27b-it | int8 | 189.13 |
| gemma-3-27b-it | int4 | 181.28 |
| gemma-3-4b-it | fp16 | 19.71 |
| gemma-3-4b-it | bf16 | 19.45 |
| gemma-3-4b-it | int8 | 82.14 |
| gemma-3-4b-it | int4 | 34.74 |

### gemma: VRAM (GB)

| model | quant | pro6000-blackwell |
|---|---|---|
| gemma-3-27b-it | fp16 | 55.08 |
| gemma-3-27b-it | bf16 | 55.08 |
| gemma-3-27b-it | int8 | 29.87 |
| gemma-3-27b-it | int4 | 17.99 |
| gemma-3-4b-it | fp16 | 8.66 |
| gemma-3-4b-it | bf16 | 8.66 |
| gemma-3-4b-it | int8 | 5.08 |
| gemma-3-4b-it | int4 | 3.53 |

