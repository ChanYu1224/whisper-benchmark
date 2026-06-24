#!/usr/bin/env python3
"""Gemma 推論 (テキスト生成) ベンチマーク。

LLM のテキスト生成はスループット指標で測る:
  - TTFT (Time To First Token): プロンプト投入から最初の生成トークンまで (= prefill レイテンシ)
  - decode throughput: 生成フェーズの tokens/sec (= 1トークンあたり生成速度)
  - end-to-end throughput: 全体の tokens/sec

fp16 / bf16 / int8 / int4 など複数条件 (--quant) を切り替えて比較できる。
環境情報込みで results/ に JSON 保存し、compare.py でマシン横断比較する。

使い方:
  python bench_gemma.py <model_id> [--quant fp16|bf16|int8|int4]
      [--max-new-tokens 256] [--n-runs 5] [--prompt "..."] [--machine LABEL]
モデル例: google/gemma-3-4b-it, google/gemma-3-27b-it
"""
import time

import torch

import bench_common

DEFAULT_PROMPT = "Explain how a transformer neural network works, in about 200 words."


def build_model(model_id, quant):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)

    kwargs = {"device_map": "cuda"}
    if quant == "fp16":
        kwargs["dtype"] = torch.float16
    elif quant == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["dtype"] = torch.float16
    elif quant == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        raise ValueError(f"unknown quant: {quant}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model


def main():
    parser = bench_common.base_arg_parser("Gemma 推論 (テキスト生成) ベンチマーク")
    parser.add_argument("model_id")
    parser.add_argument("--quant", default="fp16",
                        choices=["fp16", "bf16", "int8", "int4"],
                        help="計測条件 (dtype / 量子化)。default: fp16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] このベンチは CUDA 前提です。")
        raise SystemExit(1)

    env = bench_common.collect_env(args.machine)
    short_model = args.model_id.split("/")[-1]
    print(f"=== bench_gemma: model={args.model_id} quant={args.quant} "
          f"machine={env['machine']['label']} ===")

    tok, model = build_model(args.model_id, args.quant)

    # chat テンプレートでプロンプト整形
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     tokenize=True, return_dict=True,
                                     return_tensors="pt").to("cuda")
    prompt_tokens = inputs["input_ids"].shape[1]
    print(f"prompt_tokens={prompt_tokens} max_new_tokens={args.max_new_tokens}")

    gen_kwargs = dict(do_sample=False, use_cache=True,
                      pad_token_id=tok.pad_token_id or tok.eos_token_id)

    def sync():
        torch.cuda.synchronize()

    @torch.inference_mode()
    def generate(n_new):
        return model.generate(**inputs, max_new_tokens=n_new, min_new_tokens=n_new,
                              **gen_kwargs)

    # warmup (計測から除外)
    sync(); generate(args.max_new_tokens); sync()

    # --- TTFT: 1トークンだけ生成 = prefill レイテンシ ---
    ttfts = []
    for _ in range(args.n_runs):
        sync(); t0 = time.perf_counter()
        generate(1)
        sync(); ttfts.append(time.perf_counter() - t0)
    ttft = min(ttfts)

    # --- end-to-end & decode throughput ---
    totals = []
    for i in range(args.n_runs):
        sync(); t0 = time.perf_counter()
        out = generate(args.max_new_tokens)
        sync(); dt = time.perf_counter() - t0
        totals.append(dt)
        new_tokens = out.shape[1] - prompt_tokens
        print(f"  run {i+1}/{args.n_runs}: {dt:.4f}s  new_tokens={new_tokens}  "
              f"e2e_tps={new_tokens/dt:.2f}")

    avg_total = sum(totals) / len(totals)
    best_total = min(totals)
    new_tokens = args.max_new_tokens
    # decode tps: 全体時間から prefill(TTFT) を除いた残りで (生成トークン-1) を割る
    decode_tps = (new_tokens - 1) / max(best_total - ttft, 1e-9)
    e2e_tps_avg = new_tokens / avg_total
    e2e_tps_best = new_tokens / best_total

    print("--- result ---")
    print(f"TTFT (best)        = {ttft*1000:.2f} ms")
    print(f"decode throughput  = {decode_tps:.2f} tok/s")
    print(f"e2e throughput avg = {e2e_tps_avg:.2f} tok/s  best = {e2e_tps_best:.2f} tok/s")
    print(f"total time avg     = {avg_total:.4f}s  best = {best_total:.4f}s")

    # VRAM 使用量
    vram_gb = torch.cuda.max_memory_allocated() / 1e9

    if not args.no_json:
        record = {
            "schema_version": bench_common.SCHEMA_VERSION,
            "benchmark": "gemma",
            "timestamp": bench_common.now_iso(),
            **env,
            "config": {
                "model": short_model,
                "model_id": args.model_id,
                "quant": args.quant,
                "max_new_tokens": args.max_new_tokens,
                "n_runs": args.n_runs,
                "prompt_tokens": prompt_tokens,
            },
            "metrics": {
                "ttft_ms": round(ttft * 1000, 3),
                "decode_tps": round(decode_tps, 3),
                "e2e_tps_avg": round(e2e_tps_avg, 3),
                "e2e_tps_best": round(e2e_tps_best, 3),
                "total_avg_sec": round(avg_total, 6),
                "total_best_sec": round(best_total, 6),
                "vram_gb": round(vram_gb, 3),
            },
        }
        # variant に quant を入れて条件ごとに別ファイル保存
        path = bench_common.save_json(record, "gemma", env["machine"]["label"],
                                      short_model, args.out_dir, variant=args.quant)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
