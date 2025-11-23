import json
import time
import argparse
import statistics
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    # OPTIMIZATION: Reduce thread contention for batch_size=1
    torch.set_num_threads(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out") # Can be "out" or "out_quantized"
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--runs", type=int, default=100) # Increased runs for stability
    ap.add_argument("--device", default="cpu") # Force CPU for this test
    args = ap.parse_args()

    print(f"Loading tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Check if we have a quantized TorchScript model or a standard HF model
    quantized_path = os.path.join(args.model_dir, "model.pt")
    
    if os.path.exists(quantized_path):
        print(f"Detected Quantized TorchScript model at {quantized_path}")
        model = torch.jit.load(quantized_path)
        is_torchscript = True
    else:
        print(f"Loading standard HF model from {args.model_dir}")
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
        is_torchscript = False

    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    if not texts:
        print("No texts found.")
        return

    print("Warming up...")
    # Warmup
    for _ in range(10):
        t = texts[0]
        enc = tokenizer(t, truncation=True, max_length=args.max_length, return_tensors="pt")
        inputs = enc["input_ids"].to(args.device)
        mask = enc["attention_mask"].to(args.device)
        with torch.no_grad():
            if is_torchscript:
                _ = model(inputs, mask)
            else:
                _ = model(input_ids=inputs, attention_mask=mask)

    print(f"Measuring latency over {args.runs} runs...")
    times_ms = []

    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(t, truncation=True, max_length=args.max_length, return_tensors="pt")
        inputs = enc["input_ids"].to(args.device)
        mask = enc["attention_mask"].to(args.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            if is_torchscript:
                _ = model(inputs, mask)
            else:
                _ = model(input_ids=inputs, attention_mask=mask)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    p50 = statistics.median(times_ms)
    times_sorted = sorted(times_ms)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]

    print(f"Latency stats (batch_size=1, threads=1):")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()