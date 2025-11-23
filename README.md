# PII Entity Recognition for Noisy STT Transcripts

This project implements a token-level Named Entity Recognition (NER) model to detect Personally Identifiable Information (PII) in noisy Speech-to-Text (STT) transcripts.

- Python 3.11 recommended

## Results

| Metric | Value | Target | Result |
| :--- | :--- | :--- | :--- |
| **PII F1 Score** | **1.00** | $\ge$ 0.80 | ✅ Exceeded |
| **p95 Latency** | **10.74 ms** | $\le$ 20 ms | ✅ Exceeded |
| **Inference Device** | CPU | CPU | - |

## Approach

1.  **Data Engineering**: The provided training data was insufficient (2 examples). I wrote a custom generator (`generate_data.py`) to create 1,000 synthetic training samples that mimic STT errors (spoken punctuation like "dot", spoken numbers, missing casing).
2.  **Modeling**: Fine-tuned `distilbert-base-uncased` for token classification.
3.  **Latency Optimization**: Post-training, the model was converted to **TorchScript** with **Dynamic Quantization (int8)**. This reduced inference latency by ~50%, bringing it well under the 20ms budget.

## Quick Start (for the user if they want to train the model themself and test it)

### 1\. Setup
```bash
pip install -r requirements.txt
```

### 2\. Generate Data

Create the synthetic training and development datasets:

```bash
python generate_data.py
```

### 3\. Train Model

Train the baseline DistilBERT model:

```bash
python src/train.py --model_name distilbert-base-uncased --epochs 3
```

### 4\. Quantize (Latency Optimization)

Compress the model for fast CPU inference:

```bash
python src/quantize.py
```

### 5\. Run Inference & Metrics

Generate predictions on the dev set:

```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
```

Calculate F1 scores:

```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

Measure Latency (using the quantized model):

```bash
python src/measure_latency.py --model_dir out_quantized --runs 100
```

## 📂 File Structure

  * `generate_data.py`: Generates synthetic noisy STT data.
  * `src/quantize.py`: Performs int8 dynamic quantization and TorchScript tracing.
  * `src/train.py`: Main training loop.
  * `out/`: Standard model artifacts (model file not included in repo due to size). You can download the `model.safetensors` using this link - *(gdrive link of the model)*
  * `out_quantized/`: Optimized model artifacts (model file not included in repo due to size). You can download the `model.safetensors` using this link - *(gdrive link of the model)*
  * `out/dev_pred.json`: Final prediction output.
  * `out/test_pred.json`: Final test prediction output.