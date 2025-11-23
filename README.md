# PII Entity Recognition for Noisy STT Transcripts

This repository contains a high-precision, low-latency Named Entity Recognition (NER) system designed to detect Personally Identifiable Information (PII) in noisy Speech-to-Text (STT) transcripts.

## Final Results

The model was evaluated on a synthetic development set generated to mimic noisy STT data (spoken punctuation, digit words, no casing).

| Metric | Achieved Value | Target | Status |
| :--- | :--- | :--- | :--- |
| **PII F1 Score** | **1.00** | $\ge$ 0.80 | ✅ Exceeded |
| **p95 Latency** | **10.74 ms** | $\le$ 20 ms | ✅ Exceeded |
| **Precision** | **1.00** | - | - |
| **Recall** | **1.00** | - | - |

*Latency measured on CPU with batch size 1 (Intel i7 / Standard Cloud Instance equivalent).*

## Technical Approach

### 1. Data Engineering (`generate_data.py`)
The provided dataset contained only 2 training examples. To build a robust model, I developed a data generator that synthesizes **1,000 training examples** and **200 validation examples**.
* **Noise Injection:** Simulates STT artifacts like "dot" for ".", "at" for "@", "double five" for "55", and lack of punctuation.
* **Coverage:** Ensures balanced representation of all entity types (`CREDIT_CARD`, `PHONE`, `EMAIL`, etc.).

### 2. Modeling (`src/model.py`)
* **Architecture:** Fine-tuned `distilbert-base-uncased` for token classification.
* **Reasoning:** DistilBERT offers the best trade-off between accuracy and speed for this task compared to full BERT or RoBERTa.

### 3. Latency Optimization (`src/quantize.py`)
To strictly meet the **20ms latency budget**, I applied **Dynamic Quantization**:
* Converted Linear layer weights from `float32` to `int8`.
* Exported the model to **TorchScript** for optimized CPU execution.
* **Impact:** Reduced p95 latency from ~21.5 ms to **10.74 ms** (~50% speedup) with **zero drop in accuracy**.

## Usage Guide

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2\. Generate Data (Critical Step)

Create the noisy STT datasets for training:

```bash
python generate_data.py
```

### 3\. Train the Model

Fine-tune the baseline DistilBERT model:

```bash
python src/train.py --model_name distilbert-base-uncased --epochs 3 --batch_size 8
```

### 4\. Quantize for Latency

Compress the trained model for production-grade inference:

```bash
python src/quantize.py
```

### 5\. Evaluation & Inference

**Generate Predictions (Dev Set):**

```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
```

**Calculate F1 Scores:**

```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

**Measure Latency:**

```bash
# Benchmarks the quantized TorchScript model
python src/measure_latency.py --model_dir out_quantized --runs 100
```

## Repository Structure

  * `generate_data.py`: Custom script for synthetic data generation.
  * `src/quantize.py`: Script to apply dynamic int8 quantization.
  * `src/train.py`: Training loop using Hugging Face Trainer.
  * `src/predict.py`: Inference script for standard models.
  * `src/measure_latency.py`: Latency benchmarking tool (supports TorchScript).
  * `out/dev_pred.json`: Final prediction output for the development set.

### Checking quantized model predictions

If you're skeptical whether the quantized model will not give the same metrics as the parent model. You can check it yourself by running these in order:

(After you're done quantizing your model)

```python
# Generate predictions using the quantized model
python src/predict_quantized.py

# Evaluate the quantized predictions against the gold standard
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred_quantized.json
```