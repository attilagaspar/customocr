# TrOCR Fine-Tuning Pipeline for Structured OCR

This repository contains a complete pipeline to fine-tune Microsoft's TrOCR model on a custom OCR task, tailored for numerical and structured content extracted from annotated documents (e.g., historical census tables). It includes data preprocessing, character-level tokenizer training, model training, and inference on annotated JSON files.

## 📂 Project Structure

- `coco_to_flat_json.py` – Converts COCO-format annotated data to a simplified JSON format.
- `char_tokenizer_processor.py` – Trains a character-level tokenizer and saves a HuggingFace-compatible processor.
- `json_to_trocr_dataframe.py` – Converts the simplified JSON into a HuggingFace `datasets` dataset and applies preprocessing.
- `train_trocr.py` – Fine-tunes the TrOCR model on the dataset using the custom tokenizer.
- `apply_trocr_to_json.py` – Applies the fine-tuned model to JSON files and appends OCR results to each region.
- `ocr_training_sequence.sh` – Runs the full pipeline from scratch.

## 🚀 Quick Start

### 1. Prepare your COCO-style annotations
Ensure you have a file named `coco_labels.json` with image paths and labeled regions.

### 2. Run the full training sequence

```bash
bash ocr_training_sequence.sh
