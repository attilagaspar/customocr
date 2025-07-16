from datasets import load_dataset
from PIL import Image
from transformers import TrOCRProcessor
from pathlib import Path

# === Config ===
CONVERTED_JSON_PATH = "labels.json"  # <-- output for converted file
IMAGE_FOLDER = Path("images")
MODEL_NAME = "microsoft/trocr-base-stage1"
MAX_LABEL_LENGTH = 32

# === Load dataset ===
dataset = load_dataset("json", data_files=CONVERTED_JSON_PATH, split="train")
dataset = dataset.map(lambda e: {"image_path": str(IMAGE_FOLDER / e["file_name"])})

# === Load processor ===
#processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
processor = TrOCRProcessor.from_pretrained("char_tokenizer_processor")

# === Preprocess ===
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]

    # Use tokenizer without adding extra special tokens
    encoding = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=MAX_LABEL_LENGTH,
        truncation=True,
        add_special_tokens=False,  # <-- this is key
    )

    # Convert padding tokens to -100 for label loss masking
    labels = [t if t != processor.tokenizer.pad_token_id else -100 for t in encoding.input_ids]
    
    return {"pixel_values": pixel_values, "labels": labels}


dataset = dataset.map(preprocess)
dataset.save_to_disk("trocr_dataset2")
print("✅ Saved dataset with multiline-safe text.")