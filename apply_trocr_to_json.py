import sys
import os
import json
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


#python apply_trocr_to_json.py C:\Users\agaspar\Dropbox\research\leporolt_adatok\econai\census\agricultural_census1935_layout\batch_4\images\page_4.json effocr_traindata2

def ocr_image_pil(image, processor, model, device):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def main(json_path, output_folder, model_dir="trocr_finetuned2"):
    # Load model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find image
    img_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
    img = Image.open(img_path).convert("RGB")

    # Iterate over shapes
    for shape in data.get("shapes", []):
        if shape.get("label") == "numerical_cell" and "points" in shape and len(shape["points"]) == 2:
            (x1, y1), (x2, y2) = shape["points"]
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            snippet = img.crop((x1, y1, x2, y2))
            trocr_text = ocr_image_pil(snippet, processor, model, device)
            shape["trOCR output"] = trocr_text
            print(f"Processed shape: {shape['label']} at {shape['points']} -> {trocr_text}")

    # Save updated JSON
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, os.path.basename(json_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    image_filename = os.path.splitext(os.path.basename(json_path))[0] + ".jpg"
    output_image_path = os.path.join(output_folder, image_filename)

    img.save(output_image_path, format="JPEG")
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply_trocr_to_json.py input.json output_folder")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])