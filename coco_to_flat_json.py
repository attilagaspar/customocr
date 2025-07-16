import json
import re

# === Config ===
COCO_JSON_PATH = "coco_labels.json"      # <-- your COCO json file
CONVERTED_JSON_PATH = "labels.json"      # <-- output for converted file
EOS_TOKEN = "[EOS]"                      # must match your tokenizer's special token

def clean_label(text: str) -> str:
    # Remove all whitespace except newlines
    return re.sub(r"[ \t\r\f\v]+", "", text)  # keeps \n

def convert_coco_to_flat_json(coco_path, out_path):
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id -> file_name mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build output list
    out = []
    for ann in coco["annotations"]:
        file_name = id_to_file.get(ann["image_id"])
        text = clean_label(ann.get("text", ""))
        if file_name is not None:
            text += EOS_TOKEN  # append EOS
            out.append({"file_name": file_name, "text": text})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {coco_path} to {out_path} with {len(out)} entries, EOS appended.")

# Run
convert_coco_to_flat_json(COCO_JSON_PATH, CONVERTED_JSON_PATH)
