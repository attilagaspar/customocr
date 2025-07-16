import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Split

from transformers import PreTrainedTokenizerFast, TrOCRProcessor, AutoFeatureExtractor
import os

# ==== CONFIG ====
LABELS_JSON = "labels.json"
CORPUS_TXT = "char_tokenizer_corpus.txt"
TOKENIZER_JSON = "char_tokenizer.json"
OUTPUT_DIR = "char_tokenizer_processor"
FEATURE_EXTRACTOR_BASE = "microsoft/trocr-base-stage1"
VOCAB_SIZE = 100  # enough for digits, dots, dashes, etc.

EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

def extract_text_labels():
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(CORPUS_TXT) or ".", exist_ok=True)

    with open(CORPUS_TXT, "w", encoding="utf-8") as f:
        for entry in data:
            text = entry.get("text", "").replace("\n", "\\n")  # encode newlines literally
            if text.strip():
                f.write(text + f"{EOS_TOKEN}\n")  # append EOS token string at the end
    print(f"✅ Extracted {len(data)} labels to {CORPUS_TXT}")

def train_tokenizer():
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Split("", behavior="isolated")  # isolate every character

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[PAD_TOKEN, UNK_TOKEN, "[CLS]", "[SEP]", "[MASK]", EOS_TOKEN],
    )

    tokenizer.train([CORPUS_TXT], trainer)
    tokenizer.add_special_tokens([EOS_TOKEN])
    tokenizer.save(TOKENIZER_JSON)
    print(f"✅ Tokenizer trained and saved to {TOKENIZER_JSON}")

def build_processor():
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_JSON,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    feature_extractor = TrOCRProcessor.from_pretrained(FEATURE_EXTRACTOR_BASE).image_processor
    processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=hf_tokenizer)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"✅ Processor saved to {OUTPUT_DIR}")
    print("📏 Vocab size:", len(hf_tokenizer))
    print("🔖 EOS token ID:", hf_tokenizer.eos_token_id)

def main():
    extract_text_labels()
    train_tokenizer()
    build_processor()

if __name__ == "__main__":
    main()
