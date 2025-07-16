from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor
from datasets import load_from_disk
import torch

MODEL_NAME = "microsoft/trocr-base-stage1"
PROCESSOR_PATH = "char_tokenizer_processor"
DATASET_PATH = "trocr_dataset2"
OUTPUT_DIR = "trocr_finetuned2"

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())
# Load processor and model
#processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
#model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
# Adjust max length for shorter numeric sequences
processor.tokenizer.model_max_length = 32
model.config.max_length = 32
#model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
#model.config.pad_token_id = processor.tokenizer.pad_token_id
#model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.decoder_start_token_id = processor.tokenizer.pad_token_id

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

print(processor.tokenizer.tokenize("1 305\n1 306"))
print("Vocab size:", len(processor.tokenizer))
print("Tokenizer type:", type(processor.tokenizer))

# Load dataset
dataset = load_from_disk(DATASET_PATH)
# Split into train/validation (e.g. 90/10)
split = dataset.train_test_split(test_size=0.1)
train_ds = split["train"]
eval_ds = split["test"]

# Data collator
def collate_fn(batch):
    pixel_values = torch.stack([torch.tensor(x["pixel_values"]) for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50,
    save_steps=200,
    num_train_epochs=3,
    fp16=True,
    save_total_limit=2,
    report_to="none",
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("✅ Training complete. Model saved to", OUTPUT_DIR)