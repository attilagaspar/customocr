rm -r char_tokenizer_processor
rm -r trocr_dataset2
rm -r trocr_finetuned2

python3 coco_to_flat_json.py
python3 char_tokenizer_processor.py
python3 json_to_trocr_dataframe.py
python3 train_trocr.py
python3 apply_trocr_to_json.py valid/page_4.json valid_output

