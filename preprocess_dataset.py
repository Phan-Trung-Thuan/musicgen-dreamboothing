import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoProcessor, AutoFeatureExtractor, MusicgenForConditionalGeneration
import argparse
import librosa
import numpy as np
from multiprocess import set_start_method

def main(args):
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"Loading processor and model parts from {args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = MusicgenForConditionalGeneration.from_pretrained(args.model_name_or_path)
    audio_encoder_feature_extractor = AutoFeatureExtractor.from_pretrained(model.config.audio_encoder._name_or_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_decoder = model.audio_encoder.to(device)
    model_decoder = model.decoder.to(device)

    # 1. Load Dataset from JSONL
    print(f"Loading dataset from {args.dataset_dir}...")
    data_files = {}
    for split in ["train", "test"]:
        jsonl_path = os.path.join(args.dataset_dir, f"{split}.jsonl")
        if os.path.exists(jsonl_path):
            data_files[split] = jsonl_path
    
    raw_datasets = load_dataset("json", data_files=data_files)
    
    max_target_length = args.max_duration_in_seconds * audio_encoder_feature_extractor.sampling_rate
    num_codebooks = model.decoder.config.num_codebooks
    audio_encoder_pad_token_id = model.config.decoder.pad_token_id

    # 2. Preprocessing Function
    def preprocess_function(batch):
        # Audio
        audio_path = batch["audio_path"]
        audio_array, _ = librosa.load(audio_path, sr=audio_encoder_feature_extractor.sampling_rate, mono=True)
        if len(audio_array) > max_target_length:
            audio_array = audio_array[:int(max_target_length)]
        
        # Encode Audio to Tokens (The RAM saver)
        with torch.no_grad():
            input_values = audio_encoder_feature_extractor(audio_array, sampling_rate=audio_encoder_feature_extractor.sampling_rate, return_tensors="pt").input_values.to(device)
            labels = audio_decoder.encode(input_values)["audio_codes"]
            
            # Apply delay pattern mask logic from dreambooth_musicgen.py
            pad_labels = torch.ones((1, 1, num_codebooks, 1), device=device) * audio_encoder_pad_token_id
            labels = torch.cat([pad_labels.to(labels.dtype), labels], dim=-1)
            labels, delay_pattern_mask = model_decoder.build_delay_pattern_mask(labels.squeeze(0), audio_encoder_pad_token_id, labels.shape[-1] + num_codebooks)
            labels = model_decoder.apply_delay_pattern_mask(labels, delay_pattern_mask)
            # labels is [num_codebooks, seq_len]
            batch["labels"] = labels[:, 1:].cpu().numpy()

        # Text
        text = batch["text"]
        batch["input_ids"] = processor.tokenizer(text)["input_ids"]
        
        return batch

    # 3. Map with RAM optimizations
    print("Starting preprocessing (Audio -> Encodec tokens)...")
    processed_datasets = raw_datasets.map(
        preprocess_function,
        remove_columns=raw_datasets["train"].column_names,
        writer_batch_size=100, # Flush to disk often
        desc="Preprocess and Encode"
    )

    # 4. Save to disk
    print(f"Saving processed dataset to {args.output_dir}...")
    processed_datasets.save_to_disk(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_duration_in_seconds", type=float, default=30.0)
    
    args = parser.parse_args()
    main(args)
