import os
import torch
import datasets
from datasets import load_dataset, DatasetDict
from transformers import AutoProcessor, AutoFeatureExtractor, MusicgenForConditionalGeneration
import argparse
import librosa
import numpy as np
from multiprocess import set_start_method
from concurrent.futures import ThreadPoolExecutor
import gc

def main(args):
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading processor and model parts from {args.model_name_or_path}...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    # [FIX] Load in float16 to save 50% VRAM
    model = MusicgenForConditionalGeneration.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float16
    ).to(device)
    
    audio_encoder_feature_extractor = AutoFeatureExtractor.from_pretrained(model.config.audio_encoder._name_or_path)
    
    audio_decoder = model.audio_encoder
    model_decoder = model.decoder
    # We don't need the text encoder for this step, but it's part of the model. 
    # It stays on GPU but won't be used.

    # 1. Load Dataset from JSONL
    print(f"Loading dataset from {args.dataset_dir}...")
    data_files = {}
    for split in ["train", "test"]:
        jsonl_path = os.path.join(args.dataset_dir, f"{split}.jsonl")
        if os.path.exists(jsonl_path):
            data_files[split] = jsonl_path
    
    raw_datasets = load_dataset("json", data_files=data_files)
    
    max_target_length = int(args.max_duration_in_seconds * audio_encoder_feature_extractor.sampling_rate)
    num_codebooks = model.decoder.config.num_codebooks
    audio_encoder_pad_token_id = model.config.decoder.pad_token_id
    sampling_rate = audio_encoder_feature_extractor.sampling_rate

    def load_single_audio(path):
        try:
            # librosa.load is CPU heavy
            audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
            if len(audio) > max_target_length:
                audio = audio[:max_target_length]
            if len(audio) < max_target_length:
                audio = np.pad(audio, (0, max_target_length - len(audio)))
            return audio
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(max_target_length)

    # 2. Batched Preprocessing Function
    def preprocess_function(examples):
        audio_paths = examples["audio_path"]
        
        # Parallel Audio Loading (CPU Bound)
        with ThreadPoolExecutor(max_workers=4) as executor:
            audio_arrays = list(executor.map(load_single_audio, audio_paths))
        
        # Batch Encode Audio to Tokens (GPU accelerated)
        with torch.no_grad():
            # Feature extraction (normalize)
            inputs = audio_encoder_feature_extractor(
                audio_arrays, 
                sampling_rate=sampling_rate, 
                return_tensors="pt",
                padding=True
            ).input_values.to(device).to(torch.float16) # Ensure fp16
            # Encode
            labels = audio_decoder.encode(inputs)["audio_codes"]

            # [FIX] Encodec returns (num_frames, bsz, num_codebooks, seq_len)
            # Usually num_frames is 1 for our duration. We need (bsz, num_codebooks, seq_len)
            if labels.dim() == 4:
                labels = labels.squeeze(0)

            # Clean up inputs early
            del inputs

            # Apply delay pattern mask logic
            bsz = labels.shape[0]
            # pad_labels should be (bsz, num_codebooks, 1) to match (bsz, num_codebooks, seq_len)
            pad_labels = torch.ones((bsz, num_codebooks, 1), device=device, dtype=labels.dtype) * audio_encoder_pad_token_id
            labels = torch.cat([pad_labels, labels], dim=-1)

            all_labels = []
            for i in range(bsz):
                # build_delay_pattern_mask expects (num_codebooks, seq_len)
                l, delay_pattern_mask = model_decoder.build_delay_pattern_mask(
                    labels[i], 
                    audio_encoder_pad_token_id, 
                    labels.shape[-1] + num_codebooks
                )
                l = model_decoder.apply_delay_pattern_mask(l, delay_pattern_mask)
                # The first timestamp is associated to a row full of BOS, let's get rid of it
                all_labels.append(l[:, 1:].cpu().numpy())

            # Clean up GPU
            del labels
            torch.cuda.empty_cache()
            gc.collect()

        # Text
        texts = examples["text"]
        tokenized_text = processor.tokenizer(texts, padding=False)
        
        return {
            "labels": all_labels,
            "input_ids": tokenized_text["input_ids"]
        }

    # 3. Map with Batching and Parallel Loading
    print(f"Starting optimized preprocessing (Batch size: {args.batch_size})...")
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=raw_datasets["train"].column_names,
        keep_in_memory=False, # Force disk caching to save RAM
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
    parser.add_argument("--batch_size", type=int, default=8) # Lower default batch size
    
    args = parser.parse_args()
    main(args)
