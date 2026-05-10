import os
import json
from pathlib import Path
import argparse
import random

def prepare_dataset(source_dir, caption_file, dest_dir, test_split=0.2, seed=42):
    source_dir = Path(source_dir)
    caption_file = Path(caption_file)
    dest_dir = Path(dest_dir)
    random.seed(seed)

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Load captions
    with open(caption_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)

    # 1. Map captions to existing files in source_dir (Absolute Paths)
    print(f"Scanning {source_dir} for audio files...")
    all_files = {} # filename -> Path
    for ext in ['*.wav', '*.mp3']:
        for file_path in source_dir.rglob(ext):
            all_files[file_path.name] = file_path.absolute()
    
    valid_entries = []
    for filename, data in captions.items():
        base_name = os.path.splitext(filename)[0]
        mp3_filename = f"{base_name}.mp3"
        
        actual_filename = None
        if mp3_filename in all_files:
            actual_filename = mp3_filename
        elif filename in all_files:
            actual_filename = filename
            
        if actual_filename:
            description = data.get("combined_prompt", data.get("description", data.get("combined_captions", "")))
            valid_entries.append({
                "audio_path": str(all_files[actual_filename]),
                "text": description
            })
    
    print(f"Found {len(valid_entries)} valid entries with captions.")
    
    # 2. Split into train and test
    random.shuffle(valid_entries)
    num_test = int(len(valid_entries) * test_split)
    test_entries = valid_entries[:num_test]
    train_entries = valid_entries[num_test:]
    
    # 3. Save as JSONL files (No copying of audio)
    for split_name, entries in [("train", train_entries), ("test", test_entries)]:
        if not entries:
            continue
            
        output_file = dest_dir / f"{split_name}.jsonl"
        print(f"Saving {split_name} metadata to {output_file} ({len(entries)} entries)...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Zero-copy dataset preparation complete. Metadata saved in {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare metadata with absolute paths for zero-copy training on Kaggle.")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing audio files.")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to JSON file containing captions.")
    parser.add_argument("--dest_dir", type=str, default=".", help="Destination directory for JSONL metadata.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.caption_file, args.dest_dir, args.test_split, args.seed)
