import os
import shutil
import json
from pathlib import Path
import argparse
import random

def prepare_dataset(source_dir, caption_file, dest_dir, test_split=0.1, seed=42):
    source_dir = Path(source_dir)
    caption_file = Path(caption_file)
    dest_dir = Path(dest_dir)
    random.seed(seed)

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Load captions
    with open(caption_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)

    # 1. Map captions to existing files in source_dir
    print(f"Scanning {source_dir} for audio files...")
    all_files = {} # filename -> Path
    for ext in ['*.wav', '*.mp3']:
        for file_path in source_dir.rglob(ext):
            all_files[file_path.name] = file_path
    
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
                "source_path": all_files[actual_filename],
                "file_name": actual_filename,
                "text": description
            })
    
    print(f"Found {len(valid_entries)} valid entries with captions.")
    
    # 2. Split into train and test
    random.shuffle(valid_entries)
    num_test = int(len(valid_entries) * test_split)
    test_entries = valid_entries[:num_test]
    train_entries = valid_entries[num_test:]
    
    # 3. Process splits
    for split_name, entries in [("train", train_entries), ("test", test_entries)]:
        if not entries:
            continue
            
        split_dir = dest_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / "metadata.jsonl"
        
        print(f"Processing {split_name} split ({len(entries)} files)...")
        with open(metadata_path, 'w', encoding='utf-8') as meta_f:
            for entry in entries:
                # Copy file
                try:
                    shutil.copy2(entry["source_path"], split_dir / entry["file_name"])
                    
                    # Write metadata
                    hf_entry = {
                        "file_name": entry["file_name"],
                        "text": entry["text"]
                    }
                    meta_f.write(json.dumps(hf_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error processing {entry['file_name']}: {e}")

    print(f"Dataset preparation complete at {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio dataset with train/test split for MusicGen training on Kaggle.")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing audio files.")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to JSON file containing captions.")
    parser.add_argument("--dest_dir", type=str, default="./musicfiles", help="Destination directory for processed dataset.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data to use for testing (e.g. 0.1 for 10%%).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.caption_file, args.dest_dir, args.test_split, args.seed)
