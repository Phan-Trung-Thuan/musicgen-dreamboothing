import os
import shutil
import json
from pathlib import Path
import argparse

def prepare_dataset(source_dir, caption_file, dest_dir):
    source_dir = Path(source_dir)
    caption_file = Path(caption_file)
    dest_dir = Path(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying files from {source_dir} to {dest_dir}...")
    extensions = ['*.wav', '*.mp3']
    count = 0
    for ext in extensions:
        for file_path in source_dir.rglob(ext):
            try:
                shutil.copy2(file_path, dest_dir / file_path.name)
                count += 1
            except Exception as e:
                print(f"Error copying {file_path.name}: {e}")
                pass
    print(f"Copied {count} files.")

    # Create metadata.jsonl for Hugging Face datasets
    metadata_path = dest_dir / "metadata.jsonl"
    with open(caption_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)

    with open(metadata_path, 'w', encoding='utf-8') as meta_f:
        for filename, data in captions.items():
            base_name = os.path.splitext(filename)[0]
            mp3_filename = f"{base_name}.mp3"
            
            # Check if file exists in dest_dir before adding to metadata
            if (dest_dir / mp3_filename).exists() or (dest_dir / filename).exists():
                actual_filename = mp3_filename if (dest_dir / mp3_filename).exists() else filename
                description = data.get("combined_prompt", data.get("description", ""))
                
                entry = {
                    "file_name": actual_filename,
                    "text": description
                }
                meta_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Created metadata.jsonl at {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare audio dataset for MusicGen training on Kaggle.")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing audio files.")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to JSON file containing captions.")
    parser.add_argument("--dest_dir", type=str, default="./musicfiles", help="Destination directory for processed dataset.")
    
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.caption_file, args.dest_dir)
