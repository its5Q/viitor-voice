import os
import glob
import tarfile
import json
import math
from pathlib import Path
import shutil
from collections import defaultdict
import random
import soundfile as sf

def get_file_size(filepath):
    """Get size of file in bytes."""
    return os.path.getsize(filepath)

def create_webdataset_shards(input_dir, output_dir, shard_size_gb=1):
    """
    Create WebDataset shards from a directory of MP3 and JSON files.
    
    Args:
        input_dir: Directory containing MP3 and JSON files
        output_dir: Directory to write the output tar shards
        shard_size_gb: Target size of each shard in gigabytes
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate target shard size in bytes
    target_shard_size = shard_size_gb * 1024 * 1024 * 1024
    
    # Get all MP3 files
    mp3_files = sorted(glob.glob(os.path.join(input_dir, "*.mp3")))
    
    # Check for corresponding JSON files and create pairs
    file_pairs = []
    for mp3_file in mp3_files:
        base_name = os.path.splitext(mp3_file)[0]
        json_file = base_name + ".json"
        
        if os.path.exists(json_file):
            mp3_size = get_file_size(mp3_file)
            json_size = get_file_size(json_file)

            # Checking if the MP3 is good
            try:
                with sf.SoundFile(mp3_file) as f:
                    assert f.frames > 0

                file_pairs.append({
                    "mp3": mp3_file,
                    "json": json_file,
                    "total_size": mp3_size + json_size,
                    "base_name": os.path.basename(base_name)
                })
            except Exception:
                print(f'Error in MP3 file {mp3_file}, skipping')

        else:
            print(f"Warning: No matching JSON file for {mp3_file}")
    
    # Calculate how many shards we need
    total_size = sum(pair["total_size"] for pair in file_pairs)
    num_shards = math.ceil(total_size / target_shard_size)
    print(f"Creating {num_shards} shards for {len(file_pairs)} file pairs")
    
    # Distribute pairs across shards
    shards = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    
    # Assign file pairs to shards - try to keep each around target size
    for pair in file_pairs:
        # Find the shard with the lowest current size
        smallest_shard_idx = shard_sizes.index(min(shard_sizes))
        shards[smallest_shard_idx].append(pair)
        shard_sizes[smallest_shard_idx] += pair["total_size"]
    
    # Create tar files for each shard
    for i, shard in enumerate(shards):
        shard_name = f"shard_{i:06d}.tar"
        shard_path = os.path.join(output_dir, shard_name)
        
        with tarfile.open(shard_path, "w") as tar:
            for pair in sorted(shard, key=lambda x: x["base_name"]):
                # Get the base name without extension
                base_name = pair["base_name"]
                
                # Add MP3 file first
                mp3_path = pair["mp3"]
                mp3_arcname = f"{base_name}.mp3"
                tar.add(mp3_path, arcname=mp3_arcname)
                
                # Then add JSON file
                json_path = pair["json"]
                json_arcname = f"{base_name}.json"
                tar.add(json_path, arcname=json_arcname)
        
        print(f"Created shard {shard_path} with {len(shard)} pairs ({shard_sizes[i] / (1024*1024*1024):.2f} GB)")

# Example usage
if __name__ == "__main__":
    create_webdataset_shards(
        input_dir="/mnt/c/prepared_data", 
        output_dir="/mnt/c/sharded_data",
        shard_size_gb=1
    )