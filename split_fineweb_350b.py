import os
import glob
import math

# Source directory containing the downloaded parquet files
SOURCE_DIR = "fineweb_data_350B"
# Base name for the 5 parts folders
TARGET_BASE = "fineweb_data_350B_part_"

def split_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    # Find all parquet files
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.parquet")))
    total_files = len(files)
    
    if total_files == 0:
        print(f"Error: No parquet files found in '{SOURCE_DIR}'.")
        return

    print(f"Found {total_files} files. Splitting into 5 parts...")

    num_parts = 20
    chunk_size = math.ceil(total_files / num_parts)

    for i in range(num_parts):
        part_id = i + 1
        target_dir = f"{TARGET_BASE}{part_id}"
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Determine the slice of files for this part
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_files)
        part_files = files[start_idx:end_idx]
        
        print(f"Part {part_id}: {target_dir} -> {len(part_files)} files ({start_idx} to {end_idx-1})")
        
        for src_file in part_files:
            filename = os.path.basename(src_file)
            dst_file = os.path.join(target_dir, filename)
            
            if not os.path.exists(dst_file):
                try:
                    # Create symlink using absolute paths to avoid issues
                    abs_src = os.path.abspath(src_file)
                    abs_dst = os.path.abspath(dst_file)
                    os.symlink(abs_src, abs_dst)
                except OSError as e:
                    print(f"Error linking {filename}: {e}")
    
    print("Done splitting dataset.")

if __name__ == "__main__":
    split_dataset()
