import os
from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Define paths
PARTS_COUNT = 20
BASE_PATH = "tokenized_path/fineweb_edu_350b_part"
OUTPUT_PATH = "tokenized_path/fineweb_edu_350b_merged"

def merge_datasets():
    # Dictionary to store lists of datasets for each split
    split_datasets = {}
    
    print(f"Loading {PARTS_COUNT} dataset parts...")
    
    parts_loaded = 0
    for i in range(1, PARTS_COUNT + 1):
        part_path = f"{BASE_PATH}{i}"
        if not os.path.exists(part_path):
            print(f"Warning: Path {part_path} does not exist. Skipping.")
            continue
            
        print(f"Loading {part_path}...")
        try:
            ds_dict = load_from_disk(part_path)
            
            # Check what splits are available (e.g., 'train', 'validation')
            for split_name in ds_dict.keys():
                if split_name not in split_datasets:
                    split_datasets[split_name] = []
                split_datasets[split_name].append(ds_dict[split_name])
            
            parts_loaded += 1
        except Exception as e:
            print(f"Error loading {part_path}: {e}")

    if parts_loaded == 0:
        print("No datasets loaded. Aborting merge.")
        return

    print("Concatenating datasets per split...")
    merged_splits = {}
    
    for split_name, ds_list in split_datasets.items():
        print(f"Merging split '{split_name}' with {len(ds_list)} parts...")
        merged_splits[split_name] = concatenate_datasets(ds_list)
        print(f"Split '{split_name}' size: {len(merged_splits[split_name])}")

    # Create a new DatasetDict with the merged splits
    final_dataset_dict = DatasetDict(merged_splits)
    
    print(f"Saving merged dataset dict to {OUTPUT_PATH} with num_proc=32...")
    try:
        final_dataset_dict.save_to_disk(OUTPUT_PATH, num_proc=20)
    except TypeError:
        print("Warning: This version of datasets might not support num_proc in DatasetDict.save_to_disk. Falling back to single process.")
        final_dataset_dict.save_to_disk(OUTPUT_PATH)
    
    print("Merge complete.")

if __name__ == "__main__":
    merge_datasets()
