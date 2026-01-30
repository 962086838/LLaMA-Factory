import os
import argparse
import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing

def check_file(args):
    file_path, model_path = args
    filename = os.path.basename(file_path)
    try:
        # Load the dataset (parquet)
        ds = load_dataset("parquet", data_files=[file_path], split="train", keep_in_memory=False)
        
        # Load tokenizer (lazy load in worker)
        # Using fast tokenizer might be safer for multithreading, but we are in a process.
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # We perform a simple map to force reading and processing of the text column.
        # This mirrors the LLaMA-Factory preprocessing step which triggered the crash.
        def process_batch(examples):
            # Just tokenize to trigger potential crashes
            return tokenizer(examples["text"], truncation=True, max_length=2048)

        ds.map(process_batch, batched=True, batch_size=1000, num_proc=1, desc=f"Checking {filename}")
        
        return True, file_path, None
    except Exception as e:
        return False, file_path, str(e)

def main():
    parser = argparse.ArgumentParser(description="Scan parquet files for corruption.")
    parser.add_argument("--data_dir", type=str, default="fineweb_data_350B", help="Directory containing parquet files.")
    parser.add_argument("--model_path", type=str, default="Telechat3-1p8B", help="Path to the model/tokenizer.")
    parser.add_argument("--pattern", type=str, default="*.parquet", help="File pattern to match.")
    # Assuming user wants to run this sequentially or with controlled parallelism.
    # User mentioned "worker count 1".
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent file checks.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of files to process.")
    parser.add_argument("--end_index", type=int, default=None, help="End index of files to process (exclusive).")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, args.pattern))
    files.sort()
    
    if not files:
        print(f"No files found in {args.data_dir} with pattern {args.pattern}")
        return

    # Filter files based on environment variables or arguments
    start_idx = int(os.environ.get("DEBUG_START_INDEX", args.start_index))
    end_env = os.environ.get("DEBUG_END_INDEX")
    end_idx = int(end_env) if end_env is not None else args.end_index

    total_files_found = len(files)
    if end_idx is None:
        files = files[start_idx:]
    else:
        files = files[start_idx:end_idx]

    if not files:
        print(f"No files selected in range [{start_idx}:{end_idx}]. Total files found: {total_files_found}")
        return

    print(f"Found {total_files_found} files. Processing range [{start_idx}:{len(files)+start_idx}] ({len(files)} files).")
    print(f"Starting check with {args.workers} workers...")
    
    # Prepare arguments for workers
    worker_args = [(f, args.model_path) for f in files]
    
    failed_files = []

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            # We use imap to show progress
            for success, fpath, error in tqdm(pool.imap(check_file, worker_args), total=len(files)):
                if not success:
                    print(f"\n[FAIL] {fpath}: {error}")
                    failed_files.append(fpath)
    else:
        # Sequential execution
        for arg in tqdm(worker_args):
            success, fpath, error = check_file(arg)
            if not success:
                print(f"\n[FAIL] {fpath}: {error}")
                failed_files.append(fpath)

    print("\nScan complete.")
    if failed_files:
        print(f"Found {len(failed_files)} corrupted files:")
        for f in failed_files:
            print(f)
        # Write to log file
        with open("corrupted_files.txt", "w") as f:
            for line in failed_files:
                f.write(line + "\n")
        print("List of corrupted files saved to 'corrupted_files.txt'.")
    else:
        print("All files passed validity check.")

if __name__ == "__main__":
    main()
