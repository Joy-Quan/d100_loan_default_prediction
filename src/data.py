import shutil
import kagglehub
from pathlib import Path

if __name__ == "__main__":
    target_dir = Path("data/raw") 
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset...")
    cache_path = kagglehub.dataset_download("yasserh/loan-default-dataset")
    print(f"Done.")

    if csv_files := list(Path(cache_path).glob("*.csv")):
        source_file = csv_files[0]
        destination = target_dir / "Loan_Default.csv"
        print(f"Moving dataset to {destination}...")
        shutil.copy(source_file, destination)
        print("Done.")
    else:
        print("Error: No CSV file found in the downloaded folder.")
