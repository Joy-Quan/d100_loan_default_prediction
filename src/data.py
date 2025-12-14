import shutil
import kagglehub
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

def load_data(filename: str = "Loan_Default.csv") -> pd.DataFrame:
    """
    Load the raw dataset. 
    If the file does not exist locally, download it from Kaggle automatically.
    """
    file_path = DATA_RAW_DIR / filename

    # Check if file exists, if not, download it
    if not file_path.exists():
        print(f"Dataset not found at {file_path}. Downloading from Kaggle...")
        try:
            # Ensure directory exists
            DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
            
            # Download using kagglehub (returns cache path)
            cache_path = kagglehub.dataset_download("yasserh/loan-default-dataset")
            
            # Find CSV in cache
            csv_files = list(Path(cache_path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV file found in downloaded dataset.")
            
            # Move file to our raw data directory
            source_file = csv_files[0]
            print(f"Moving file to {file_path}...")
            shutil.copy(source_file, file_path)
            print("Download complete.")
            
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    # Load and return the data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    print("Running data module as a script...")
    df = load_data()
