
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/payment_dataset.csv")

def inspect_data(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Check the downloaded filename and location."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset shape ---")
    print(df.shape)

    print("\n--- First five records ---")
    print(df.head())

    print("\n--- Column data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isna().sum())

    print("\n--- Duplicate rows ---")
    print(df.duplicated().sum())

    if "isFraud" in df.columns:
        print("\n--- Target distribution ---")
        print(df["isFraud"].value_counts(dropna=False))
        print("\n--- Target percentages ---")
        print(
            df["isFraud"].value_counts(
                normalize=True, dropna=False
            ).mul(100).round(3)
        )
    
    return df
if __name__ == "__main__":
    inspect_data(DATA_PATH)
