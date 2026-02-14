import pandas as pd
from pathlib import Path


def load_classmates(csv_path: str) -> tuple[list[str], list[str]]:
    """
    Load classmates data from CSV.

    Parameters
    ----------
    csv_path : str
        Path to the classmates CSV file.

    Returns
    -------
    tuple[list[str], list[str]]
        List of names and list of corresponding text descriptions.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns (name, text).")

    names = df.iloc[:, 0].astype(str).tolist()
    texts = df.iloc[:, 1].astype(str).tolist()

    return names, texts
