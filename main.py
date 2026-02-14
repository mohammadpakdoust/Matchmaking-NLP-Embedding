from pathlib import Path

from src.io_utils import load_classmates
from src.embeddings import build_person_embeddings, save_embeddings


def main() -> None:
    """
    Main pipeline:
    1. Load classmates
    2. Generate embeddings
    3. Save embeddings to disk
    """

    # ---- Configuration ----
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "classmates.csv"
    OUTPUT_PATH = BASE_DIR / "embeddings.json"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Load data ----
    names, texts = load_classmates(str(DATA_PATH))

    print(f"Loaded {len(names)} classmates.")

    # ---- Build embeddings ----
    person_embeddings = build_person_embeddings(
        names=names,
        texts=texts,
        model_name=MODEL_NAME,
    )

    print("Embeddings generated.")

    # ---- Save embeddings ----
    save_embeddings(person_embeddings, str(OUTPUT_PATH))

    print(f"Embeddings saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
