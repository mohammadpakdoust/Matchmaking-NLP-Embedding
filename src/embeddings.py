import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def build_person_embeddings(
    names: List[str],
    texts: List[str],
    model_name: str,
) -> Dict[str, np.ndarray]:
    """
    Generate sentence embeddings for each person.
    """
    if len(names) != len(texts):
        raise ValueError("Names and texts must have the same length.")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=False)

    return {names[i]: embeddings[i] for i in range(len(names))}


def save_embeddings(person_embeddings: Dict[str, np.ndarray], out_path: str) -> None:
    """
    Save embeddings to disk as JSON.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        name: vector.astype(float).tolist()
        for name, vector in person_embeddings.items()
    }

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a JSON file and convert lists back to numpy arrays.
    """
    with open(path, "r") as f:
        data = json.load(f)

    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
