import argparse
import json
from pathlib import Path

import numpy as np


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text())
    return {k: np.array(v, dtype=float) for k, v in data.items()}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="embeddings_old.json")
    parser.add_argument("--new", default="embeddings.json")
    parser.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args()

    old = load_embeddings(args.old)
    new = load_embeddings(args.new)

    for name in args.names:
        if name not in old or name not in new:
            print(f"[SKIP] {name} missing in old/new embeddings")
            continue
        sim = cosine_similarity(old[name], new[name])
        print(f"{name}: cosine_similarity = {sim:.6f}")


if __name__ == "__main__":
    main()
