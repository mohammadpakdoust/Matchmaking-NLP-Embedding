import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

from src.io_utils import load_classmates
from src.embeddings import build_person_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="classmates.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--best", default="results/umap_tuning/best_umap_params.json")
    parser.add_argument("--out", default="results/umap_tuning/visualization_best.png")
    args = parser.parse_args()

    best = json.loads(Path(args.best).read_text())
    params = best["best_params"]
    seed = best["seed"]

    names, texts = load_classmates(args.csv)
    emb = build_person_embeddings(names, texts, model_name=args.model)

    X = np.array([emb[n] for n in names], dtype=np.float32)
    X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        spread=params["spread"],
        metric=params["metric"],
    )
    Y = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(Y[:, 0], Y[:, 1])
    for i, name in enumerate(names):
        plt.annotate(name, (Y[i, 0], Y[i, 1]), fontsize=6)
    plt.axis("off")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


if __name__ == "__main__":
    main()
