import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

from src.io_utils import load_classmates
from src.embeddings import build_person_embeddings, load_embeddings  # adjust if your functions differ


def plot_points(coords, labels, out_path: Path):
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)

    for i, name in enumerate(labels):
        plt.annotate(name, (x[i], y[i]), fontsize=6)

    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="classmates.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 99, 123])
    parser.add_argument("--outdir", default="results/seed_sensitivity")
    args = parser.parse_args()

    names, texts = load_classmates(args.csv)
    person_embeddings = build_person_embeddings(names, texts, model_name=args.model)

    X = np.array(list(person_embeddings.values()))
    X_scaled = StandardScaler().fit_transform(X)

    for seed in args.seeds:
        reducer = umap.UMAP(random_state=seed)
        coords = reducer.fit_transform(X_scaled)

        out_path = Path(args.outdir) / f"visualization_seed_{seed}.png"
        plot_points(coords, names, out_path)

    print(f"Saved {len(args.seeds)} images to {args.outdir}")


if __name__ == "__main__":
    main()
