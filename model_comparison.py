import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text())
    return {k: np.array(v, dtype=float) for k, v in data.items()}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine(a, b)


def rank_against_anchor(emb: dict[str, np.ndarray], anchor: str) -> list[str]:
    if anchor not in emb:
        raise ValueError(f"Anchor '{anchor}' not found in embeddings.")
    a = emb[anchor]
    sims = []
    for name, vec in emb.items():
        if name == anchor:
            continue
        sims.append((name, cosine_similarity(a, vec)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in sims]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minilm", default="embeddings_minilm.json")
    parser.add_argument("--mpnet", default="embeddings_mpnet.json")
    parser.add_argument("--anchor", default="Mohammad Pakdoust")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    emb_a = load_embeddings(args.minilm)
    emb_b = load_embeddings(args.mpnet)

    rank_a = rank_against_anchor(emb_a, args.anchor)
    rank_b = rank_against_anchor(emb_b, args.anchor)

    common = [n for n in rank_a if n in set(rank_b)]
    pos_a = {n: i + 1 for i, n in enumerate(rank_a)}  # ranks start at 1
    pos_b = {n: i + 1 for i, n in enumerate(rank_b)}

    r_a = [pos_a[n] for n in common]
    r_b = [pos_b[n] for n in common]

    rho, p = spearmanr(r_a, r_b)

    print(f"Anchor: {args.anchor}")
    print(f"Compared classmates: {len(common)}")
    print(f"Spearman rho (rank correlation): {rho:.4f} (p={p:.4g})\n")

    print(f"Top {args.top} matches (MiniLM):")
    for n in rank_a[: args.top]:
        print(f"  {n}")

    print(f"\nTop {args.top} matches (MPNet):")
    for n in rank_b[: args.top]:
        print(f"  {n}")

    shifts = [(n, pos_a[n] - pos_b[n]) for n in common]  # + means moved up in MPNet
    shifts.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nBiggest rank shifts (+ = closer in MPNet):")
    for n, d in shifts[:10]:
        print(f"  {n:25s} shift={d:+d}")


if __name__ == "__main__":
    main()
