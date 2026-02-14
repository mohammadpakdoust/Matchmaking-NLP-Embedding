import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for rows of X."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def euclidean_distance_matrix(Y: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for rows of Y."""
    return cdist(Y, Y, metric="euclidean")


def rank_indices_from_scores(scores: np.ndarray, self_index: int) -> List[int]:
    """
    Convert a score vector into a ranking list of indices (excluding self).
    Higher score = better rank (closer).
    """
    idx = np.argsort(-scores)  # descending
    idx = [int(i) for i in idx if int(i) != int(self_index)]
    return idx


def rank_indices_from_distances(dists: np.ndarray, self_index: int) -> List[int]:
    """
    Convert a distance vector into a ranking list of indices (excluding self).
    Lower distance = better rank (closer).
    """
    idx = np.argsort(dists)  # ascending
    idx = [int(i) for i in idx if int(i) != int(self_index)]
    return idx


def spearman_rank_correlation(rank_a: List[int], rank_b: List[int]) -> float:
    """Spearman rho between two ranking lists over the same items."""
    if len(rank_a) != len(rank_b):
        raise ValueError("Ranking lists must have same length.")
    # map item -> rank position (1..N)
    pos_a = {item: i + 1 for i, item in enumerate(rank_a)}
    pos_b = {item: i + 1 for i, item in enumerate(rank_b)}
    items = list(pos_a.keys())
    ra = [pos_a[i] for i in items]
    rb = [pos_b[i] for i in items]
    rho, _ = spearmanr(ra, rb)
    return float(rho)


def average_student_spearman(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Average Spearman correlation across all students between:
    - cosine similarity ranking in embedding space (X)
    - euclidean distance ranking in UMAP space (Y)
    """
    cos_sim = cosine_similarity_matrix(X)          # (n,n) higher=closer
    umap_dist = euclidean_distance_matrix(Y)        # (n,n) lower=closer
    n = X.shape[0]
    rhos = []
    for i in range(n):
        rank_emb = rank_indices_from_scores(cos_sim[i], i)
        rank_umap = rank_indices_from_distances(umap_dist[i], i)
        rhos.append(spearman_rank_correlation(rank_emb, rank_umap))
    return float(np.nanmean(rhos))


def save_json(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
