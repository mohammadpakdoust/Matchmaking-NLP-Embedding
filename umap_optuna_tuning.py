# umap_optuna_tuning.py
"""
Q4 - UMAP Hyperparameter Tuning (Optuna)

Objective:
Maximize the average Spearman rank correlation across students between:
1) Ranking by cosine similarity in embedding space (high-dim)
2) Ranking by Euclidean distance in 2D UMAP space

Notes:
- UMAP constraint: min_dist must be <= spread. We enforce this in sampling.
- We fix random_state (seed) so trials are comparable (UMAP is stochastic).
"""

import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

from src.io_utils import load_classmates
from src.embeddings import build_person_embeddings
from src.umap_eval import average_student_spearman, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="classmates.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results/umap_tuning")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data + build embeddings
    names, texts = load_classmates(args.csv)
    person_embeddings = build_person_embeddings(names, texts, model_name=args.model)

    # Matrix of embeddings in consistent order
    X = np.array([person_embeddings[n] for n in names], dtype=np.float32)

    # Scale embeddings before UMAP (common practice; helps stability)
    X = StandardScaler().fit_transform(X)

    def objective(trial: optuna.Trial) -> float:
        # --- UMAP hyperparameters to tune ---
        n_neighbors = trial.suggest_int("n_neighbors", 2, min(30, len(names) - 1))

        # Sample spread first, then enforce min_dist <= spread (UMAP constraint)
        spread = trial.suggest_float("spread", 0.5, 2.5)
        min_dist = trial.suggest_float("min_dist", 0.0, min(0.99, spread))

        metric = trial.suggest_categorical("metric", ["cosine", "euclidean"])

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            metric=metric,
            random_state=args.seed,  # fixed seed so trials are comparable
        )

        try:
            Y = reducer.fit_transform(X)
        except ValueError:
            # If any rare invalid combo slips through, penalize the trial
            return -1.0

        # Compute average Spearman correlation across all students
        score = average_student_spearman(X, Y)
        trial.set_user_attr("avg_spearman", score)
        return score

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    # Save best params
    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "seed": int(args.seed),
        "model": args.model,
        "csv": args.csv,
        "n_students": int(len(names)),
    }
    save_json(best, str(outdir / "best_umap_params.json"))

    # Save trial history (CSV)
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        row = {"trial": int(t.number), "value": float(t.value)}
        row.update(t.params)
        rows.append(row)

    pd.DataFrame(rows).sort_values("value", ascending=False).to_csv(
        outdir / "umap_trials.csv", index=False
    )

    print("Best avg Spearman:", study.best_value)
    print("Best params:", study.best_params)
    print("Saved:", outdir / "best_umap_params.json")
    print("Saved:", outdir / "umap_trials.csv")


if __name__ == "__main__":
    main()
