import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.database.connection import get_connection

FEATURES = [
    "home_avg_gf_5","home_avg_ga_5","home_pts_5","home_winrate_5",
    "away_avg_gf_5","away_avg_ga_5","away_pts_5","away_winrate_5",
    "diff_pts_5","diff_avg_gf_5","diff_avg_ga_5","diff_winrate_5"
]

def main():
    ds = pd.read_csv("dataset_homewin.csv")
    ds["match_date"] = pd.to_datetime(ds["match_date"])
    ds = ds.sort_values("match_date").reset_index(drop=True)

    X = ds[FEATURES]
    y = ds["home_win"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    accs, losses, aucs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)

        accs.append(accuracy_score(yte, pred))
        losses.append(log_loss(yte, p))
        try:
            aucs.append(roc_auc_score(yte, p))
        except Exception:
            pass

    metrics = {
        "cv_accuracy_mean": float(sum(accs)/len(accs)),
        "cv_logloss_mean": float(sum(losses)/len(losses)),
        "cv_auc_mean": float(sum(aucs)/len(aucs)) if aucs else None,
        "rows": int(len(ds)),
        "features": FEATURES
    }

    print("OK Metricas:", metrics)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analytics.TrainingRuns (model_name, seasons, metrics_json)
        OUTPUT INSERTED.run_id
        VALUES (?, ?, ?)
    """, "logreg_homewin_v1", "2022-2024", json.dumps(metrics))
    run_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    print("OK Guardado TrainingRun run_id =", run_id)

if __name__ == "__main__":
    main()
