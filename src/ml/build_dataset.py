import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.database.connection import get_connection

N_FORM = 10  # ultimos 10 partidos

def load_matches() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("""
        SELECT
            match_id, league_id, home_team_id, away_team_id,
            match_date, home_goals, away_goals, status
        FROM core.Matches
        WHERE status = 'finished'
          AND home_goals IS NOT NULL AND away_goals IS NOT NULL
    """, conn)
    conn.close()

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date")
    return df

def add_team_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["match_id","league_id","match_date","home_team_id","away_team_id","home_goals","away_goals"]].copy()
    home.rename(columns={
        "home_team_id":"team_id", "away_team_id":"opp_id",
        "home_goals":"gf", "away_goals":"ga"
    }, inplace=True)
    home["is_home"] = 1

    away = df[["match_id","league_id","match_date","home_team_id","away_team_id","home_goals","away_goals"]].copy()
    away.rename(columns={
        "away_team_id":"team_id", "home_team_id":"opp_id",
        "away_goals":"gf", "home_goals":"ga"
    }, inplace=True)
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)
    long.sort_values(["team_id","match_date"], inplace=True)

    long["win"]  = (long["gf"] > long["ga"]).astype(int)
    long["draw"] = (long["gf"] == long["ga"]).astype(int)
    long["loss"] = (long["gf"] < long["ga"]).astype(int)

    g = long.groupby("team_id", group_keys=False)

    long["form_points"] = (3*long["win"] + 1*long["draw"]).astype(float)

    long["avg_gf_5"] = g["gf"].apply(lambda s: s.shift(1).rolling(N_FORM).mean())
    long["avg_ga_5"] = g["ga"].apply(lambda s: s.shift(1).rolling(N_FORM).mean())
    long["pts_5"]    = g["form_points"].apply(lambda s: s.shift(1).rolling(N_FORM).sum())
    long["winrate_5"]= g["win"].apply(lambda s: s.shift(1).rolling(N_FORM).mean())

    home_feats = long[long["is_home"]==1][["match_id","avg_gf_5","avg_ga_5","pts_5","winrate_5"]].copy()
    home_feats.columns = ["match_id","home_avg_gf_5","home_avg_ga_5","home_pts_5","home_winrate_5"]

    away_feats = long[long["is_home"]==0][["match_id","avg_gf_5","avg_ga_5","pts_5","winrate_5"]].copy()
    away_feats.columns = ["match_id","away_avg_gf_5","away_avg_ga_5","away_pts_5","away_winrate_5"]

    out = df.merge(home_feats, on="match_id", how="left").merge(away_feats, on="match_id", how="left")

    out["home_win"] = (out["home_goals"] > out["away_goals"]).astype(int)

    out["diff_pts_5"]     = out["home_pts_5"] - out["away_pts_5"]
    out["diff_avg_gf_5"]  = out["home_avg_gf_5"] - out["away_avg_gf_5"]
    out["diff_avg_ga_5"]  = out["home_avg_ga_5"] - out["away_avg_ga_5"]
    out["diff_winrate_5"] = out["home_winrate_5"] - out["away_winrate_5"]

    out = out.dropna().reset_index(drop=True)
    return out

def main():
    df = load_matches()
    ds = add_team_rolling_features(df)

    ds.to_csv("dataset_homewin.csv", index=False)
    print("OK Dataset creado:", ds.shape, "| guardado en dataset_homewin.csv")

if __name__ == "__main__":
    main()
