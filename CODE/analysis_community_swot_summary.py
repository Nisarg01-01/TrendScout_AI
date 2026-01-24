import os
import sys
from typing import Dict, Any

import pandas as pd
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


def _is_junk_entity_name(name: str) -> bool:
    n = str(name or "").strip()
    if not n:
        return True
    low = n.lower()
    junk = {
        "ai",
        "artificial intelligence",
        "generative ai",
        "ai systems",
        "businesses",
        "companies",
        "company",
        "the company",
        "your company",
        "tech companies",
        "data centers",
        "unknown",
    }
    if low in junk:
        return True
    if len(n) <= 2:
        return True
    if low.endswith(" company") or low.endswith(" companies"):
        return True
    return False


def load_data() -> Dict[str, pd.DataFrame]:
    communities_path = os.path.join(config.DATA_DIR, "article_communities.parquet")
    if not os.path.exists(communities_path):
        raise FileNotFoundError(f"Communities file not found: {communities_path}. Run analysis_article_communities.py first.")

    if not os.path.exists(config.KPI_ENTITIES_FILE):
        raise FileNotFoundError(f"KPI/Entities file not found: {config.KPI_ENTITIES_FILE}")

    if not os.path.exists(config.SNIPPETS_FILE):
        raise FileNotFoundError(f"Snippets file not found: {config.SNIPPETS_FILE}")

    df_comm = pd.read_parquet(communities_path)
    df_kpi = pd.read_parquet(config.KPI_ENTITIES_FILE)
    df_snip = pd.read_parquet(config.SNIPPETS_FILE)

    return {"communities": df_comm, "kpi": df_kpi, "snip": df_snip}


def enrich_with_community(df_kpi: pd.DataFrame, df_comm: pd.DataFrame) -> pd.DataFrame:
    """Join KPI data with community IDs, filtering out unassigned articles."""
    
    if "article_id" not in df_comm.columns:
        raise ValueError("article_communities.parquet must contain 'article_id'. Re-run analysis_article_communities.py.")

    if "article_id" not in df_kpi.columns:
        # Map snippet -> article, then join communities by article_id.
        if os.path.exists(config.SNIPPETS_FILE):
            df_snip = pd.read_parquet(config.SNIPPETS_FILE, columns=["snippet_id", "article_id"])
            merged = df_kpi.merge(df_snip, how="left", on="snippet_id")
        else:
            merged = df_kpi.copy()
            merged["article_id"] = None
    else:
        merged = df_kpi.copy()

    merged = merged.merge(df_comm, how="left", on="article_id")
    # Drop articles not assigned to any community
    merged = merged[merged["community_id"].notna()].copy()
    return merged


def compute_community_swot(df_kpi_comm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SWOT counts and entity mentions for each community."""
    
    rows = []

    # Aggregate SWOT analysis counts
    df_swot = df_kpi_comm[df_kpi_comm["category"] == "SWOT"].copy()
    for (cid, swot_type), g in df_swot.groupby(["community_id", "detail_type"]):
        rows.append(
            {
                "community_id": cid,
                "kind": "SWOT",
                "swot_type": swot_type,
                "count": len(g),
            }
        )

    # Aggregate entity mentions and sentiment
    df_ent = df_kpi_comm[df_kpi_comm["category"] == "Entity"].copy()
    df_ent = df_ent[df_ent["entity_name"].notna() & (df_ent["entity_name"].str.strip() != "")]
    df_ent = df_ent[~df_ent["entity_name"].astype(str).map(_is_junk_entity_name)]
    ent_agg = (
        df_ent.groupby(["community_id", "entity_name"], as_index=False)
        .agg(mentions=("entity_name", "count"), avg_stance=("stance", "mean"))
    )

    return pd.DataFrame(rows), ent_agg


def compute_community_temporal(df_kpi_comm: pd.DataFrame, df_snip: pd.DataFrame) -> pd.DataFrame:
    """Group entity mentions by month to track temporal trends per community."""
    
    # Join with published dates from snippets
    df_snip_small = df_snip[["snippet_id", "published"]].copy()
    df = df_kpi_comm.merge(df_snip_small, how="left", on="snippet_id")

    # Convert to datetime and extract year-month
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df["year_month"] = df["published_dt"].dt.to_period("M").astype(str)

    # Count entity mentions per community per month
    df_ent = df[df["category"] == "Entity"].copy()
    df_ent = df_ent[df_ent["entity_name"].notna() & (df_ent["entity_name"].astype(str).str.strip() != "")]
    df_ent = df_ent[~df_ent["entity_name"].astype(str).map(_is_junk_entity_name)]
    temporal = (
        df_ent.groupby(["community_id", "year_month"], as_index=False)
        .agg(entity_mentions=("entity_name", "count"))
    )
    return temporal


def compute_forecast(temporal: pd.DataFrame) -> pd.DataFrame:
    """Generate forecast for next month's mentions using linear regression."""
    forecasts = []
    
    if temporal.empty:
        return pd.DataFrame()

    # Ensure we sort by date
    temporal = temporal.sort_values(["community_id", "year_month"])

    for cid, group in temporal.groupby("community_id"):
        if len(group) < 2:
            # Not enough data points to forecast
            forecasts.append({
                "community_id": cid,
                "slope": 0.0,
                "predicted_next_month": group["entity_mentions"].iloc[-1] if not group.empty else 0
            })
            continue

        # Create X (time steps) and Y (mentions)
        y = group["entity_mentions"].values
        x = np.arange(len(y))

        # Simple linear regression: y = mx + c
        # We use numpy's polyfit for a degree 1 polynomial (line)
        slope, intercept = np.polyfit(x, y, 1)

        # Predict next step
        next_x = len(y)
        prediction = slope * next_x + intercept

        forecasts.append({
            "community_id": cid,
            "slope": slope,
            "predicted_next_month": max(0, prediction) # No negative mentions
        })

    return pd.DataFrame(forecasts)


def compute_ranking(ent_agg: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    The Grand Ranking Algorithm.
    Score = alpha*centrality + beta*(pos-neg) + gamma*recency + epsilon*community_growth
    
    Here:
    - Centrality/Popularity = 'mentions'
    - Sentiment = 'avg_stance'
    - Community Growth = 'slope' from forecast
    """
    if ent_agg.empty:
        return pd.DataFrame()

    # Merge entity data with community forecast
    # If no forecast (e.g. new community), fill with 0
    merged = ent_agg.merge(forecast_df, on="community_id", how="left").fillna(0)

    # Weights (Alpha, Beta, Gamma/Epsilon)
    W_MENTIONS = 1.0
    W_STANCE = 5.0   # Stance is -1 to 1, so we boost its impact
    W_GROWTH = 10.0  # Growth slope is usually small, boost it

    # Calculate composite score
    # Keep raw stance values: negative stance reduces score
    merged["score"] = (
        (merged["mentions"] * W_MENTIONS) + 
        (merged["avg_stance"] * W_STANCE) + 
        (merged["slope"] * W_GROWTH)
    )

    # Sort by score descending
    ranked = merged.sort_values("score", ascending=False).reset_index(drop=True)
    
    # Add rank column
    ranked["rank"] = ranked.index + 1
    
    return ranked


def export_summaries(swot_rows: pd.DataFrame, ent_agg: pd.DataFrame, temporal: pd.DataFrame, forecast: pd.DataFrame, ranking: pd.DataFrame):
    out_swot = os.path.join(config.DATA_DIR, "community_swot_summary.parquet")
    out_ent = os.path.join(config.DATA_DIR, "community_entity_summary.parquet")
    out_temp = os.path.join(config.DATA_DIR, "community_temporal_summary.parquet")
    out_forecast = os.path.join(config.DATA_DIR, "community_forecast.parquet")
    out_ranking = os.path.join(config.DATA_DIR, "entity_ranking.parquet")

    if not swot_rows.empty:
        swot_rows.to_parquet(out_swot, index=False)
        print(f"Saved community SWOT summary to {out_swot}")
    else:
        print("No SWOT rows to save.")

    if not ent_agg.empty:
        ent_agg.to_parquet(out_ent, index=False)
        print(f"Saved community entity summary to {out_ent}")
    else:
        print("No entity aggregation to save.")

    if not temporal.empty:
        temporal.to_parquet(out_temp, index=False)
        print(f"Saved community temporal summary to {out_temp}")
    else:
        print("No temporal aggregation to save.")

    if not forecast.empty:
        forecast.to_parquet(out_forecast, index=False)
        print(f"Saved community forecast to {out_forecast}")
    
    if not ranking.empty:
        ranking.to_parquet(out_ranking, index=False)
        print(f"Saved entity ranking to {out_ranking}")


def print_quick_sample(ent_agg: pd.DataFrame, ranking: pd.DataFrame):
    if ent_agg.empty:
        return

    # Show top few communities with top entities
    print("\nSample community -> top entities:")
    for cid in ent_agg["community_id"].dropna().unique()[:5]:
        sub = ent_agg[ent_agg["community_id"] == cid].sort_values("mentions", ascending=False).head(5)
        print(f"\nCommunity {cid}:")
        print(sub[["entity_name", "mentions", "avg_stance"]].to_string(index=False))

    if not ranking.empty:
        print("\n--- TOP 10 TRENDING ENTITIES ---")
        print(ranking[["rank", "entity_name", "community_id", "score", "mentions", "slope"]].head(10).to_string(index=False))


def main():
    data = load_data()
    df_comm = data["communities"]
    df_kpi = data["kpi"]
    df_snip = data["snip"]

    print(f"Loaded {len(df_comm)} community mappings, {len(df_kpi)} KPI/entity rows, {len(df_snip)} snippets.")

    df_kpi_comm = enrich_with_community(df_kpi, df_comm)
    print(f"Rows with community assignment: {len(df_kpi_comm)}")

    swot_rows, ent_agg = compute_community_swot(df_kpi_comm)
    temporal = compute_community_temporal(df_kpi_comm, df_snip)
    
    # New steps: Forecast and Rank
    forecast = compute_forecast(temporal)
    ranking = compute_ranking(ent_agg, forecast)

    export_summaries(swot_rows, ent_agg, temporal, forecast, ranking)
    print_quick_sample(ent_agg, ranking)


if __name__ == "__main__":
    main()
