import os
import sys
from typing import Dict, Any

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


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
    """
    I'm joining the KPI data with the community IDs we found earlier.
    This way, every entity mention or SWOT point knows which 'neighborhood' of articles it belongs to.
    """
    merged = df_kpi.merge(df_comm, how="left", on="snippet_id")
    # If an article didn't make it into a community (maybe it was isolated), we'll drop it here.
    merged = merged[merged["community_id"].notna()].copy()
    return merged


def compute_community_swot(df_kpi_comm: pd.DataFrame) -> pd.DataFrame:
    """
    Let's crunch the numbers for each community.
    I'll count up the Strengths, Weaknesses, Opportunities, and Threats.
    I'll also see which entities are the most popular in each group and what the general vibe (stance) is.
    """
    rows = []

    # First, let's handle the SWOT analysis counts
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

    # Now, let's look at the entities: who are they and how are they perceived?
    df_ent = df_kpi_comm[df_kpi_comm["category"] == "Entity"].copy()
    df_ent = df_ent[df_ent["entity_name"].notna() & (df_ent["entity_name"].str.strip() != "")]
    ent_agg = (
        df_ent.groupby(["community_id", "entity_name"], as_index=False)
        .agg(mentions=("entity_name", "count"), avg_stance=("stance", "mean"))
    )

    return pd.DataFrame(rows), ent_agg


def compute_community_temporal(df_kpi_comm: pd.DataFrame, df_snip: pd.DataFrame) -> pd.DataFrame:
    """
    Time to look at the timeline.
    I'm grouping the data by month to see how active each community has been over time.
    """
    # I need the published dates from the snippets file
    df_snip_small = df_snip[["snippet_id", "published"]].copy()
    df = df_kpi_comm.merge(df_snip_small, how="left", on="snippet_id")

    # Let's make sure the dates are actual datetime objects so we can work with them
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df["year_month"] = df["published_dt"].dt.to_period("M").astype(str)

    # Counting entity mentions per community for each month
    df_ent = df[df["category"] == "Entity"].copy()
    temporal = (
        df_ent.groupby(["community_id", "year_month"], as_index=False)
        .agg(entity_mentions=("entity_name", "count"))
    )
    return temporal


def export_summaries(swot_rows: pd.DataFrame, ent_agg: pd.DataFrame, temporal: pd.DataFrame):
    out_swot = os.path.join(config.DATA_DIR, "community_swot_summary.parquet")
    out_ent = os.path.join(config.DATA_DIR, "community_entity_summary.parquet")
    out_temp = os.path.join(config.DATA_DIR, "community_temporal_summary.parquet")

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


def print_quick_sample(ent_agg: pd.DataFrame):
    if ent_agg.empty:
        return

    # Show top few communities with top entities
    print("\nSample community -> top entities:")
    for cid in ent_agg["community_id"].dropna().unique()[:5]:
        sub = ent_agg[ent_agg["community_id"] == cid].sort_values("mentions", ascending=False).head(5)
        print(f"\nCommunity {cid}:")
        print(sub[["entity_name", "mentions", "avg_stance"]].to_string(index=False))


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

    export_summaries(swot_rows, ent_agg, temporal)
    print_quick_sample(ent_agg)


if __name__ == "__main__":
    main()
