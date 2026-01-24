#!/usr/bin/env python3
"""
Lightweight data audit for TrendScout AI parquet artifacts.

This does not require Neo4j/Chroma/Ollama. It helps answer:
- Do we have reasonable dates?
- Are links present and unique?
- Are snippets duplicated?
- Does extraction output have expected columns?
"""

import os
import pandas as pd

import config


def _exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def _print(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def audit_articles():
    _print("ARTICLES")
    if not _exists(config.ARTICLES_FILE):
        print(f"[WARN] Missing: {config.ARTICLES_FILE}")
        return

    df = pd.read_parquet(config.ARTICLES_FILE)
    print(f"rows: {len(df)}")
    cols = list(df.columns)
    print(f"cols: {cols}")

    expected = ["canonical_url", "article_id", "text"]
    missing_expected = [c for c in expected if c not in df.columns]
    if missing_expected:
        print(f"[WARN] Missing expected columns (new pipeline): {missing_expected}")
        print("       This indicates the ingestion step has not been re-run with the updated schema.")

    if "link" in df.columns:
        link_na = df["link"].isna().sum()
        print(f"link missing: {link_na}")
        if df["link"].notna().any():
            nunique = df["link"].dropna().nunique()
            print(f"unique links: {nunique} ({nunique/len(df):.2%} of rows)")

    if "published" in df.columns:
        published_dt = pd.to_datetime(df["published"], errors="coerce", utc=True)
        bad = published_dt.isna().sum()
        print(f"published parse failures: {bad}")
        ok = published_dt.dropna()
        if not ok.empty:
            print(f"published range: {ok.min()} -> {ok.max()}")


def audit_snippets():
    _print("SNIPPETS")
    if not _exists(config.SNIPPETS_FILE):
        print(f"[WARN] Missing: {config.SNIPPETS_FILE}")
        return

    df = pd.read_parquet(config.SNIPPETS_FILE)
    print(f"rows: {len(df)}")
    cols = list(df.columns)
    print(f"cols: {cols}")

    if "article_id" not in df.columns:
        print("[WARN] Missing article_id in snippets. This indicates legacy snippets.parquet.")
        print("       Fix: rebuild snippets from articles or run migration/upgrade scripts.")

    if "snippet_id" in df.columns:
        nunique = df["snippet_id"].nunique()
        print(f"unique snippet_id: {nunique} ({nunique/len(df):.2%} of rows)")

    if "chunk_index" in df.columns:
        if "article_id" in df.columns:
            key = df["article_id"].fillna("").astype(str) + "|" + df["chunk_index"].fillna(-1).astype(int).astype(str)
            dupe = key.duplicated().sum()
            print(f"potential duplicates by (article_id, chunk_index): {dupe}")
        elif "link" in df.columns:
            key = df["link"].fillna("").astype(str) + "|" + df["chunk_index"].fillna(-1).astype(int).astype(str)
            dupe = key.duplicated().sum()
            print(f"potential duplicates by (link, chunk_index): {dupe}")
            if dupe:
                print("       This usually happens if older runs appended snippets with random snippet_id values.")
                print("       Fix: `python CODE/upgrade_legacy_data.py` (or `python CODE/migrate_data_ids.py`).")

    if "text" in df.columns:
        empty = (df["text"].astype(str).str.strip() == "").sum()
        print(f"empty text rows: {empty}")


def audit_extraction():
    _print("KPI/ENTITY EXTRACTION")
    if not _exists(config.KPI_ENTITIES_FILE):
        print(f"[WARN] Missing: {config.KPI_ENTITIES_FILE}")
        return

    df = pd.read_parquet(config.KPI_ENTITIES_FILE)
    print(f"rows: {len(df)}")
    cols = list(df.columns)
    print(f"cols: {cols}")

    # Staleness check: after upgrading/promoting snippets, older extraction rows may reference
    # snippet_ids that no longer exist. This can silently pollute downstream graph/ranking.
    if _exists(config.SNIPPETS_FILE) and "snippet_id" in df.columns:
        try:
            df_snip = pd.read_parquet(config.SNIPPETS_FILE, columns=["snippet_id"])
            current_ids = set(df_snip["snippet_id"].astype(str))
            extracted_ids = set(df["snippet_id"].astype(str))
            missing = extracted_ids - current_ids
            if missing:
                print(f"[WARN] extraction snippet_id not in current snippets: {len(missing)}")
                print("       This usually means snippets were rebuilt (new IDs) but extraction was not re-run.")
                print("       Fix: `python CODE/extract_llm.py` (optionally delete/overwrite old kpi_entities.parquet).")

            coverage_missing = current_ids - extracted_ids
            if coverage_missing:
                print(f"[WARN] snippets missing from extraction output: {len(coverage_missing)}")
                print("       This means some snippets have not been processed (or produced no rows).")
                print("       Fix: re-run `python CODE/extract_llm.py` until this is 0.")
        except Exception as e:
            print(f"[WARN] Could not validate extraction snippet_ids against snippets: {e}")

    if "category" in df.columns:
        print("category counts:")
        print(df["category"].value_counts(dropna=False).head(20).to_string())

    if "snippet_id" in df.columns:
        print(f"unique snippet_id: {df['snippet_id'].nunique()}")

    for col in ["kpi_amount", "kpi_count", "kpi_investors", "kpi_stage"]:
        if col in df.columns:
            print(f"{col} non-null: {df[col].notna().sum()}")


def main():
    audit_articles()
    audit_snippets()
    audit_extraction()


if __name__ == "__main__":
    main()
