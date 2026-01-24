#!/usr/bin/env python3
"""
Migrate existing parquet artifacts to deterministic snippet IDs.

Why:
- Older runs may have random UUID snippet_id values, causing duplicates and inconsistent joins.
- The current preprocess uses deterministic ids; this script helps migrate existing data without a full rebuild.

What it does:
- Reads DATA/snippets.parquet and computes the deterministic snippet_id from (link, chunk_index).
- Produces *_migrated.parquet outputs for:
  - snippets.parquet
  - kpi_entities.parquet
  - snippets_embeddings.parquet
  - article_communities.parquet (if present)

It does NOT update Neo4j or ChromaDB. After migration, you should rebuild those stores.
"""

import argparse
import os
import pandas as pd

import config
from utils.id_utils import make_article_id, make_snippet_id


def migrate_snippets(df_snip: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if "snippet_id" not in df_snip.columns:
        raise ValueError("snippets parquet missing 'snippet_id'")
    if "chunk_index" not in df_snip.columns:
        raise ValueError("snippets parquet missing 'chunk_index'")

    mapping: dict[str, str] = {}
    new_ids = []
    new_article_ids = []

    for row in df_snip.itertuples(index=False):
        old_id = str(getattr(row, "snippet_id"))
        link = getattr(row, "link", "")
        source = getattr(row, "source", "")
        title = getattr(row, "title", "")
        published = getattr(row, "published", "")
        chunk_index = int(getattr(row, "chunk_index"))

        article_id = getattr(row, "article_id", "") or make_article_id(link, source, title, published)
        new_id = make_snippet_id(article_id, chunk_index)
        mapping[old_id] = new_id
        new_ids.append(new_id)
        new_article_ids.append(article_id)

    out = df_snip.copy()
    out["snippet_id_old"] = out["snippet_id"].astype(str)
    out["snippet_id"] = new_ids
    if "article_id" not in out.columns:
        out["article_id"] = new_article_ids

    out = out.drop_duplicates(subset=["snippet_id"], keep="first").reset_index(drop=True)
    return out, mapping


def rewrite_snippet_id(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if "snippet_id" not in df.columns:
        return df
    out = df.copy()
    out["snippet_id_old"] = out["snippet_id"].astype(str)
    out["snippet_id"] = out["snippet_id"].astype(str).map(mapping).fillna(out["snippet_id"].astype(str))
    return out


def main():
    parser = argparse.ArgumentParser(description="Migrate parquet artifacts to deterministic snippet IDs.")
    parser.add_argument("--out-dir", default=config.DATA_DIR, help="Directory to write migrated parquet files into.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(config.SNIPPETS_FILE):
        raise SystemExit(f"snippets parquet not found: {config.SNIPPETS_FILE}")

    df_snip = pd.read_parquet(config.SNIPPETS_FILE)
    migrated_snip, mapping = migrate_snippets(df_snip)

    out_snip = os.path.join(args.out_dir, "snippets_migrated.parquet")
    migrated_snip.to_parquet(out_snip, index=False)
    print(f"[OK] wrote {out_snip} (rows={len(migrated_snip)})")

    if os.path.exists(config.KPI_ENTITIES_FILE):
        df_kpi = pd.read_parquet(config.KPI_ENTITIES_FILE)
        migrated_kpi = rewrite_snippet_id(df_kpi, mapping)
        out_kpi = os.path.join(args.out_dir, "kpi_entities_migrated.parquet")
        migrated_kpi.to_parquet(out_kpi, index=False)
        print(f"[OK] wrote {out_kpi} (rows={len(migrated_kpi)})")

    emb_path = os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")
    if os.path.exists(emb_path):
        df_emb = pd.read_parquet(emb_path)
        migrated_emb = rewrite_snippet_id(df_emb, mapping)
        out_emb = os.path.join(args.out_dir, "snippets_embeddings_migrated.parquet")
        migrated_emb.to_parquet(out_emb, index=False)
        print(f"[OK] wrote {out_emb} (rows={len(migrated_emb)})")

    comm_path = os.path.join(config.DATA_DIR, "article_communities.parquet")
    if os.path.exists(comm_path):
        df_comm = pd.read_parquet(comm_path)
        migrated_comm = rewrite_snippet_id(df_comm, mapping)
        out_comm = os.path.join(args.out_dir, "article_communities_migrated.parquet")
        migrated_comm.to_parquet(out_comm, index=False)
        print(f"[OK] wrote {out_comm} (rows={len(migrated_comm)})")

    print("\nNext steps:")
    print("- Rebuild Neo4j graph (graph_build + community detection) using migrated snippet ids.")
    print("- Rebuild ChromaDB from migrated embeddings or re-run rag_index + load_chroma.")


if __name__ == "__main__":
    main()
