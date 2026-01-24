#!/usr/bin/env python3
"""
Upgrade legacy parquet artifacts to the new, deterministic ID scheme.

This is useful if you already have:
- DATA/articles_raw.parquet without article_id/canonical_url/text columns
- DATA/snippets.parquet with duplicate (link, chunk_index) rows

It produces new files (does not overwrite by default):
- articles_raw_v2.parquet (adds canonical_url, article_id, text, published_iso_utc)
- snippets_v2.parquet (re-built from articles, deterministic snippet_id, includes article_id)
"""

import argparse
import os

import pandas as pd

import config
from preprocess import clean_text, chunk_text
from utils.id_utils import canonicalize_url, make_article_id, make_snippet_id


def normalize_articles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "link" not in out.columns:
        out["link"] = ""

    if "canonical_url" not in out.columns:
        out["canonical_url"] = out["link"].astype(str).map(canonicalize_url)
        out.loc[out["canonical_url"].astype(str).str.strip() == "", "canonical_url"] = out["link"].astype(str)

    if "published" not in out.columns:
        out["published"] = ""

    published_dt = pd.to_datetime(out["published"], errors="coerce", utc=True)
    out["published_iso_utc"] = published_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.loc[published_dt.isna(), "published_iso_utc"] = out.loc[published_dt.isna(), "published"].astype(str)

    if "text" not in out.columns:
        if "summary" in out.columns:
            out["text"] = out["summary"].astype(str)
        else:
            out["text"] = ""

    if "article_id" not in out.columns:
        out["article_id"] = ""

    def _calc_article_id(row) -> str:
        existing = str(row.get("article_id", "") or "").strip()
        if existing:
            return existing
        return make_article_id(
            row.get("canonical_url") or row.get("link") or "",
            row.get("source") or "Unknown",
            row.get("title") or "",
            row.get("published_iso_utc") or row.get("published") or "",
        )

    out["article_id"] = out.apply(_calc_article_id, axis=1)
    return out


def rebuild_snippets(df_articles: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in df_articles.itertuples(index=False):
        article_id = getattr(row, "article_id", "")
        if not article_id:
            continue
        text = clean_text(getattr(row, "text", "") or "")
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "snippet_id": make_snippet_id(article_id, idx),
                    "article_id": article_id,
                    "source": getattr(row, "source", "Unknown"),
                    "title": getattr(row, "title", ""),
                    "link": getattr(row, "link", ""),
                    "canonical_url": getattr(row, "canonical_url", getattr(row, "link", "")),
                    "text": chunk,
                    "published": getattr(row, "published_iso_utc", getattr(row, "published", "")),
                    "chunk_index": idx,
                    "parent_fetched_at": getattr(row, "fetched_at", ""),
                }
            )
    df_snip = pd.DataFrame.from_records(records)
    if df_snip.empty:
        return df_snip
    df_snip = df_snip.drop_duplicates(subset=["snippet_id"], keep="first").reset_index(drop=True)
    return df_snip


def main():
    parser = argparse.ArgumentParser(description="Upgrade legacy TrendScout parquet artifacts.")
    parser.add_argument("--articles-in", default=config.ARTICLES_FILE)
    parser.add_argument("--articles-out", default=os.path.join(config.DATA_DIR, "articles_raw_v2.parquet"))
    parser.add_argument("--snippets-out", default=os.path.join(config.DATA_DIR, "snippets_v2.parquet"))
    args = parser.parse_args()

    if not os.path.exists(args.articles_in):
        raise SystemExit(f"Missing input: {args.articles_in}")

    df_articles = pd.read_parquet(args.articles_in)
    df_articles_v2 = normalize_articles(df_articles)
    df_articles_v2.to_parquet(args.articles_out, index=False)
    print(f"[OK] wrote {args.articles_out} (rows={len(df_articles_v2)})")

    df_snip_v2 = rebuild_snippets(df_articles_v2)
    df_snip_v2.to_parquet(args.snippets_out, index=False)
    print(f"[OK] wrote {args.snippets_out} (rows={len(df_snip_v2)})")

    print("\nNext:")
    print(f"- Point config to use `{os.path.basename(args.articles_out)}` and `{os.path.basename(args.snippets_out)}` OR rename them to overwrite the legacy files.")
    print("- Re-run extraction and rebuild Neo4j/Chroma after switching to v2 artifacts.")


if __name__ == "__main__":
    main()

