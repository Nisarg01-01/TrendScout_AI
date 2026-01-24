#!/usr/bin/env python3
"""
Promote v2 parquet artifacts to become the active DATA files (with backups).

This is intended after running `upgrade_legacy_data.py`, which produces:
- DATA/articles_raw_v2.parquet
- DATA/snippets_v2.parquet

Why:
- Legacy artifacts may have duplicate snippets and lack article_id/canonical_url/text fields.
- Downstream artifacts (kpi_entities, embeddings, communities, rankings) become inconsistent with new snippet_ids.

What this script does:
1) Ensures v2 artifacts exist (runs upgrade_legacy_data.py if needed).
2) Moves current artifacts to timestamped backups.
3) Renames v2 artifacts to the canonical filenames used by the pipeline.
4) Backs up dependent artifacts that should be regenerated (kpi_entities, embeddings, communities, rankings, etc.).
"""

import os
import subprocess
import sys
import shutil
from datetime import datetime, timezone

import config


def backup_file(path: str, stamp: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    backup_path = f"{path}.bak_{stamp}"
    # In some sandboxed environments, renames/moves can be blocked even when writes are allowed.
    # Copying provides a robust backup without requiring rename permissions.
    shutil.copy2(path, backup_path)
    return backup_path


def copy_overwrite(src: str, dst: str) -> None:
    # Avoid os.replace/os.rename (can be blocked); write bytes to destination instead.
    with open(src, "rb") as r, open(dst, "wb") as w:
        shutil.copyfileobj(r, w, length=1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Promote v2 parquet artifacts to become active DATA files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak_* copies before overwriting.")
    parser.add_argument("--skip-dependent", action="store_true", help="Do not back up dependent artifacts.")
    parser.add_argument("--skip-upgrade", action="store_true", help="Do not run upgrade_legacy_data.py if v2 files are missing.")
    parser.add_argument("--v2-articles", default=None, help="Path to v2 articles parquet (default: DATA/articles_raw_v2.parquet).")
    parser.add_argument("--v2-snippets", default=None, help="Path to v2 snippets parquet (default: DATA/snippets_v2.parquet).")
    parser.add_argument("--articles-out", default=config.ARTICLES_FILE, help="Destination articles parquet (default: config.ARTICLES_FILE).")
    parser.add_argument("--snippets-out", default=config.SNIPPETS_FILE, help="Destination snippets parquet (default: config.SNIPPETS_FILE).")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_dir = config.DATA_DIR

    v2_articles = args.v2_articles or os.path.join(data_dir, "articles_raw_v2.parquet")
    v2_snippets = args.v2_snippets or os.path.join(data_dir, "snippets_v2.parquet")

    if not (os.path.exists(v2_articles) and os.path.exists(v2_snippets)):
        if args.skip_upgrade:
            print("[FAIL] v2 artifacts missing and --skip-upgrade was set.")
            return 1
        print("[INFO] v2 artifacts not found; running upgrade_legacy_data.py ...")
        script = os.path.join(config.CODE_DIR, "upgrade_legacy_data.py")
        subprocess.run([sys.executable, script], check=True)

    if not (os.path.exists(v2_articles) and os.path.exists(v2_snippets)):
        print("[FAIL] v2 artifacts still missing after upgrade.")
        print(f"  expected: {v2_articles}")
        print(f"  expected: {v2_snippets}")
        return 1

    # Backup primary artifacts
    primary = [
        args.articles_out,
        args.snippets_out,
    ]

    # Backup dependent artifacts that should be regenerated under the new IDs
    dependent = [
        config.KPI_ENTITIES_FILE,
        config.ENTITY_MAP_FILE,
        os.path.join(data_dir, "snippets_embeddings.parquet"),
        os.path.join(data_dir, "article_communities.parquet"),
        os.path.join(data_dir, "kpi_clusters.parquet"),
        os.path.join(data_dir, "community_entity_summary.parquet"),
        os.path.join(data_dir, "community_swot_summary.parquet"),
        os.path.join(data_dir, "community_temporal_summary.parquet"),
        os.path.join(data_dir, "community_forecast.parquet"),
        os.path.join(data_dir, "entity_rankings.parquet"),
        os.path.join(data_dir, "entity_ranking.parquet"),
        os.path.join(data_dir, "temporal_features.parquet"),
    ]

    if args.no_backup:
        print("[INFO] Skipping backups (--no-backup).")
    else:
        print(f"[INFO] Backing up existing artifacts with suffix .bak_{stamp}")
        to_backup = primary + ([] if args.skip_dependent else dependent)
        for path in to_backup:
            bak = backup_file(path, stamp)
            if bak:
                print(f"[OK] backup: {os.path.basename(path)} -> {os.path.basename(bak)}")

    # Promote v2 to active filenames
    copy_overwrite(v2_articles, args.articles_out)
    copy_overwrite(v2_snippets, args.snippets_out)
    print(f"[OK] promoted: {os.path.basename(v2_articles)} -> {os.path.basename(args.articles_out)}")
    print(f"[OK] promoted: {os.path.basename(v2_snippets)} -> {os.path.basename(args.snippets_out)}")
    print("[INFO] Note: v2 files are left in place (no rename/delete) for sandbox compatibility.")

    print("\nNext recommended steps:")
    print("- Re-run extraction: python CODE/extract_llm.py")
    print("- Rebuild graph: python CODE/graph_build.py")
    print("- Rebuild communities/clusters/ranking as needed via python CODE/run_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
