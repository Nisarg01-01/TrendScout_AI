import os
import shutil
import argparse
import config


def _safe_under_repo(path: str) -> bool:
    base = os.path.abspath(config.BASE_DIR)
    p = os.path.abspath(path)
    try:
        common = os.path.commonpath([base, p])
    except Exception:
        return False
    return common == base


def _rm_file(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    if not _safe_under_repo(path):
        raise SystemExit(f"Refusing to delete outside repo: {path}")
    os.remove(path)
    return True


def _rm_dir(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    if not _safe_under_repo(path):
        raise SystemExit(f"Refusing to delete outside repo: {path}")
    shutil.rmtree(path, ignore_errors=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset TrendScout local data artifacts (parquet + ChromaDB).",
    )
    parser.add_argument("--yes", action="store_true", help="Required confirmation flag to delete files.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete all known DATA artifacts and ChromaDB (equivalent to --data --chroma).",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Delete DATA/*.parquet artifacts produced by the pipeline (keeps ChromaDB).",
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Delete the local ChromaDB directory (CODE/chroma_db).",
    )
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to delete without --yes. Example: python CODE/reset_data.py --all --yes")

    do_data = bool(args.all) or bool(args.data)
    do_chroma = bool(args.all) or bool(args.chroma)

    deleted = 0

    if do_data:
        targets = [
            config.ARTICLES_FILE,
            config.SNIPPETS_FILE,
            config.ENTITY_MAP_FILE,
            config.KPI_ENTITIES_FILE,
            os.path.join(config.DATA_DIR, "snippets_embeddings.parquet"),
            os.path.join(config.DATA_DIR, "article_communities.parquet"),
            os.path.join(config.DATA_DIR, "entity_rankings.parquet"),
            os.path.join(config.DATA_DIR, "temporal_features.parquet"),
        ]
        for p in targets:
            try:
                if _rm_file(p):
                    print(f"[OK] Deleted file: {p}")
                    deleted += 1
            except FileNotFoundError:
                pass

    if do_chroma:
        try:
            if os.path.exists(config.CHROMA_DB_PATH):
                _rm_dir(config.CHROMA_DB_PATH)
                print(f"[OK] Deleted dir: {config.CHROMA_DB_PATH}")
                deleted += 1
        except FileNotFoundError:
            pass

    if deleted == 0:
        print("[INFO] Nothing to delete.")
    else:
        print(f"[OK] Deleted {deleted} item(s).")


if __name__ == "__main__":
    main()

