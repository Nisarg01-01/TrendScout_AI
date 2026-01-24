import os
import sys
import unittest
import uuid

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "CODE")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import config
import promote_v2_data
import upgrade_legacy_data


class TestUpgradeAndPromote(unittest.TestCase):
    def test_upgrade_legacy_data_normalizes_and_rebuilds_snippets(self):
        df_articles = pd.DataFrame(
            [
                {
                    "source": "TechCrunch",
                    "title": "Example",
                    "link": "https://example.com/a?utm_source=x#frag",
                    "summary": "Hello world. This is a test.",
                    "published": "2025-01-01T00:00:00Z",
                    "fetched_at": "2025-01-01T01:00:00Z",
                }
            ]
        )

        df_v2 = upgrade_legacy_data.normalize_articles(df_articles)
        self.assertIn("canonical_url", df_v2.columns)
        self.assertIn("article_id", df_v2.columns)
        self.assertIn("text", df_v2.columns)
        self.assertTrue(df_v2.loc[0, "canonical_url"].startswith("https://example.com/a"))
        self.assertTrue(str(df_v2.loc[0, "article_id"]).strip())

        # Force small chunks for deterministic, multi-snippet output.
        orig_chunk_size = config.CHUNK_SIZE
        orig_overlap = config.CHUNK_OVERLAP
        try:
            config.CHUNK_SIZE = 8
            config.CHUNK_OVERLAP = 0
            df_snip = upgrade_legacy_data.rebuild_snippets(df_v2)
        finally:
            config.CHUNK_SIZE = orig_chunk_size
            config.CHUNK_OVERLAP = orig_overlap

        self.assertTrue(len(df_snip) >= 1)
        self.assertIn("snippet_id", df_snip.columns)
        self.assertIn("article_id", df_snip.columns)
        self.assertEqual(df_snip["article_id"].nunique(), 1)
        self.assertEqual(df_snip["snippet_id"].nunique(), len(df_snip))

    def test_promote_v2_overwrites_dest_files(self):
        # Use DATA/ for compatibility with restricted environments where only the repo is writable.
        # Note: some sandboxes block creating *subdirectories* under DATA/, so write files directly.
        base_root = os.path.join(ROOT, "DATA")
        self.assertTrue(os.path.isdir(base_root), "Expected DATA/ directory to exist")
        token = uuid.uuid4().hex
        current_articles = os.path.join(base_root, f"_test_current_articles_{os.getpid()}_{token}.parquet")
        current_snippets = os.path.join(base_root, f"_test_current_snippets_{os.getpid()}_{token}.parquet")
        v2_articles = os.path.join(base_root, f"_test_v2_articles_{os.getpid()}_{token}.parquet")
        v2_snippets = os.path.join(base_root, f"_test_v2_snippets_{os.getpid()}_{token}.parquet")

        try:
            pd.DataFrame([{"a": 1}]).to_parquet(current_articles, index=False)
            pd.DataFrame([{"a": 1}]).to_parquet(current_snippets, index=False)
            pd.DataFrame([{"a": 2}]).to_parquet(v2_articles, index=False)
            pd.DataFrame([{"a": 2}]).to_parquet(v2_snippets, index=False)

            rc = promote_v2_data.main(
                [
                    "--no-backup",
                    "--skip-dependent",
                    "--skip-upgrade",
                    "--v2-articles",
                    v2_articles,
                    "--v2-snippets",
                    v2_snippets,
                    "--articles-out",
                    current_articles,
                    "--snippets-out",
                    current_snippets,
                ]
            )
            self.assertEqual(rc, 0)

            df_articles_after = pd.read_parquet(current_articles)
            df_snippets_after = pd.read_parquet(current_snippets)
            self.assertEqual(int(df_articles_after.loc[0, "a"]), 2)
            self.assertEqual(int(df_snippets_after.loc[0, "a"]), 2)
        finally:
            # Best-effort cleanup (may be restricted in some sandboxes).
            for p in [current_articles, current_snippets, v2_articles, v2_snippets]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
