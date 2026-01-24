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
import preprocess
from utils.id_utils import make_article_id, make_snippet_id


class TestPreprocess(unittest.TestCase):
    def test_process_articles_is_incremental_and_writes_article_id(self):
        # Use DATA/ for compatibility with restricted environments where only the repo is writable.
        # Note: some sandboxes block creating *subdirectories* under DATA/, so write files directly.
        base_root = os.path.join(ROOT, "DATA")
        self.assertTrue(os.path.isdir(base_root), "Expected DATA/ directory to exist")
        token = uuid.uuid4().hex
        test_articles = os.path.join(base_root, f"_test_articles_{os.getpid()}_{token}.parquet")
        test_snippets = os.path.join(base_root, f"_test_snippets_{os.getpid()}_{token}.parquet")

        orig = {
            "DATA_DIR": config.DATA_DIR,
            "ARTICLES_FILE": config.ARTICLES_FILE,
            "SNIPPETS_FILE": config.SNIPPETS_FILE,
            "CHUNK_SIZE": config.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
        }
        try:
            config.DATA_DIR = base_root
            config.ARTICLES_FILE = test_articles
            config.SNIPPETS_FILE = test_snippets
            config.CHUNK_SIZE = 10
            config.CHUNK_OVERLAP = 0

            url = "https://example.com/a"
            article_id = make_article_id(url, "Src", "Title", "2025-01-01")

            df_articles = pd.DataFrame(
                [
                    {
                        "source": "Src",
                        "title": "Title",
                        "link": url,
                        "canonical_url": url,
                        "article_id": article_id,
                        "text": "hello world this is a test",
                        "published": "2025-01-01T00:00:00Z",
                        "fetched_at": "2025-01-01T01:00:00Z",
                    }
                ]
            )
            df_articles.to_parquet(config.ARTICLES_FILE, index=False)

            preprocess.process_articles()
            df_snip = pd.read_parquet(config.SNIPPETS_FILE)
            self.assertTrue(len(df_snip) > 0)
            self.assertIn("article_id", df_snip.columns)
            self.assertTrue((df_snip["article_id"] == article_id).all())

            expected_first_id = make_snippet_id(article_id, 0)
            self.assertEqual(df_snip.iloc[0]["snippet_id"], expected_first_id)

            # Run again; should not add duplicates
            preprocess.process_articles()
            df_snip_2 = pd.read_parquet(config.SNIPPETS_FILE)
            self.assertEqual(len(df_snip_2), len(df_snip))
        finally:
            config.DATA_DIR = orig["DATA_DIR"]
            config.ARTICLES_FILE = orig["ARTICLES_FILE"]
            config.SNIPPETS_FILE = orig["SNIPPETS_FILE"]
            config.CHUNK_SIZE = orig["CHUNK_SIZE"]
            config.CHUNK_OVERLAP = orig["CHUNK_OVERLAP"]

            # Best-effort cleanup (may be restricted in some sandboxes).
            for p in [test_articles, test_snippets]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
