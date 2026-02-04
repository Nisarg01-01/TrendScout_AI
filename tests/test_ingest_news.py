import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "CODE")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import ingest_news


class TestIngestNews(unittest.TestCase):
    def test_is_ai_relevant_keywords(self):
        self.assertTrue(ingest_news.is_ai_relevant("https://example.com/feed", "AI startup raises $10M", ""))
        # Must be AI-related AND have startup/market signal (or come from a startup feed).
        self.assertFalse(ingest_news.is_ai_relevant("https://example.com/feed", "AI in sports highlights", "AI is used in football"))
        self.assertFalse(ingest_news.is_ai_relevant("https://example.com/feed", "Cooking tips", "Pasta recipe"))

    @patch.object(ingest_news, "fetch_full_text", autospec=True)
    def test_parse_feed_emits_article_id_and_text(self, mock_fetch):
        mock_fetch.return_value = "Full article body"
        orig_min = getattr(ingest_news.config, "MIN_ARTICLE_TEXT_CHARS", 0)
        orig_fetch = getattr(ingest_news.config, "FETCH_FULL_TEXT", True)
        ingest_news.config.MIN_ARTICLE_TEXT_CHARS = 0
        ingest_news.config.FETCH_FULL_TEXT = True

        parsed = SimpleNamespace(
            feed={"title": "TestFeed"},
            entries=[
                {
                    "title": "AI startup raises $10M",
                    "summary": "<p>Generative AI company announced funding</p>",
                    "link": "https://example.com/a?utm_source=x",
                    "published": "Mon, 01 Jan 2025 00:00:00 GMT",
                },
                {
                    "title": "Unrelated cooking news",
                    "summary": "<p>Pasta recipe</p>",
                    "link": "https://example.com/b",
                    "published": "Mon, 01 Jan 2025 00:00:00 GMT",
                },
            ],
        )
        fake_feedparser = SimpleNamespace(parse=lambda _: parsed)

        try:
            with patch.object(ingest_news, "feedparser", new=fake_feedparser):
                rows = ingest_news.parse_feed("https://example.com/feed")
        finally:
            ingest_news.config.MIN_ARTICLE_TEXT_CHARS = orig_min
            ingest_news.config.FETCH_FULL_TEXT = orig_fetch
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("article_id", row)
        self.assertTrue(row["article_id"])
        self.assertEqual(row["text"], "Full article body")
        self.assertEqual(row["canonical_url"], "https://example.com/a")

    def test_should_skip_full_text_fetch_for_techcrunch_video(self):
        self.assertTrue(
            ingest_news._should_skip_full_text_fetch(
                "https://techcrunch.com/video/spacex-is-coming-to-the-public-markets-and-secondaries-are-already-on-fire/"
            )
        )

    @patch.object(ingest_news, "requests", autospec=True)
    def test_fetch_full_text_skips_request_when_url_is_skipped(self, mock_requests):
        # If the URL is in the skip list, we should not attempt any HTTP request.
        mock_requests.get.side_effect = AssertionError("requests.get should not be called for skipped URLs")
        txt = ingest_news.fetch_full_text("https://techcrunch.com/video/some-video/")
        self.assertEqual(txt, "")
