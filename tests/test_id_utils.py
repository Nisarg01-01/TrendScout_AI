import unittest
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "CODE")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.id_utils import canonicalize_url, make_article_id, make_snippet_id


class TestIdUtils(unittest.TestCase):
    def test_canonicalize_url_strips_fragment_and_tracking(self):
        url = "https://example.com/path?a=1&utm_source=x&fbclid=abc#section"
        out = canonicalize_url(url)
        self.assertEqual(out, "https://example.com/path?a=1")

    def test_make_article_id_is_deterministic(self):
        aid1 = make_article_id("https://example.com/a", "Src", "Title", "2025-01-01")
        aid2 = make_article_id("https://example.com/a", "Src", "Title", "2025-01-01")
        self.assertEqual(aid1, aid2)
        self.assertTrue(len(aid1) > 0)

    def test_make_snippet_id_is_deterministic(self):
        article_id = make_article_id("https://example.com/a", "Src", "Title", "2025-01-01")
        sid1 = make_snippet_id(article_id, 0)
        sid2 = make_snippet_id(article_id, 0)
        sid3 = make_snippet_id(article_id, 1)
        self.assertEqual(sid1, sid2)
        self.assertNotEqual(sid1, sid3)
