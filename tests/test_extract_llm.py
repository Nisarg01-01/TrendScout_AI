import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "CODE")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import extract_llm


class TestExtractLlm(unittest.TestCase):
    def test_process_single_row_emits_meta_on_no_json(self):
        original = extract_llm.extract_from_snippet
        try:
            extract_llm.extract_from_snippet = lambda text, model, num_predict=512: None
            row = {"snippet_id": "s1", "text": "hello"}
            out = extract_llm.process_single_row(row, model="x", num_predict=10)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["snippet_id"], "s1")
            self.assertEqual(out[0]["category"], "Meta")
            self.assertEqual(out[0]["detail_type"], "Error")
        finally:
            extract_llm.extract_from_snippet = original

    def test_process_single_row_emits_meta_on_empty_payload(self):
        payload = {"entities": [], "sector": "AI", "industry": "AI", "kpis": [], "swot": [], "stance": 0.0}
        original = extract_llm.extract_from_snippet
        try:
            extract_llm.extract_from_snippet = lambda text, model, num_predict=512: payload
            row = {"snippet_id": "s2", "text": "no signal"}
            out = extract_llm.process_single_row(row, model="x", num_predict=10)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["snippet_id"], "s2")
            self.assertEqual(out[0]["category"], "Meta")
            self.assertEqual(out[0]["detail_type"], "Empty")
        finally:
            extract_llm.extract_from_snippet = original

    def test_kpi_rows_attach_to_primary_entity(self):
        payload = {
            "primary_entity": "OpenAI",
            "entities": [{"name": "OpenAI", "type": "Big Tech"}, {"name": "Sequoia Capital", "type": "Investor"}],
            "sector": "Enterprise Software",
            "industry": "Generative AI",
            "kpis": [
                {
                    "type": "Funding",
                    "entity": "OpenAI",
                    "amount": "$10M",
                    "stage": "Series A",
                    "investors": ["Sequoia Capital"],
                    "value_text": "raised $10M",
                    "polarity": 1,
                    "confidence": 0.9,
                }
            ],
            "swot": [],
            "stance": 0.1,
        }

        original = extract_llm.extract_from_snippet
        try:
            extract_llm.extract_from_snippet = lambda text, model, num_predict=512: payload
            row = {"snippet_id": "s3", "text": "OpenAI raised $10M in Series A led by Sequoia Capital."}
            out = extract_llm.process_single_row(row, model="x", num_predict=10)

            funding = [r for r in out if r.get("category") == "KPI" and r.get("detail_type") == "Funding"]
            self.assertEqual(len(funding), 1)
            self.assertEqual(funding[0]["entity_name"], "OpenAI")
            self.assertTrue(float(funding[0].get("kpi_amount", 0.0)) > 0.0)
        finally:
            extract_llm.extract_from_snippet = original
