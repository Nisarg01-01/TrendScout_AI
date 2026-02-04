import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT, "CODE")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import extract_llm


class TestExtractLlm(unittest.TestCase):
    def test_parse_first_json_object_ignores_trailing_text(self):
        raw = '{"a": 1, "entities": [], "kpis": [], "swot": [], "stance": 0.0}\n\nextra commentary'
        obj = extract_llm.parse_first_json_object(raw)
        self.assertIsInstance(obj, dict)
        self.assertEqual(obj.get("a"), 1)

    def test_parse_first_json_object_does_not_span_multiple_objects(self):
        raw = '{"entities": [], "kpis": [], "swot": [], "stance": 0.0}{"oops":true}'
        obj = extract_llm.parse_first_json_object(raw)
        self.assertIsInstance(obj, dict)
        self.assertIn("entities", obj)
        self.assertNotIn("oops", obj)

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

    def test_structured_kpis_capture_counterparts(self):
        payload = {
            "primary_entity": "OpenAI",
            "entities": [{"name": "OpenAI", "type": "Big Tech"}, {"name": "AI", "type": "Other"}],
            "sector": "Enterprise Software",
            "industry": "Generative AI",
            "kpis": [
                {
                    "type": "Acquisition",
                    "entity": "OpenAI",
                    "target": "XYZ Labs",
                    "description": "acqui-hire",
                    "value_text": "acquired XYZ Labs",
                    "polarity": 0,
                    "confidence": 0.8,
                },
                {
                    "type": "Competition",
                    "entity": "OpenAI",
                    "competitor": "Google",
                    "description": "market position",
                    "value_text": "behind Google",
                    "polarity": -1,
                    "confidence": 0.7,
                },
            ],
            "swot": [],
            "stance": 0.0,
        }

        original = extract_llm.extract_from_snippet
        try:
            extract_llm.extract_from_snippet = lambda text, model, num_predict=512: payload
            row = {"snippet_id": "s4", "text": "OpenAI acquired XYZ Labs. OpenAI is behind Google in enterprise AI."}
            out = extract_llm.process_single_row(row, model="x", num_predict=10)

            ents = [r for r in out if r.get("category") == "Entity"]
            self.assertEqual([e["entity_name"] for e in ents], ["OpenAI"])  # "AI" should be filtered as junk

            acq = [r for r in out if r.get("category") == "KPI" and r.get("detail_type") == "Acquisition"]
            self.assertEqual(len(acq), 1)
            self.assertEqual(acq[0].get("kpi_target"), "XYZ Labs")
            self.assertIn("acquired", acq[0].get("detail_value", ""))

            comp = [r for r in out if r.get("category") == "KPI" and r.get("detail_type") == "Competition"]
            self.assertEqual(len(comp), 1)
            self.assertEqual(comp[0].get("kpi_competitor"), "Google")
        finally:
            extract_llm.extract_from_snippet = original

    def test_process_single_row_supports_injected_extract_fn(self):
        calls = {"max_input_tokens": None, "trust_remote_code": None}

        def fake_hf(text, model, num_predict=512, prompt_style="full", max_input_tokens=None, trust_remote_code=False):
            calls["max_input_tokens"] = max_input_tokens
            calls["trust_remote_code"] = trust_remote_code
            return {
                "primary_entity": "OpenAI",
                "entities": [{"name": "OpenAI", "type": "Big Tech"}],
                "sector": "Enterprise Software",
                "industry": "AI",
                "kpis": [],
                "swot": [],
                "stance": 0.0,
            }

        original = extract_llm.extract_from_snippet_hf
        try:
            extract_llm.extract_from_snippet_hf = fake_hf
            row = {"snippet_id": "s5", "text": "OpenAI announced something."}
            out = extract_llm.process_single_row(
                row,
                model="hf-model",
                num_predict=16,
                prompt_style="compact",
                extract_fn=extract_llm.extract_from_snippet_hf,
                hf_max_input_tokens=123,
                hf_trust_remote_code=True,
            )

            self.assertEqual(calls["max_input_tokens"], 123)
            self.assertEqual(calls["trust_remote_code"], True)
            ents = [r for r in out if r.get("category") == "Entity"]
            self.assertEqual([e["entity_name"] for e in ents], ["OpenAI"])
        finally:
            extract_llm.extract_from_snippet_hf = original
