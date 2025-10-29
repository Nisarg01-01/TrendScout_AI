"""
ner_kpi.py
-----------
Extracts named entities (startups, investors, people) and KPI events
(Funding, Hiring, Product, Partnership, Risk) from snippets.
Adds automatic 'is_ai' flag for organizations whose name or context indicates AI.
"""

import os
import re
import spacy
import pandas as pd
from pathlib import Path

SNIPPET_PATH = Path("data/snippets.parquet")
OUTPUT_PATH = Path("data/kpi_entities.parquet")

nlp = spacy.load("en_core_web_sm")

# KPI keyword rules
KPI_KEYWORDS = {
    "Funding": ["raised", "funding", "investment", "seed round", "series a", "series b", "backed by", "secured"],
    "Hiring": ["hiring", "recruiting", "joined", "appointed", "expanding team"],
    "Product": ["launched", "released", "unveiled", "demoed", "announced"],
    "Partnership": ["partnered", "collaborated", "integration", "alliance", "joined forces"],
    "Risk": ["lawsuit", "shutdown", "fired", "layoff", "breach", "bankrupt", "regulatory"]
}

POSITIVE_WORDS = ["raised", "launched", "hiring", "growth", "secured", "acquired", "expanded"]
NEGATIVE_WORDS = ["layoff", "lawsuit", "fired", "shutdown", "bankrupt", "risk", "decline"]

AI_KEYWORDS = [
    " ai ", "artificial intelligence", "machine learning", "deep learning",
    "generative ai", "large language model", "llm", "autonomous", "computer vision", "nlp"
]


def detect_kpi_type(text: str):
    tl = text.lower()
    for kpi, words in KPI_KEYWORDS.items():
        if any(w in tl for w in words):
            return kpi
    return None


def detect_stance(text: str):
    tl = text.lower()
    if any(w in tl for w in POSITIVE_WORDS):
        return "+"
    if any(w in tl for w in NEGATIVE_WORDS):
        return "-"
    return "0"


def classify_entity(ent):
    if ent.label_ == "ORG":
        return "organization"
    if ent.label_ == "PERSON":
        return "person"
    if ent.label_ == "MONEY":
        return "money"
    if ent.label_ == "GPE":
        return "location"
    return ent.label_.lower()


def is_ai_related(entity_text: str, context_text: str):
    """Flag AI-related entities using name patterns and context keywords."""
    name = entity_text.lower()
    context = context_text.lower()
    # Common AI hints in entity names
    ai_patterns = [
        " ai", "ai ", "artificial intelligence", "machine learning",
        "deep learning", "llm", "gpt", "mistral", "claude", "openai",
        "anthropic", "stability", "hugging face", "vertex ai", "copilot",
        "chatbot", "autonomous", "neural", "vision model"
    ]
    combined = f"{name} {context}"
    return any(pat in combined for pat in ai_patterns)

def main():
    if not SNIPPET_PATH.exists():
        print(f"File not found: {SNIPPET_PATH}")
        return

    os.makedirs("data", exist_ok=True)
    df = pd.read_parquet(SNIPPET_PATH)
    print(f"Loaded {len(df)} snippets for entity/KPI extraction.")

    records = []
    for _, row in df.iterrows():
        text = row["snippet_text"]
        doc = nlp(text)
        kpi = detect_kpi_type(text)
        stance = detect_stance(text)

        for ent in doc.ents:
            if ent.label_ not in ["ORG", "PRODUCT", "PERSON", "MONEY", "GPE"]:
                continue
            entity_text = ent.text.strip()
            if len(entity_text) < 3 or re.fullmatch(r"\d+", entity_text):
                continue
            entity_type = classify_entity(ent)
            ai_flag = is_ai_related(entity_text, text)

            records.append({
                "snippet_id": row["snippet_id"],
                "entity": entity_text.title(),
                "entity_type": entity_type,
                "is_ai": ai_flag,
                "kpi_type": kpi,
                "stance": stance,
                "confidence": 1.0 if kpi else 0.5,
                "excerpt": text[:300],
                "source": row.get("source", ""),
                "title": row.get("title", ""),
                "link": row.get("link", "")
            })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Extracted {len(out_df)} entity-KPI pairs → {OUTPUT_PATH}")

    print("\nSample extracted entities (AI flagged):")
    print(out_df.head(10)[["entity", "entity_type", "is_ai", "kpi_type", "stance"]])


if __name__ == "__main__":
    main()
