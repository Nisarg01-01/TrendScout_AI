import os
import sys
import textwrap

# Ensure project root is on sys.path so we can import retrieval_service
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrieval_service import TrendScoutBackend

EVAL_QUERIES = [
    # Funding-focused, broad
    "Which new AI startups raised funding recently? List company, round, and investors.",
    "What notable AI fundraising rounds were announced this week?",
    "Name a few AI agents or tooling startups that just closed funding.",
    # Sector / domain mapping
    "Give examples of recently funded AI startups in fintech and healthtech.",
    "Which enterprise AI or B2B AI startups have raised money lately?",
    "List any AI infrastructure or model provider startups that recently raised funding.",
    # Geography
    "What are some recently funded AI startups in Europe or the UK?",
    "Which AI startups in India or Southeast Asia have raised funding recently?",
    # KPI / deal details
    "For one recent AI funding round, summarize the amount, round type, lead investor, and startup focus.",
    "What are common ticket sizes (in USD) for the AI deals mentioned recently?",
    # Trend / pattern questions
    "What trends do you see in recent AI startup funding?",
    "Which AI subdomains (e.g., agents, copilots, infrastructure, robotics) are attracting the most recent funding?",
]


def run_eval():
    backend = TrendScoutBackend()
    print("Running evaluation on", len(EVAL_QUERIES), "queries...\n")
    for i, q in enumerate(EVAL_QUERIES, 1):
        print("=" * 80)
        print(f"Q{i}: {q}")
        print("-" * 80)
        try:
            answer = backend.generate_answer(q)
        except Exception as e:
            print("[ERROR]", e)
            continue
        print(textwrap.fill(answer, width=100))
        print()


if __name__ == "__main__":
    run_eval()
