"""
TrendScout AI - Evaluation Pipeline

This script provides a framework for evaluating the quality and accuracy of the TrendScout AI system.
It includes tests for individual components (KPI extraction, clustering) and the end-to-end retrieval system.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import ollama
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import config
from retrieval_service import TrendScoutBackend
from extract_llm import extract_from_snippet
from utils.neo4j_utils import get_neo4j_driver

# --- Golden Datasets ---

GOLDEN_KPI_SNIPPETS = [
    {
        "text": "The AI startup 'InnovateAI' has just secured a $25M Series B funding round led by Sequoia Capital and Andreessen Horowitz.",
        "expected": {
            "entities": [{"name": "InnovateAI", "type": "Startup"}, {"name": "Sequoia Capital", "type": "VC"}, {"name": "Andreessen Horowitz", "type": "VC"}],
            "kpis": [{"type": "Funding", "amount": 25000000, "stage": "Series B", "investors": ["Sequoia Capital", "Andreessen Horowitz"]}]
        }
    },
    {
        "text": "Tech giant 'MegaCorp' announced a strategic partnership with 'InnovateAI' to integrate its generative AI platform into their cloud services.",
        "expected": {
            "entities": [{"name": "MegaCorp", "type": "Big Tech"}, {"name": "InnovateAI", "type": "Startup"}],
            "kpis": [{"type": "Partnership", "partner": "MegaCorp"}]
        }
    },
    {
        "text": "Following its recent funding, 'InnovateAI' is on a hiring spree, looking to add 50 new machine learning engineers to its team.",
        "expected": {
            "entities": [{"name": "InnovateAI", "type": "Startup"}],
            "kpis": [{"type": "Hiring", "count": 50, "roles": ["machine learning engineer"]}]
        }
    }
]

GOLDEN_QUERIES = [
    {
        "query": "What was the last funding round for Anthropic?",
        "ideal_facts": ["funding", "claude", "amount", "investors"]
    },
    {
        "query": "Compare OpenAI and Google on recent product launches.",
        "ideal_facts": ["OpenAI", "Google", "GPT-4o", "Gemini", "comparison", "features"]
    },
    {
        "query": "What are the biggest threats facing Mistral AI?",
        "ideal_facts": ["Mistral AI", "threat", "competition", "regulation", "market share"]
    },
    {
        "query": "Who are the top 3 trending startups in AI right now?",
        "ideal_facts": ["ranking", "score", "momentum", "top 3"]
    }
]


def color_print(text, color='green'):
    colors = {'green': '\033[92m', 'yellow': '\033[93m', 'red': '\033[91m', 'blue': '\033[94m', 'end': '\033[0m'}
    print(f"{colors.get(color, '')}{text}{colors['end']}")


def evaluate_kpi_extraction():
    """
    Evaluates the accuracy of the LLM-based KPI extraction against a golden dataset.
    """
    color_print("\n--- Evaluating Component: KPI Extraction ---", 'blue')
    scores = {"json_validity": [], "entity_precision": [], "entity_recall": [], "kpi_type_accuracy": [], "kpi_value_accuracy": []}

    for item in tqdm(GOLDEN_KPI_SNIPPETS, desc="Testing KPI Snippets"):
        text = item["text"]
        expected = item["expected"]
        actual = extract_from_snippet(text, config.LLM_MODEL)

        if actual is None:
            scores["json_validity"].append(0)
            continue
        scores["json_validity"].append(1)

        # Evaluate Entities
        expected_entities = {e['name'].lower() for e in expected.get('entities', [])}
        actual_entities = {e.get('name', '').lower() for e in actual.get('entities', [])}
        
        true_positives = len(expected_entities.intersection(actual_entities))
        precision = true_positives / len(actual_entities) if actual_entities else 0
        recall = true_positives / len(expected_entities) if expected_entities else 0
        scores["entity_precision"].append(precision)
        scores["entity_recall"].append(recall)

        # Evaluate KPIs
        expected_kpis = expected.get('kpis', [])
        actual_kpis = actual.get('kpis', [])
        if expected_kpis and actual_kpis:
            ek = expected_kpis[0]
            ak = actual_kpis[0]
            scores["kpi_type_accuracy"].append(1 if ek['type'] == ak.get('type') else 0)

            # Check numeric values with tolerance
            if ek['type'] == 'Funding':
                expected_val = ek.get('amount', 0)
                actual_val = ak.get('amount', 0)
                scores["kpi_value_accuracy"].append(1 if abs(expected_val - actual_val) < 0.01 * expected_val else 0)
            elif ek['type'] == 'Hiring':
                expected_val = ek.get('count', 0)
                actual_val = ak.get('count', 0)
                scores["kpi_value_accuracy"].append(1 if expected_val == actual_val else 0)

    # Print Report
    color_print("\nKPI Extraction Report Card:", 'blue')
    for metric, values in scores.items():
        if values:
            avg_score = np.mean(values)
            color = 'green' if avg_score > 0.7 else 'yellow' if avg_score > 0.5 else 'red'
            color_print(f"  - {metric:<20}: {avg_score:.2f}", color)


def evaluate_community_quality():
    """
    Calculates modularity and silhouette scores to evaluate cluster quality.
    """
    color_print("\n--- Evaluating Component: Community & Cluster Quality ---", 'blue')
    
    # 1. Article Community Modularity (G_a)
    try:
        import networkx as nx
        try:
            from community import community_louvain  # type: ignore
        except Exception:
            community_louvain = None

        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("MATCH (a1:Article)-[r:CO_LINK]-(a2:Article) RETURN a1.id AS source, a2.id AS target, r.weight AS weight")
            edges = [(r['source'], r['target'], r['weight']) for r in result]
        
        if edges:
            G = nx.Graph()
            G.add_weighted_edges_from(edges)
            
            df_comm = pd.read_parquet(os.path.join(config.DATA_DIR, "article_communities.parquet"))
            id_col = "article_id" if "article_id" in df_comm.columns else "snippet_id"
            partition = dict(zip(df_comm[id_col].astype(str), df_comm['community_id']))
            
            # Filter partition to nodes present in the graph
            filtered_partition = {node: comm for node, comm in partition.items() if node in G}

            if filtered_partition:
                if community_louvain is not None:
                    modularity = community_louvain.modularity(filtered_partition, G)
                else:
                    from networkx.algorithms.community.quality import modularity as nx_modularity  # type: ignore

                    # Convert mapping to list[set] expected by nx modularity
                    comm_to_nodes = {}
                    for node, cid in filtered_partition.items():
                        comm_to_nodes.setdefault(cid, set()).add(node)
                    modularity = nx_modularity(G, list(comm_to_nodes.values()), weight="weight")
                color_print(f"  - Article Community Modularity (G_a): {modularity:.4f}", 'green' if modularity > 0.4 else 'yellow')
            else:
                color_print("  - Could not calculate modularity: No nodes from partition in graph.", 'red')
        else:
            color_print("  - Skipping Modularity: No CO_LINK edges found in graph.", 'yellow')

    except Exception as e:
        color_print(f"  - Modularity calculation failed: {e}", 'red')

    # 2. KPI Cluster Silhouette Score (G_k) - for a sample cluster
    try:
        from sentence_transformers import SentenceTransformer
        df_kpi_clusters = pd.read_parquet(os.path.join(config.DATA_DIR, "kpi_clusters.parquet"))
        df_snippets = pd.read_parquet(config.SNIPPETS_FILE)
        
        # Pick a medium-sized article cluster to test
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster) WHERE c.size > 10 AND c.size < 100
                RETURN c.id as cluster_id LIMIT 1
            """)
            sample_cluster_id = result.single()
        
        if sample_cluster_id:
            sample_cluster_id = sample_cluster_id['cluster_id']
            # Get snippets and their KPI cluster labels for this article cluster
            snippet_ids_in_cluster = df_snippets[df_snippets['link'].str.contains(f"cluster_{sample_cluster_id}", na=False)]['snippet_id'] # Heuristic
            
            # This part is complex as we need to map snippets to their embeddings and kpi cluster labels
            # For simplicity, we'll assume we can get this mapping. A full implementation would require more joins.
            color_print("  - Silhouette Score (G_k): Placeholder - requires complex data joins.", 'yellow')
        else:
            color_print("  - Skipping Silhouette Score: No suitable sample cluster found.", 'yellow')

    except Exception as e:
        color_print(f"  - Silhouette Score calculation failed: {e}", 'red')


def evaluate_retrieval_end_to_end():
    """
    Tests the full RAG pipeline against a set of golden queries using an LLM-as-a-Judge.
    """
    color_print("\n--- Evaluating End-to-End: RAG Retrieval & Synthesis ---", 'blue')
    backend = TrendScoutBackend()
    
    judge_prompt_template = """
    You are an impartial judge. Evaluate the quality of a generated answer based on the provided context.
    Score on two metrics from 1 (bad) to 5 (excellent). Return ONLY a valid JSON object.

    Context: "{context}"
    Question: "{question}"
    Answer: "{answer}"

    Metrics:
    1. Faithfulness: Does the answer ONLY contain information supported by the context? (5 = fully supported, 1 = contains hallucinations)
    2. Relevancy: Does the answer directly address the user's question? (5 = perfectly relevant, 1 = irrelevant)

    JSON Response:
    {{
      "faithfulness": <score>,
      "relevancy": <score>,
      "reasoning": "Brief justification for your scores."
    }}
    """
    
    overall_scores = {"faithfulness": [], "relevancy": []}

    for item in tqdm(GOLDEN_QUERIES, desc="Testing Golden Queries"):
        query = item["query"]
        result = backend.generate_answer(query, return_context=True)
        
        context_for_judge = (
            f"Vector Context: {result['vector_context']}\n"
            f"Graph Context: {result['graph_context']}\n"
            f"Community Context: {result['community_context']}"
        )
        
        judge_prompt = judge_prompt_template.format(
            context=context_for_judge[:4000], # Truncate context to fit model window
            question=query,
            answer=result["answer"]
        )
        
        try:
            response = ollama.generate(model=config.LLM_MODEL, prompt=judge_prompt, format='json')
            scores = json.loads(response['response'])
            overall_scores["faithfulness"].append(scores.get("faithfulness", 0))
            overall_scores["relevancy"].append(scores.get("relevancy", 0))
            color_print(f"\nQuery: {query}\n  Faithfulness: {scores.get('faithfulness', 'N/A')}, Relevancy: {scores.get('relevancy', 'N/A')}\n  Reasoning: {scores.get('reasoning', '')}", 'green')
        except Exception as e:
            color_print(f"\nQuery: {query}\n  LLM-as-a-Judge failed: {e}", 'red')

    color_print("\nEnd-to-End Retrieval Report Card:", 'blue')
    for metric, values in overall_scores.items():
        if values:
            avg_score = np.mean(values)
            color = 'green' if avg_score >= 4.0 else 'yellow' if avg_score >= 3.0 else 'red'
            color_print(f"  - Average {metric:<12}: {avg_score:.2f} / 5.0", color)


def generate_swot_analysis(entity1: str, entity2: str):
    """
    Generates a comparative SWOT analysis for two entities.
    """
    color_print(f"\n--- Generating SWOT Analysis: {entity1} vs {entity2} ---", 'blue')
    backend = TrendScoutBackend()
    query = f"Compare {entity1} and {entity2}"
    
    # The prompt in retrieval_service is already tuned for comparison.
    # We just need to call it and display the result.
    answer = backend.generate_answer(query)
    
    print("\n" + answer)


def main():
    parser = argparse.ArgumentParser(description="Run the TrendScout AI evaluation pipeline.")
    parser.add_argument(
        '--suite',
        type=str,
        choices=['all', 'component', 'retrieval', 'swot'],
        default='all',
        help="Which evaluation suite to run."
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('ENTITY1', 'ENTITY2'),
        help="Run a SWOT comparison for two specific entities (e.g., --compare OpenAI Google)."
    )
    args = parser.parse_args()

    # Check if Ollama is running
    try:
        ollama.list()
    except Exception:
        color_print("Error: Ollama is not running. Please start the Ollama service to run evaluations.", 'red')
        return

    if args.suite in ['all', 'component']:
        evaluate_kpi_extraction()
        evaluate_community_quality()
    
    if args.suite in ['all', 'retrieval']:
        evaluate_retrieval_end_to_end()

    if args.compare:
        generate_swot_analysis(args.compare[0], args.compare[1])
    elif args.suite == 'swot':
        # Run a default comparison if --suite swot is specified without --compare
        generate_swot_analysis("OpenAI", "Google")


if __name__ == "__main__":
    main()
