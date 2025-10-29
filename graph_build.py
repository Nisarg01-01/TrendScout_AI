"""
graph_build.py
--------------
Builds and updates Article + KPI/Snippet graphs in Neo4j efficiently.

Features:
- Batched UNWIND writes for speed
- Progress bars for all major steps
- Append vs Rebuild mode
- PageRank + Louvain clusters
- Persists 'is_ai' flag on Entity nodes

Env vars:
  NEO4J_URI
  NEO4J_USER or NEO4J_USERNAME
  NEO4J_PASSWORD
  GRAPH_MODE=append or rebuild  (optional)
"""

import os
import itertools
from collections import defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import community as community_louvain
except Exception as e:
    raise SystemExit("Install 'python-louvain': pip install python-louvain") from e


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
ART_PATH = Path("data/articles_raw.parquet")
SNP_PATH = Path("data/snippets.parquet")
KE_PATH = Path("data/kpi_entities.parquet")

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
GRAPH_MODE = os.getenv("GRAPH_MODE", "append").lower()  # append | rebuild

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASS]):
    raise SystemExit("Missing Neo4j env vars: NEO4J_URI, NEO4J_USER/USERNAME, NEO4J_PASSWORD")

# -------------------------------------------------------------------
# NEO4J HELPERS
# -------------------------------------------------------------------
def run_query(driver, query, params=None):
    with driver.session() as session:
        session.run(query, params or {})

def batch_write(driver, query, rows, batch_size=100, desc="Batch"):
    """Write data to Neo4j in batches for speed."""
    for i in tqdm(range(0, len(rows), batch_size), desc=desc, ncols=80):
        chunk = rows[i:i + batch_size]
        with driver.session() as session:
            session.run(query, {"rows": chunk})

# -------------------------------------------------------------------
# GRAPH UTILITIES
# -------------------------------------------------------------------
def build_article_edges(kpi_df: pd.DataFrame):
    """Return dict {(link1, link2): weight} for shared org mentions."""
    orgs = kpi_df[kpi_df["entity_type"] == "organization"].copy()
    orgs["link"] = orgs["link"].astype(str)

    inv = defaultdict(set)
    for _, r in orgs.iterrows():
        if not r["link"]:
            continue
        inv[r["entity"].strip()].add(r["link"])

    pair_weights = defaultdict(int)
    for links in inv.values():
        links = list(links)
        if len(links) < 2:
            continue
        for l1, l2 in itertools.combinations(sorted(links), 2):
            pair_weights[(l1, l2)] += 1
            pair_weights[(l2, l1)] += 1
    return pair_weights

def infer_article_metrics(pair_weights):
    """Compute PageRank + Louvain clusters."""
    G = nx.DiGraph()
    for (l1, l2), w in pair_weights.items():
        if w > 0:
            G.add_edge(l1, l2, weight=w)
    if not G:
        return {}, {}
    pr = nx.pagerank(G, weight="weight")
    partition = community_louvain.best_partition(G.to_undirected(), weight="weight")
    return pr, partition

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    if not ART_PATH.exists() or not SNP_PATH.exists() or not KE_PATH.exists():
        raise SystemExit("Missing required parquet files in data/")

    articles = pd.read_parquet(ART_PATH)
    snippets = pd.read_parquet(SNP_PATH)
    kpi = pd.read_parquet(KE_PATH)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # --- rebuild / append logic ---
    if GRAPH_MODE == "rebuild":
        print("Rebuild mode active → wiping Neo4j database...")
        run_query(driver, "MATCH (n) DETACH DELETE n")
    else:
        print("Append mode active → merging data with existing graph.")

    # --- create constraints ---
    print("Ensuring constraints...")
    constraint_queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.link IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Snippet) REQUIRE s.snippet_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (k:KPI) REQUIRE k.type IS UNIQUE",
    ]
    for q in constraint_queries:
        run_query(driver, q)

    # --- Article nodes ---
    print("Uploading articles...")
    article_rows = [
        {
            "link": str(r.get("link", "")),
            "title": str(r.get("title", "")),
            "published": str(r.get("published", "")),
            "source": str(r.get("source", "")),
        }
        for _, r in articles.iterrows()
    ]
    batch_write(driver,
        "UNWIND $rows AS row "
        "MERGE (a:Article {link:row.link}) "
        "SET a.title=row.title, a.published=row.published, a.source=row.source",
        article_rows, batch_size=50, desc="Articles")

    # --- Entities + Mentions (AI tagging supported) ---
    print("Uploading entities (with AI tags)...")
    org_rows = kpi[kpi["entity_type"] == "organization"][["entity", "link", "is_ai"]].dropna()
    entity_rows = [
        {
            "name": str(r["entity"]).strip().title(),
            "link": str(r["link"]),
            "is_ai": bool(r.get("is_ai", False)),
        }
        for _, r in org_rows.iterrows()
    ]
    batch_write(driver,
        "UNWIND $rows AS row "
        "MERGE (e:Entity {name:row.name}) "
        "SET e.kind='organization', e.is_ai=row.is_ai "
        "WITH e, row "
        "MATCH (a:Article {link:row.link}) "
        "MERGE (a)-[:MENTIONS]->(e)",
        entity_rows, batch_size=100, desc="Entities")

    # --- Snippets + IN relationships ---
    print("Uploading snippets...")
    snippet_rows = [
        {"snippet_id": str(r["snippet_id"]), "text": str(r["snippet_text"]), "link": str(r["link"])}
        for _, r in snippets.iterrows()
    ]
    batch_write(driver,
        "UNWIND $rows AS row "
        "MERGE (s:Snippet {snippet_id:row.snippet_id}) "
        "SET s.text=row.text "
        "WITH s, row "
        "MATCH (a:Article {link:row.link}) "
        "MERGE (s)-[:IN]->(a)",
        snippet_rows, batch_size=100, desc="Snippets")

    # --- KPI nodes + ABOUT relationships ---
    print("Uploading KPI relationships...")
    kpi_rows = [
        {
            "snippet_id": str(r["snippet_id"]),
            "kpi_type": str(r.get("kpi_type") or "Unknown"),
            "stance": str(r.get("stance") or "0"),
            "confidence": float(r.get("confidence") or 0.5),
        }
        for _, r in kpi.iterrows()
    ]
    batch_write(driver,
        "UNWIND $rows AS row "
        "MERGE (k:KPI {type:row.kpi_type}) "
        "WITH row, k "
        "MATCH (s:Snippet {snippet_id:row.snippet_id}) "
        "MERGE (s)-[r:ABOUT]->(k) "
        "SET r.stance=row.stance, r.confidence=row.confidence",
        kpi_rows, batch_size=100, desc="KPI/ABOUT")

    # --- CO_LINK edges ---
    print("Building article co-links...")
    pair_weights = build_article_edges(kpi)
    co_rows = [{"l1": l1, "l2": l2, "w": int(w)} for (l1, l2), w in pair_weights.items() if l1 != l2 and w > 0]
    batch_write(driver,
        "UNWIND $rows AS row "
        "MATCH (a1:Article {link:row.l1}), (a2:Article {link:row.l2}) "
        "MERGE (a1)-[r:CO_LINK]->(a2) "
        "SET r.w = coalesce(r.w,0) + row.w",
        co_rows, batch_size=100, desc="CO_LINK")

    # --- Centrality + Cluster metrics ---
    print("Computing PageRank + clusters...")
    pr, partition = infer_article_metrics(pair_weights)
    metric_rows = [
        {"link": l, "centrality": float(pr.get(l, 0.0)), "cluster_id": int(partition.get(l, -1))}
        for l in set(pr.keys()) | set(partition.keys())
    ]
    batch_write(driver,
        "UNWIND $rows AS row "
        "MATCH (a:Article {link:row.link}) "
        "SET a.centrality=row.centrality, a.cluster_id=row.cluster_id",
        metric_rows, batch_size=50, desc="Article metrics")

    driver.close()
    print("\n✅ Graph construction complete. All nodes and relationships written to Neo4j.")


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
