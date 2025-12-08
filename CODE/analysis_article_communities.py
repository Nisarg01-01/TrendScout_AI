import os
import sys
from collections import defaultdict

import pandas as pd
import networkx as nx
from community import community_louvain
from neo4j import GraphDatabase


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


def load_kpi_entities() -> pd.DataFrame:
    """Load and filter entity extraction results from parquet file."""
    
    if not os.path.exists(config.KPI_ENTITIES_FILE):
        raise FileNotFoundError(f"KPI/Entities file not found: {config.KPI_ENTITIES_FILE}")

    df = pd.read_parquet(config.KPI_ENTITIES_FILE)
    df = df[df["category"] == "Entity"].copy()
    df = df[df["entity_name"].notna() & (df["entity_name"].str.strip() != "")]
    return df


def build_article_graph(df_entities: pd.DataFrame) -> nx.Graph:
    """Build article co-occurrence graph where edges represent shared entity mentions."""
    
    # Map each article to its mentioned entities
    snippet_to_ents: dict[str, set[str]] = defaultdict(set)
    for row in df_entities.itertuples(index=False):
        snippet_to_ents[row.snippet_id].add(str(row.entity_name).strip())

    # Now, initialize the graph with all our articles as nodes
    G = nx.Graph()
    for sid in snippet_to_ents.keys():
        G.add_node(sid)

    snippet_ids = list(snippet_to_ents.keys())
    n = len(snippet_ids)
    print(f"Building article graph over {n} snippets...")

    # Compare articles pairwise to find shared entities
    # O(n^2) complexity acceptable for moderate dataset sizes
    for i in range(n):
        s_i = snippet_ids[i]
        ents_i = snippet_to_ents[s_i]
        if not ents_i:
            continue
        for j in range(i + 1, n):
            s_j = snippet_ids[j]
            ents_j = snippet_to_ents[s_j]
            if not ents_j:
                continue
            shared = ents_i.intersection(ents_j)
            if shared:
                G.add_edge(s_i, s_j, weight=len(shared))

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def build_article_graph_from_neo4j(driver) -> nx.Graph:
    """
    Builds a networkx graph directly from the :CO_LINK relationships in Neo4j.
    This is the primary method for loading the graph for community detection.
    """
    print("Building graph from :CO_LINK edges in Neo4j...")
    G = nx.Graph()
    with driver.session() as session:
        result = session.run("""
            MATCH (a1:Article)-[r:CO_LINK]->(a2:Article)
            RETURN a1.id AS source, a2.id AS target, r.weight AS weight
        """)
        edges = [(r['source'], r['target'], r['weight']) for r in result]
    
    if edges:
        G.add_weighted_edges_from(edges)
    
    print(f"Graph loaded from Neo4j with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def run_louvain(G: nx.Graph) -> dict:
    """Apply Louvain community detection algorithm to find article clusters."""
    
    if G.number_of_edges() == 0:
        # No edges: assign each node to separate community
        return {node: idx for idx, node in enumerate(G.nodes())}

    partition = community_louvain.best_partition(G, weight="weight")
    print(f"Detected {len(set(partition.values()))} communities.")
    return partition


def export_communities(partition: dict, out_path: str):
    rows = [
        {"snippet_id": sid, "community_id": cid}
        for sid, cid in partition.items()
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} article-community mappings to {out_path}")


def store_communities_in_neo4j(partition: dict, driver):
    """
    Store cluster assignments in Neo4j as :Cluster nodes with :HAS relationships.
    This enables queries like 'get all articles in cluster X' directly from the graph.
    """
    print(f"Storing {len(set(partition.values()))} clusters in Neo4j...")
    
    # Group articles by cluster
    clusters = {}
    for article_id, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(article_id)
    
    with driver.session() as session:
        for cluster_id, article_ids in clusters.items():
            session.run("""
                MERGE (c:Cluster {id: $cluster_id})
                SET c.size = $size
                WITH c
                UNWIND $article_ids AS aid
                MATCH (a:Article {id: aid})
                MERGE (c)-[:HAS]->(a)
            """, {
                'cluster_id': int(cluster_id),
                'size': len(article_ids),
                'article_ids': article_ids
            })
    
    print("✅ Clusters stored in Neo4j with :HAS relationships")


def main():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )
    
    try:
        # Build graph from existing CO_LINK edges in Neo4j
        G = build_article_graph_from_neo4j(driver)
        
        # Fallback: if no CO_LINK edges exist, use old method
        if G.number_of_edges() == 0:
            print("No CO_LINK edges found in Neo4j. Using fallback entity co-occurrence method...")
            print("Loading KPI/entity data...")
            df_ent = load_kpi_entities()
            print(f"Loaded {len(df_ent)} entity rows.")
            G = build_article_graph(df_ent)
        
        # Run community detection
        partition = run_louvain(G)
        
        # Store in both parquet and Neo4j
        out_path = os.path.join(config.DATA_DIR, "article_communities.parquet")
        export_communities(partition, out_path)
        store_communities_in_neo4j(partition, driver)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()
