import os
import sys
from collections import defaultdict

import pandas as pd
import networkx as nx
from neo4j import GraphDatabase


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


try:
    # Provided by the `python-louvain` package (module name: `community`)
    from community import community_louvain  # type: ignore
except Exception:  # pragma: no cover
    community_louvain = None


def load_kpi_entities() -> pd.DataFrame:
    """Load and filter entity extraction results from parquet file."""
    
    if not os.path.exists(config.KPI_ENTITIES_FILE):
        raise FileNotFoundError(f"KPI/Entities file not found: {config.KPI_ENTITIES_FILE}")

    df = pd.read_parquet(config.KPI_ENTITIES_FILE)
    df = df[df["category"] == "Entity"].copy()
    df = df[df["entity_name"].notna() & (df["entity_name"].str.strip() != "")]
    return df


def build_article_graph(
    df_entities: pd.DataFrame,
    snippet_to_article: dict[str, str],
    all_article_ids: list[str] | None = None,
) -> nx.Graph:
    """Build article co-occurrence graph where edges represent shared entity mentions."""

    article_to_ents: dict[str, set[str]] = defaultdict(set)
    for row in df_entities.itertuples(index=False):
        sid = str(row.snippet_id)
        aid = snippet_to_article.get(sid)
        if not aid:
            continue
        article_to_ents[aid].add(str(row.entity_name).strip())

    # Initialize graph nodes. If we have a full article list, include isolated nodes too.
    G = nx.Graph()
    if all_article_ids:
        for aid in all_article_ids:
            G.add_node(aid)
            article_to_ents.setdefault(aid, set())
    else:
        for aid in article_to_ents.keys():
            G.add_node(aid)

    article_ids = list(G.nodes())
    n = len(article_ids)
    print(f"Building article graph over {n} articles...")

    # Compare articles pairwise to find shared entities (O(n^2) acceptable for demo scale).
    for i in range(n):
        a_i = article_ids[i]
        ents_i = article_to_ents[a_i]
        if not ents_i:
            continue
        for j in range(i + 1, n):
            a_j = article_ids[j]
            ents_j = article_to_ents[a_j]
            if not ents_j:
                continue
            shared = ents_i.intersection(ents_j)
            if shared:
                G.add_edge(a_i, a_j, weight=len(shared))

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def build_article_graph_from_neo4j(driver, all_article_ids: list[str] | None = None) -> nx.Graph:
    """
    Builds a networkx graph directly from the :CO_LINK relationships in Neo4j.
    This is the primary method for loading the graph for community detection.
    """
    print("Building graph from :CO_LINK edges in Neo4j...")
    G = nx.Graph()
    if all_article_ids:
        for aid in all_article_ids:
            G.add_node(aid)
    with driver.session() as session:
        result = session.run("""
            MATCH (a1:Article)-[r:CO_LINK]-(a2:Article)
            RETURN a1.id AS source, a2.id AS target, r.weight AS weight
        """)
        edges = []
        seen = set()
        for r in result:
            s = str(r["source"])
            t = str(r["target"])
            if not s or not t or s == "nan" or t == "nan":
                continue
            a, b = (s, t) if s < t else (t, s)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            edges.append((a, b, float(r["weight"] or 0.0)))
    
    if edges:
        G.add_weighted_edges_from(edges)
    
    print(f"Graph loaded from Neo4j with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def run_louvain(G: nx.Graph) -> dict:
    """Apply Louvain community detection algorithm to find article clusters."""
    
    if G.number_of_edges() == 0:
        # No edges: assign each node to separate community
        return {node: idx for idx, node in enumerate(G.nodes())}

    # Prefer python-louvain if available (fast + stable API)
    if community_louvain is not None:
        partition = community_louvain.best_partition(G, weight="weight")
        print(f"Detected {len(set(partition.values()))} communities.")
        return partition

    # Fallback: NetworkX built-in Louvain (no extra dependency).
    # Produces a list of sets, one per community.
    from networkx.algorithms.community import louvain_communities  # type: ignore

    communities = louvain_communities(G, weight="weight", seed=42)
    partition = {}
    for cid, nodes in enumerate(communities):
        for node in nodes:
            partition[node] = cid
    print(f"Detected {len(communities)} communities (networkx fallback).")
    return partition


def export_communities(partition: dict, out_path: str):
    rows = [
        {"article_id": str(aid), "community_id": int(cid)}
        for aid, cid in partition.items()
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} article-community mappings to {out_path}")


def store_communities_in_neo4j(partition: dict, driver, reset_existing: bool = True):
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
        if reset_existing:
            session.run("MATCH (c:Cluster) DETACH DELETE c")
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
    
    print("Clusters stored in Neo4j with :HAS relationships")


def main():
    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )
    
    try:
        expected_articles = None
        all_article_ids: list[str] | None = None
        snippet_to_article: dict[str, str] = {}
        if os.path.exists(config.SNIPPETS_FILE):
            try:
                df_snip = pd.read_parquet(config.SNIPPETS_FILE, columns=["snippet_id", "article_id"])
                df_snip = df_snip.dropna(subset=["snippet_id", "article_id"]).copy()
                df_snip["snippet_id"] = df_snip["snippet_id"].astype(str)
                df_snip["article_id"] = df_snip["article_id"].astype(str)
                snippet_to_article = dict(zip(df_snip["snippet_id"], df_snip["article_id"]))
                all_article_ids = sorted(df_snip["article_id"].unique().tolist())
                expected_articles = len(all_article_ids)
            except Exception:
                expected_articles = None

        # Build graph from existing CO_LINK edges in Neo4j (include isolated articles too).
        G = build_article_graph_from_neo4j(driver, all_article_ids=all_article_ids)
        
        # Fallback: CO_LINK graph may be too sparse; use entity co-occurrence across snippets.
        min_nodes = 30
        if expected_articles is not None:
            min_nodes = max(min_nodes, int(expected_articles * 0.60))

        # If we have no edges at all, community detection can't work; fallback to co-occurrence.
        # If edges exist but are very sparse, co-occurrence can still be better at demo scale.
        sparse_edges = G.number_of_edges() < max(1, int(G.number_of_nodes() * 0.5))
        if G.number_of_edges() == 0 or (G.number_of_nodes() >= min_nodes and sparse_edges):
            reason = "no CO_LINK edges" if G.number_of_edges() == 0 else f"sparse CO_LINK graph (edges={G.number_of_edges()}, nodes={G.number_of_nodes()})"
            print(f"[WARN] Using fallback entity co-occurrence method due to {reason}...")
            print("Loading KPI/entity data...")
            df_ent = load_kpi_entities()
            print(f"Loaded {len(df_ent)} entity rows.")
            G = build_article_graph(df_ent, snippet_to_article=snippet_to_article, all_article_ids=all_article_ids)
        
        # Run community detection
        partition = run_louvain(G)
        
        # Store in both parquet and Neo4j
        out_path = os.path.join(config.DATA_DIR, "article_communities.parquet")
        export_communities(partition, out_path)
        store_communities_in_neo4j(partition, driver, reset_existing=True)
        
    finally:
        driver.close()


if __name__ == "__main__":
    main()
