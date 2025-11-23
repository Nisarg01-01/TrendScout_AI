import os
import sys
from collections import defaultdict

import pandas as pd
import networkx as nx
from community import community_louvain


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


def load_kpi_entities() -> pd.DataFrame:
    """
    I'm grabbing the extraction results from our parquet file.
    I'll filter it down to just the 'Entity' rows and make sure we don't have any empty names.
    """
    if not os.path.exists(config.KPI_ENTITIES_FILE):
        raise FileNotFoundError(f"KPI/Entities file not found: {config.KPI_ENTITIES_FILE}")

    df = pd.read_parquet(config.KPI_ENTITIES_FILE)
    df = df[df["category"] == "Entity"].copy()
    df = df[df["entity_name"].notna() & (df["entity_name"].str.strip() != "")]
    return df


def build_article_graph(df_entities: pd.DataFrame) -> nx.Graph:
    """
    Here I'm constructing a network where articles are connected if they talk about the same things.
    Each node is an article (snippet_id), and the connection strength (weight) depends on how many entities they share.
    """
    # First, let's map each article to the set of entities it mentions
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

    # I'm comparing every article with every other article to find connections.
    # It's a bit heavy (O(n^2)), but for a few thousand articles, it's perfectly fine.
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


def run_louvain(G: nx.Graph) -> dict:
    """
    Time to find the communities! I'm using the Louvain method to detect clusters of related articles based on the edge weights we calculated.
    """
    if G.number_of_edges() == 0:
        # If there are no connections, everyone is in their own lonely community
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


def main():
    print("Loading KPI/entity data...")
    df_ent = load_kpi_entities()
    print(f"Loaded {len(df_ent)} entity rows.")

    G = build_article_graph(df_ent)
    partition = run_louvain(G)

    out_path = os.path.join(config.DATA_DIR, "article_communities.parquet")
    export_communities(partition, out_path)


if __name__ == "__main__":
    main()
