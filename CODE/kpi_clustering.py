"""
KPI Clustering Module - Implements G_k (KPI Graph) layer
Clusters snippets within each article cluster using HDBSCAN on embeddings
Creates :KPICluster nodes and relationships in Neo4j
"""

import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import hdbscan
import config
from typing import Dict, List

class KPIClusterer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def close(self):
        self.driver.close()
    
    def get_snippets_per_article_cluster(self) -> Dict[int, List[Dict]]:
        """Get all snippets grouped by article cluster."""
        print("Loading snippets from Neo4j...")
        
        snippets_by_cluster = {}
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)-[:HAS]->(a:Article)<-[:IN]-(s:Snippet)
                RETURN c.id as cluster_id, 
                       s.id as snippet_id,
                       s.text as text,
                       s.date as date
                ORDER BY c.id
            """)
            
            for record in result:
                cid = record['cluster_id']
                if cid not in snippets_by_cluster:
                    snippets_by_cluster[cid] = []
                
                snippets_by_cluster[cid].append({
                    'id': record['snippet_id'],
                    'text': record['text'],
                    'date': record['date']
                })
        
        print(f"[OK] Loaded snippets from {len(snippets_by_cluster)} article clusters")
        return snippets_by_cluster
    
    def cluster_snippets_in_article_cluster(self, cluster_id: int, snippets: List[Dict]) -> List[int]:
        """
        Run HDBSCAN on snippet embeddings within one article cluster.
        Returns cluster assignments for each snippet.
        """
        if len(snippets) < 5:
            # Too few snippets, put all in one KPI cluster
            return [0] * len(snippets), 1
        
        # Extract texts
        texts = [s['text'] for s in snippets]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Run HDBSCAN
        # min_cluster_size: minimum snippets per KPI theme
        # min_samples: core points needed
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(snippets) // 10),
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # HDBSCAN uses -1 for noise, convert to separate cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return labels.tolist(), n_clusters
    
    def create_snippet_similarity_edges(self, cluster_id: int, snippets: List[Dict]):
        """Create Snippet-Snippet similarity edges using cosine similarity."""
        if len(snippets) < 2:
            return 0
        
        texts = [s['text'] for s in snippets]
        snippet_ids = [s['id'] for s in snippets]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute pairwise similarities
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Create edges for top-k similar pairs (k=5)
        edges = []
        for i in range(len(snippet_ids)):
            # Get top 6 similar (including self)
            top_indices = np.argsort(similarity_matrix[i])[-6:][::-1]
            
            for j in top_indices[1:]:  # Skip self (first one)
                if similarity_matrix[i][j] > 0.7:  # Threshold
                    edges.append({
                        'sid1': snippet_ids[i],
                        'sid2': snippet_ids[j],
                        'similarity': float(similarity_matrix[i][j])
                    })
        
        # Store in Neo4j
        if edges:
            with self.driver.session() as session:
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (s1:Snippet {id: edge.sid1})
                    MATCH (s2:Snippet {id: edge.sid2})
                    MERGE (s1)-[r:SIMILAR_TO]-(s2)
                    SET r.similarity = edge.similarity
                """, {'edges': edges})
        
        return len(edges)
    
    def store_kpi_clusters(self, article_cluster_id: int, snippets: List[Dict], kpi_labels: List[int]):
        """Store KPI cluster assignments in Neo4j."""
        # Group snippets by KPI cluster
        kpi_clusters = {}
        for snippet, label in zip(snippets, kpi_labels):
            if label not in kpi_clusters:
                kpi_clusters[label] = []
            kpi_clusters[label].append(snippet['id'])
        
        with self.driver.session() as session:
            for kpi_cluster_id, snippet_ids in kpi_clusters.items():
                # Create unique ID: article_cluster_id + kpi_cluster_id
                cluster_node_id = f"ac{article_cluster_id}_kpi{kpi_cluster_id}"
                
                session.run("""
                    MERGE (kc:KPICluster {id: $cluster_id})
                    SET kc.article_cluster_id = $article_cluster_id,
                        kc.kpi_cluster_id = $kpi_cluster_id,
                        kc.size = $size
                    WITH kc
                    UNWIND $snippet_ids AS sid
                    MATCH (s:Snippet {id: sid})
                    MERGE (kc)-[:HAS]->(s)
                """, {
                    'cluster_id': cluster_node_id,
                    'article_cluster_id': article_cluster_id,
                    'kpi_cluster_id': kpi_cluster_id,
                    'size': len(snippet_ids),
                    'snippet_ids': snippet_ids
                })
    
    def run_full_pipeline(self):
        """Execute complete KPI clustering pipeline."""
        print("\n" + "="*80)
        print("STARTING KPI CLUSTERING PIPELINE (G_k)")
        print("="*80 + "\n")
        
        # Get snippets grouped by article cluster
        snippets_by_cluster = self.get_snippets_per_article_cluster()
        
        if not snippets_by_cluster:
            print("[WARN] No snippets found in graph. Run graph_build.py first.")
            return
        
        total_kpi_clusters = 0
        total_edges = 0
        
        for article_cluster_id, snippets in snippets_by_cluster.items():
            print(f"\nProcessing Article Cluster {article_cluster_id} ({len(snippets)} snippets)...")
            
            # Create similarity edges
            edge_count = self.create_snippet_similarity_edges(article_cluster_id, snippets)
            total_edges += edge_count
            print(f"  [OK] Created {edge_count} Snippet-Snippet similarity edges")
            
            # Cluster snippets
            kpi_labels, n_kpi_clusters = self.cluster_snippets_in_article_cluster(article_cluster_id, snippets)
            print(f"  [OK] Detected {n_kpi_clusters} KPI clusters (HDBSCAN)")
            
            # Store in Neo4j
            self.store_kpi_clusters(article_cluster_id, snippets, kpi_labels)
            print(f"  [OK] Stored KPI cluster assignments")
            
            total_kpi_clusters += n_kpi_clusters
        
        print("\n" + "="*80)
        print(f"[OK] KPI CLUSTERING COMPLETE")
        print(f"   Article Clusters: {len(snippets_by_cluster)}")
        print(f"   Total KPI Clusters: {total_kpi_clusters}")
        print(f"   Snippet-Snippet Edges: {total_edges}")
        print("="*80 + "\n")
        
        # Export summary
        self.export_summary(snippets_by_cluster, total_kpi_clusters)
    
    def export_summary(self, snippets_by_cluster: Dict, total_kpi_clusters: int):
        """Export KPI clustering summary to parquet."""
        rows = []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (kc:KPICluster)-[:HAS]->(s:Snippet)
                RETURN kc.id as kpi_cluster_id,
                       kc.article_cluster_id as article_cluster_id,
                       kc.size as size,
                       collect(s.id) as snippet_ids
            """)
            
            for record in result:
                rows.append({
                    'kpi_cluster_id': record['kpi_cluster_id'],
                    'article_cluster_id': record['article_cluster_id'],
                    'size': record['size'],
                    'snippet_count': len(record['snippet_ids'])
                })
        
        if rows:
            df = pd.DataFrame(rows)
            out_path = os.path.join(config.DATA_DIR, "kpi_clusters.parquet")
            df.to_parquet(out_path, index=False)
            print(f"[OK] Saved KPI cluster summary to {out_path}")


def main():
    clusterer = KPIClusterer()
    try:
        clusterer.run_full_pipeline()
    finally:
        clusterer.close()


if __name__ == "__main__":
    main()
