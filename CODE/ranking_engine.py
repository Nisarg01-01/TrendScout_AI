"""
Ranking & Scoring Module for TrendScout AI
Implements cluster-wise startup scoring based on:
- Connectivity (PageRank/degree within cluster)
- KPI stance score (positive/negative/neutral with confidence)
- Recency boost (30-60 day window)
- Investor quality (optional)

Composite Score = α·Centrality + β·KPI_Stance + γ·Recency + δ·InvestorQuality
"""

import os
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
import config
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class StartupRanker:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        
        # Scoring weights
        self.alpha = 0.3  # Centrality
        self.beta = 0.4   # KPI stance
        self.gamma = 0.2  # Recency
        self.delta = 0.1  # Investor quality
        
        # Temporal windows
        self.recency_window = 60  # days
        self.tau = 30  # decay parameter
    
    def close(self):
        self.driver.close()
    
    def compute_centrality_per_cluster(self) -> Dict[int, Dict[str, float]]:
        """
        Compute PageRank for entities within each cluster.
        Returns: {cluster_id: {entity_name: pagerank_score}}
        """
        print("Computing PageRank centrality per cluster...")
        
        cluster_centrality = {}
        
        with self.driver.session() as session:
            # Get all clusters
            result = session.run("MATCH (c:Cluster) RETURN c.id as cluster_id")
            cluster_ids = [r['cluster_id'] for r in result]
            
            for cid in cluster_ids:
                # Build subgraph for this cluster
                # Articles in cluster + entities they mention
                result = session.run("""
                    MATCH (c:Cluster {id: $cid})-[:HAS]->(a:Article)-[r:MENTIONS]->(e:Entity)
                    RETURN a.id as article_id, e.name as entity_name, r.stance as stance
                """, {'cid': cid})
                
                # Build bipartite graph: Article <-> Entity
                G = nx.Graph()
                for record in result:
                    G.add_edge(f"a_{record['article_id']}", f"e_{record['entity_name']}")
                
                if G.number_of_nodes() == 0:
                    continue
                
                # Compute PageRank
                try:
                    pr = nx.pagerank(G, weight='weight')
                    
                    # Extract entity scores only
                    entity_scores = {
                        k.replace('e_', ''): v 
                        for k, v in pr.items() 
                        if k.startswith('e_')
                    }
                    
                    cluster_centrality[cid] = entity_scores
                except:
                    cluster_centrality[cid] = {}
        
        print(f"✅ Computed centrality for {len(cluster_centrality)} clusters")
        return cluster_centrality
    
    def compute_kpi_stance_scores(self) -> Dict[str, Dict[int, float]]:
        """
        Compute KPI stance scores for each entity in each cluster.
        Aggregates positive/negative KPIs with recency decay.
        Returns: {entity_name: {cluster_id: stance_score}}
        """
        print("Computing KPI stance scores...")
        
        entity_stance = defaultdict(lambda: defaultdict(float))
        now = datetime.now()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)-[:HAS]->(a:Article)-[:MENTIONS]->(e:Entity)
                MATCH (a)<-[:IN]-(s:Snippet)-[:ABOUT]->(k:KPI)
                WHERE k.stance IS NOT NULL AND k.date IS NOT NULL
                RETURN e.name as entity_name, c.id as cluster_id, 
                       k.stance as stance, k.date as date
            """)
            
            for record in result:
                entity = record['entity_name']
                cid = record['cluster_id']
                stance = record['stance']
                
                try:
                    date_str = record['date']
                    kpi_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    days_old = (now - kpi_date).days
                    
                    # Recency decay
                    recency_weight = np.exp(-days_old / self.tau)
                    
                    # Weighted stance
                    entity_stance[entity][cid] += stance * recency_weight
                except:
                    # If date parsing fails, use stance without decay
                    entity_stance[entity][cid] += stance
        
        print(f"✅ Computed stance scores for {len(entity_stance)} entities")
        return dict(entity_stance)
    
    def compute_investor_quality_scores(self) -> Dict[str, Dict[int, float]]:
        """
        Compute investor quality scores for each entity in each cluster.
        Returns: {entity_name: {cluster_id: investor_score}}
        """
        print("Computing investor quality scores...")
        
        entity_investor_scores = defaultdict(lambda: defaultdict(float))
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)-[:HAS]->(a:Article)-[:MENTIONS]->(e:Entity)
                MATCH (e)-[:FUNDED_BY]->(i:Investor)
                RETURN e.name as entity_name, c.id as cluster_id,
                       avg(i.prestige) as avg_prestige,
                       max(i.prestige) as max_prestige,
                       count(i) as investor_count
            """)
            
            for record in result:
                entity = record['entity_name']
                cid = record['cluster_id']
                
                # Combine average and max prestige with investor count boost
                avg_prestige = record['avg_prestige'] or 0.0
                max_prestige = record['max_prestige'] or 0.0
                investor_count = record['investor_count'] or 0
                
                # Score: weighted average of max and avg, with count bonus
                score = (0.6 * max_prestige + 0.4 * avg_prestige) * min(1.0, investor_count / 3.0)
                
                entity_investor_scores[entity][cid] = score
        
        print(f"✅ Computed investor scores for {len(entity_investor_scores)} entities")
        return dict(entity_investor_scores)
    
    def compute_recency_scores(self) -> Dict[str, Dict[int, float]]:
        """
        Compute recency boost: more weight to mentions in last 30-60 days.
        Returns: {entity_name: {cluster_id: recency_score}}
        """
        print("Computing recency scores...")
        
        entity_recency = defaultdict(lambda: defaultdict(float))
        now = datetime.now()
        cutoff = now - timedelta(days=self.recency_window)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)-[:HAS]->(a:Article)-[:MENTIONS]->(e:Entity)
                WHERE a.published IS NOT NULL AND a.published <> ""
                RETURN e.name as entity_name, c.id as cluster_id, a.published as date
            """)
            
            for record in result:
                entity = record['entity_name']
                cid = record['cluster_id']
                
                try:
                    date_str = record['date']
                    pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    pub_date = pub_date.replace(tzinfo=None) # Make it naive to compare with naive cutoff
                    if pub_date >= cutoff:
                        days_since = (now - pub_date).days
                        boost = 1.0 - (days_since / self.recency_window)
                        entity_recency[entity][cid] += boost
                except:
                    pass
        
        print(f"✅ Computed recency for {len(entity_recency)} entities")
        return dict(entity_recency)
    
    def compute_composite_scores(self) -> pd.DataFrame:
        """
        Compute final composite scores for all entities in all clusters.
        Returns DataFrame with columns: entity_name, cluster_id, score, rank
        """
        print("\n🎯 Computing composite startup scores...")
        
        centrality = self.compute_centrality_per_cluster()
        stance = self.compute_kpi_stance_scores()
        recency = self.compute_recency_scores()
        investor = self.compute_investor_quality_scores()
        
        # Combine scores
        scores = []
        
        # Get all entities per cluster
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Cluster)-[:HAS]->(a:Article)-[:MENTIONS]->(e:Entity)
                RETURN DISTINCT c.id as cluster_id, e.name as entity_name
            """)
            
            for record in result:
                entity = record['entity_name']
                cid = record['cluster_id']
                
                # Get individual scores (default to 0 if missing)
                cent_score = centrality.get(cid, {}).get(entity, 0.0)
                stance_score = stance.get(entity, {}).get(cid, 0.0)
                rec_score = recency.get(entity, {}).get(cid, 0.0)
                inv_score = investor.get(entity, {}).get(cid, 0.0)
                
                # Normalize stance (can be negative)
                stance_norm = (stance_score + 10) / 20  # Map [-10, 10] to [0, 1]
                stance_norm = max(0, min(1, stance_norm))
                
                # Composite score
                composite = (
                    self.alpha * cent_score +
                    self.beta * stance_norm +
                    self.gamma * rec_score +
                    self.delta * inv_score
                )
                
                scores.append({
                    'entity_name': entity,
                    'cluster_id': cid,
                    'centrality': cent_score,
                    'kpi_stance': stance_score,
                    'recency': rec_score,
                    'investor_quality': inv_score,
                    'composite_score': composite
                })
        
        df = pd.DataFrame(scores)
        
        # Rank within each cluster
        df['rank'] = df.groupby('cluster_id')['composite_score'].rank(ascending=False, method='dense')
        df = df.sort_values(['cluster_id', 'rank'])
        
        print(f"\n✅ Computed scores for {len(df)} entity-cluster pairs")
        return df
    
    def store_scores_in_neo4j(self, df: pd.DataFrame):
        """Store computed scores back into Neo4j for fast retrieval."""
        print("\nStoring scores in Neo4j...")
        
        with self.driver.session() as session:
            for _, row in df.iterrows():
                session.run("""
                    MATCH (c:Cluster {id: $cluster_id})-[:HAS]->(a:Article)-[:MENTIONS]->(e:Entity {name: $entity_name})
                    WITH e, c, $score as score, $rank as rank
                    MERGE (e)-[r:RANKED_IN]->(c)
                    SET r.score = score, r.rank = rank,
                        r.centrality = $centrality,
                        r.kpi_stance = $kpi_stance,
                        r.recency = $recency,
                        r.investor_quality = $investor_quality
                """, {
                    'cluster_id': int(row['cluster_id']),
                    'entity_name': row['entity_name'],
                    'score': float(row['composite_score']),
                    'rank': int(row['rank']),
                    'centrality': float(row['centrality']),
                    'kpi_stance': float(row['kpi_stance']),
                    'recency': float(row['recency']),
                    'investor_quality': float(row['investor_quality'])
                })
        
        print("✅ Scores stored with :RANKED_IN relationships")
    
    def export_rankings(self, df: pd.DataFrame):
        """Export rankings to parquet file."""
        out_path = os.path.join(config.DATA_DIR, "entity_rankings.parquet")
        df.to_parquet(out_path, index=False)
        print(f"✅ Saved rankings to {out_path}")
    
    def get_top_entities_in_cluster(self, cluster_id: int, top_k: int = 10) -> pd.DataFrame:
        """Retrieve top-k ranked entities for a specific cluster."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[r:RANKED_IN]->(c:Cluster {id: $cluster_id})
                RETURN e.name as entity, r.score as score, r.rank as rank,
                       r.centrality as centrality, r.kpi_stance as kpi_stance,
                       r.recency as recency
                ORDER BY r.rank ASC
                LIMIT $top_k
            """, {'cluster_id': cluster_id, 'top_k': top_k})
            
            rows = []
            for record in result:
                rows.append(dict(record))
            
            return pd.DataFrame(rows)


def main():
    ranker = StartupRanker()
    
    try:
        # Compute all scores
        df_rankings = ranker.compute_composite_scores()
        
        # Display top 20 overall
        print("\n" + "="*80)
        print("TOP 20 STARTUPS BY COMPOSITE SCORE")
        print("="*80)
        top_20 = df_rankings.nlargest(20, 'composite_score')
        print(top_20[['entity_name', 'cluster_id', 'composite_score', 'rank']].to_string(index=False))
        
        # Store and export
        ranker.store_scores_in_neo4j(df_rankings)
        ranker.export_rankings(df_rankings)
        
        print("\n✅ Ranking pipeline complete!")
        
    finally:
        ranker.close()


if __name__ == "__main__":
    main()
