import os
import pandas as pd
import numpy as np
import config
from typing import List, Dict, Any
import json
from datetime import datetime
try:
    import ollama  # type: ignore
except ModuleNotFoundError:
    ollama = None

try:
    from neo4j import GraphDatabase  # type: ignore
except ModuleNotFoundError:
    GraphDatabase = None

try:
    import chromadb  # type: ignore
except ModuleNotFoundError:
    chromadb = None

import re


class CommunityAnalytics:
    """Bridge to community analysis data from parquet files and Neo4j.
    Provides access to community memberships, SWOT data, temporal trends, and rankings.
    """

    def __init__(self, graph_store=None):
        self._loaded = False
        self.graph_store = graph_store

    def _ensure_loaded(self):
        if self._loaded:
            return

        base = config.DATA_DIR
        comm_path = os.path.join(base, "article_communities.parquet")
        ent_path = os.path.join(base, "community_entity_summary.parquet")
        swot_path = os.path.join(base, "community_swot_summary.parquet")
        temp_path = os.path.join(base, "community_temporal_summary.parquet")
        # Current file name (plural) written by CODE/ranking_engine.py
        rank_path = os.path.join(base, "entity_rankings.parquet")
        # Backward compatibility (older name)
        rank_path_legacy = os.path.join(base, "entity_ranking.parquet")
        temp_feat_path = os.path.join(base, "temporal_features.parquet")
        forecast_path = os.path.join(base, "community_forecast.parquet")

        # Minimal requirement: communities (article_communities.parquet).
        # entity summary is optional because we can query Neo4j directly when available.
        if not os.path.exists(comm_path):
            self._loaded = True
            self.df_comm = None
            self.df_ent = None
            self.df_swot = None
            self.df_temp = None
            self.df_rank = None
            self.df_temp_feat = None
            self.df_forecast = None
            return

        self.df_comm = pd.read_parquet(comm_path)
        self.df_ent = pd.read_parquet(ent_path) if os.path.exists(ent_path) else None
        self.df_swot = pd.read_parquet(swot_path) if os.path.exists(swot_path) else None
        self.df_temp = pd.read_parquet(temp_path) if os.path.exists(temp_path) else None
        if os.path.exists(rank_path):
            self.df_rank = pd.read_parquet(rank_path)
        elif os.path.exists(rank_path_legacy):
            self.df_rank = pd.read_parquet(rank_path_legacy)
        else:
            self.df_rank = None
        self.df_temp_feat = pd.read_parquet(temp_feat_path) if os.path.exists(temp_feat_path) else None
        self.df_forecast = pd.read_parquet(forecast_path) if os.path.exists(forecast_path) else None
        self._loaded = True

    def get_entity_communities(self, entity_name: str) -> List[int]:
        """Get community IDs where the specified entity appears."""
        
        self._ensure_loaded()
        if self.df_ent is not None:
            sub = self.df_ent[self.df_ent["entity_name"].astype(str).str.lower() == entity_name.lower()]
            return sorted(sub["community_id"].dropna().unique().tolist())

        # Fallback: derive communities from Neo4j (Entity mentioned by Article in Cluster).
        if self.graph_store is None:
            return []
        try:
            with self.graph_store.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(a:Article)<-[:HAS]-(c:Cluster)
                    RETURN DISTINCT c.id as cluster_id
                    """,
                    {"entity_name": entity_name},
                )
                ids: List[int] = []
                for r in result:
                    try:
                        ids.append(int(r.get("cluster_id")))
                    except Exception:
                        continue
                return sorted(set(ids))
        except Exception:
            return []

    def get_top_entities_in_community(self, community_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-ranked entities in a community from Neo4j or parquet fallback."""
        
        # Try Neo4j first for real-time rankings
        try:
            with self.graph_store.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)-[r:RANKED_IN]->(c:Cluster {id: $cluster_id})
                    RETURN e.name as entity_name,
                           r.score as score,
                           r.rank as rank,
                           r.centrality as centrality,
                           r.kpi_stance as kpi_stance,
                           r.recency as recency,
                           r.investor_quality as investor_quality
                    ORDER BY r.rank ASC
                    LIMIT $top_k
                """, {'cluster_id': community_id, 'top_k': top_k})
                
                entities = []
                for record in result:
                    entities.append({
                        "entity_name": record['entity_name'],
                        "rank": int(record['rank']),
                        "score": float(record['score']),
                        "centrality": float(record['centrality']) if record['centrality'] else 0.0,
                        "kpi_stance": float(record['kpi_stance']) if record['kpi_stance'] else 0.0,
                        "recency": float(record['recency']) if record['recency'] else 0.0,
                        "investor_quality": float(record['investor_quality']) if record['investor_quality'] else 0.0,
                    })
                
                if entities:
                    return entities
        except:
            pass  # Fall back to parquet
        
        # Fallback to old method
        self._ensure_loaded()
        if self.df_ent is None:
            return []
        sub = self.df_ent[self.df_ent["community_id"] == community_id]
        if sub.empty:
            return []
        sub = sub.sort_values("mentions", ascending=False).head(top_k)
        return [
            {
                "entity_name": row["entity_name"],
                "mentions": int(row["mentions"]),
                "avg_stance": float(row["avg_stance"]) if pd.notna(row["avg_stance"]) else None,
            }
            for _, row in sub.iterrows()
        ]

    def get_swot_for_community(self, community_id: int) -> Dict[str, int]:
        """Get SWOT distribution (Strengths, Weaknesses, Opportunities, Threats counts) for a community."""
        
        self._ensure_loaded()
        if self.df_swot is None:
            return {}
        sub = self.df_swot[self.df_swot["community_id"] == community_id]
        if sub.empty:
            return {}
        agg = sub.groupby("swot_type")["count"].sum().to_dict()
        return {str(k): int(v) for k, v in agg.items()}

    def get_quantifiable_swot_for_entity(self, entity_name: str) -> Dict[str, Any]:
        """
        Get quantifiable SWOT metrics for a single entity.
        - SWOT Volume (counts)
        - KPI Stance Score (from ranking data)
        - KPI Polarity Delta (from temporal features)
        """
        self._ensure_loaded()
        metrics = {}

        # Get Stance Score from ranking data
        if self.df_rank is not None:
            entity_rank_data = self.df_rank[self.df_rank["entity_name"].astype(str).str.lower() == entity_name.lower()]
            if not entity_rank_data.empty:
                cols = set(entity_rank_data.columns)
                score_col = "score" if "score" in cols else ("composite_score" if "composite_score" in cols else None)
                stance_col = "kpi_stance" if "kpi_stance" in cols else ("avg_stance" if "avg_stance" in cols else None)

                if score_col is not None:
                    top_rank = entity_rank_data.sort_values(score_col, ascending=False).iloc[0]
                else:
                    top_rank = entity_rank_data.iloc[0]

                if stance_col is not None:
                    try:
                        metrics["kpi_stance_score"] = float(top_rank.get(stance_col, 0.0) or 0.0)
                    except Exception:
                        metrics["kpi_stance_score"] = 0.0
                else:
                    metrics["kpi_stance_score"] = 0.0

        # Get Polarity Delta from temporal features
        if self.df_temp_feat is not None:
            entity_temp_data = self.df_temp_feat[self.df_temp_feat['entity_name'].str.lower() == entity_name.lower()]
            if not entity_temp_data.empty:
                metrics['kpi_polarity_delta'] = entity_temp_data.iloc[0].get('kpi_polarity_delta', 0.0)
        
        return metrics

    def get_temporal_for_entity(self, entity_name: str, community_id: int | None = None) -> List[Dict[str, Any]]:
        """Get monthly mention counts to track temporal trends for an entity or community."""
        
        self._ensure_loaded()
        if self.df_temp is None:
            return []
        # Currently df_temp only has community_id, year_month, entity_mentions
        # so we expose community-level trend, optionally filtered by community.
        if community_id is not None:
            sub = self.df_temp[self.df_temp["community_id"] == community_id]
        else:
            sub = self.df_temp
        if sub.empty:
            return []
        sub = sub.sort_values("year_month")
        return [
            {
                "year_month": row["year_month"],
                "entity_mentions": int(row["entity_mentions"]),
            }
            for _, row in sub.iterrows()
        ]

    def get_top_trending_entities(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top trending entities based on our calculated score.
        """
        self._ensure_loaded()
        if self.df_rank is None:
            return []

        cols = set(self.df_rank.columns)

        # Legacy format (older analytics pipeline)
        if {"rank", "entity_name", "score"}.issubset(cols):
            sub = self.df_rank.sort_values("rank").head(top_k)
            out: List[Dict[str, Any]] = []
            for _, row in sub.iterrows():
                out.append(
                    {
                        "rank": int(row.get("rank", 0) or 0),
                        "entity_name": row.get("entity_name"),
                        "score": float(row.get("score", 0.0) or 0.0),
                        "mentions": int(row.get("mentions", 0) or 0) if "mentions" in cols else None,
                        "growth_slope": float(row.get("slope", 0.0) or 0.0) if "slope" in cols else None,
                        "cluster_id": int(row.get("cluster_id", 0) or 0) if "cluster_id" in cols else None,
                    }
                )
            return out

        # Current format written by CODE/ranking_engine.py
        if {"entity_name", "composite_score"}.issubset(cols):
            try:
                sub = (
                    self.df_rank.sort_values("composite_score", ascending=False)
                    .dropna(subset=["entity_name"])
                    .drop_duplicates(subset=["entity_name"], keep="first")
                    .head(top_k)
                )
            except Exception:
                sub = self.df_rank.sort_values("composite_score", ascending=False).head(top_k)
            out2: List[Dict[str, Any]] = []
            for i, (_, row) in enumerate(sub.iterrows(), start=1):
                out2.append(
                    {
                        # `rank` in entity_rankings.parquet is cluster-local (often repeats 1,2,3...).
                        # For "top trending" we want a global list, so use enumeration.
                        "rank": int(i),
                        "entity_name": row.get("entity_name"),
                        "score": float(row.get("composite_score", 0.0) or 0.0),
                        "cluster_id": int(row.get("cluster_id", 0) or 0),
                        "recency": float(row.get("recency", 0.0) or 0.0) if "recency" in cols else None,
                        "centrality": float(row.get("centrality", 0.0) or 0.0) if "centrality" in cols else None,
                        "growth_slope": None,
                    }
                )
            return out2

        return []

    def get_community_forecast(self, community_id: int) -> Dict[str, Any]:
        """
        Get the growth forecast for a specific community.
        """
        self._ensure_loaded()
        if self.df_forecast is None:
            return {}
        
        sub = self.df_forecast[self.df_forecast["community_id"] == community_id]
        if sub.empty:
            return {}
        
        row = sub.iloc[0]
        return {
            "slope": float(row["slope"]),
            "predicted_next_month": float(row["predicted_next_month"])
        }

class VectorStore:
    def __init__(self):
        if chromadb is None:
            print("Vector store unavailable: 'chromadb' is not installed.")
            self.client = None
            self.collection = None
            return

        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        try:
            self.collection = self.client.get_collection("trendscout_snippets")
            print(f"Vector Store loaded: {self.collection.count()} documents from ChromaDB.")
        except ValueError:
            print("Vector Store collection not found. Run load_chroma.py first.")
            self.collection = None

    def clean_text(self, text: str) -> str:
        """Clean text by removing flattened table data and excessive pricing information."""
        
        # Pattern: Multiple occurrences of prices ($X.XX) in close proximity
        # If a segment has more than 3 prices, it's likely a flattened table.
        if text.count('$') > 2:
            # Pattern to detect flattened table rows with multiple prices
            # Matches: (Words/Symbols) + $Price + (Words/Symbols) + $Price ...
            pattern = r'((?:[A-Za-z0-9\.\(\)\-\>\<\=]+\s+){1,8}\$\d+\.\d+(?:\s+|$)){2,}'
            text = re.sub(pattern, ' [TABLE DATA REMOVED] ', text)
            
            # Fallback: if excessive prices remain, remove text between dollar signs
            if text.count('$') > 4:
                 text = re.sub(r'\$.*?\$', ' [PRICING] ', text)
        return text

    def search(self, query: str, top_k: int = 5, recency_bias: bool = False) -> List[Dict[str, Any]]:
        if self.collection is None:
            return []

        if ollama is None:
            raise RuntimeError("Ollama python package is not installed. Run: pip install -r requirements.txt")

        # Embed query
        response = ollama.embeddings(model=config.LLM_MODEL, prompt=query)
        query_embedding = response['embedding']

        # Query ChromaDB
        # Fetch more results if recency_bias is enabled for re-ranking
        fetch_k = top_k * 3 if recency_bias else top_k
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results['ids'][0]:
            return []

        # Process results
        processed_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (lower distance = higher similarity)
            score = 1 - distance 

            published_str = metadata.get('published', '')
            
            # Recency Logic
            final_score = score
            if recency_bias and published_str:
                try:
                    pub_date = pd.to_datetime(published_str, errors='coerce', utc=True)
                    if pd.notna(pub_date):
                        now = pd.Timestamp.now(tz='UTC')
                        days_old = (now - pub_date).days
                        if days_old < 0: days_old = 0
                        # Decay: 100 days = 0.66 weight
                        decay_factor = 1 / (1 + 0.005 * days_old)
                        final_score = score * decay_factor
                except:
                    pass

            processed_results.append({
                "score": float(final_score),
                "text": self.clean_text(text),
                "source": metadata.get('source', 'Unknown'),
                "published": published_str,
                "title": metadata.get("title", "Unknown"),
                "link": metadata.get("link", ""),
                "article_id": metadata.get("article_id", ""),
                "snippet_id": metadata.get("snippet_id", ""),
            })

        # Sort by final score and take top_k
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        return processed_results[:top_k]

class GraphStore:
    def __init__(self):
        self.driver = None
        self.has_has_kpi = False
        if GraphDatabase is None:
            print("Graph store unavailable: 'neo4j' is not installed.")
            return
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            self.verify_connection()
            self.has_has_kpi = self._relationship_exists("HAS_KPI")
        except Exception as e:
            print(f"Graph Store connection failed: {e}")

    def verify_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")

    def _relationship_exists(self, rel_type: str) -> bool:
        if not self.driver:
            return False
        rt = str(rel_type or "").strip()
        if not rt:
            return False
        with self.driver.session() as session:
            try:
                # Avoid Neo4j notifications like "relationship type does not exist" by first
                # checking the schema for the relationship type.
                types: set[str] = set()
                try:
                    res = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t")
                    types = {str(r["t"]) for r in res if r and r.get("t") is not None}
                except Exception:
                    res = session.run("CALL db.relationshipTypes()")
                    # Some Neo4j versions return a single column record; accept either dict-like or positional.
                    for r in res:
                        if isinstance(r, dict) and "relationshipType" in r:
                            types.add(str(r["relationshipType"]))
                        else:
                            try:
                                types.add(str(r[0]))
                            except Exception:
                                pass

                if rt not in types:
                    return False

                q = f"MATCH ()-[r:{rt}]->() RETURN count(r) as c"
                rec = session.run(q).single()
                return bool(rec and (rec.get("c") or 0) > 0)
            except Exception:
                return False

    def close(self):
        if self.driver:
            self.driver.close()

    def get_entity_types(self, names: List[str]) -> Dict[str, str]:
        """Fetch entity types for a list of entity names from Neo4j."""
        if not self.driver or not names:
            return {}
        cleaned = [str(n).strip() for n in names if isinstance(n, str) and str(n).strip()]
        if not cleaned:
            return {}
        q = """
        UNWIND $names AS name
        MATCH (e:Entity {name: name})
        RETURN e.name as name, e.type as type
        """
        out: Dict[str, str] = {}
        with self.driver.session() as session:
            try:
                res = session.run(q, names=cleaned)
                for r in res:
                    n = str(r.get("name") or "").strip()
                    t = str(r.get("type") or "").strip()
                    if n:
                        out[n] = t
            except Exception:
                return {}
        return out

    def get_kpi_breakdown_for_entity(self, entity_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get detailed KPI information for an entity from the Snippet-KPI graph.
        Returns structured KPIs organized by type: funding, hiring, partnership, product.
        """
        if not self.driver:
            return {"funding": [], "hiring": [], "partnership": [], "product": []}
        if not self.has_has_kpi:
            # Avoid Neo4j warnings when KPI graph isn't present yet.
            return {
                "funding": [],
                "hiring": [],
                "layoffs": [],
                "partnership": [],
                "product": [],
                "acquisition": [],
                "competition": [],
                "regulation": [],
                "lawsuit": [],
                "security": [],
                "outage": [],
                "pricing": [],
                "policy": [],
                "other": [],
            }
        
        query = """
        MATCH (e:Entity {name: $entity_name})-[:HAS_KPI]->(k:KPI)<-[:ABOUT]-(s:Snippet)-[:IN]->(a:Article)
        WITH s, k, a
        ORDER BY k.date DESC
        LIMIT $limit
        RETURN s.text as snippet_text,
               k.type as kpi_type,
               k.stance as stance,
               k.date as published,
               k.value as kpi_value,
               coalesce(k.source_url, a.link) as source_url,
               a.title as title,
               a.source as source
        """
        
        result_dict = {
            "funding": [],
            "hiring": [],
            "layoffs": [],
            "partnership": [],
            "product": [],
            "acquisition": [],
            "competition": [],
            "regulation": [],
            "lawsuit": [],
            "security": [],
            "outage": [],
            "pricing": [],
            "policy": [],
            "other": [],
        }
        
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name, limit=limit)
            for record in result:
                kpi_type_raw = record.get('kpi_type', 'other')
                kpi_type = str(kpi_type_raw).strip().lower() if kpi_type_raw is not None else "other"
                if kpi_type not in result_dict:
                    kpi_type = "other"
                kpi_info = {
                    "snippet": record['snippet_text'],
                    "stance": float(record['stance']) if record['stance'] else 0.0,
                    "published": record['published'],
                    "kpi_value": record['kpi_value'],
                    "source_url": record.get('source_url'),
                    "title": record.get('title'),
                    "source": record.get('source'),
                }
                
                result_dict[kpi_type].append(kpi_info)
        
        return result_dict

    def get_entity_rankings(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch :RANKED_IN rows (score breakdown) for an entity."""
        if not self.driver:
            return []
        query = """
        MATCH (e:Entity {name: $entity_name})-[r:RANKED_IN]->(c:Cluster)
        RETURN c.id as cluster_id,
               r.rank as rank,
               r.score as composite_score,
               r.centrality as centrality,
               r.kpi_stance as kpi_stance,
               r.kpi_momentum as kpi_momentum,
               r.kpi_risk as kpi_risk,
               r.recency as recency,
               r.investor_quality as investor_quality
        ORDER BY r.score DESC
        LIMIT $limit
        """
        out: List[Dict[str, Any]] = []
        with self.driver.session() as session:
            try:
                res = session.run(query, entity_name=str(entity_name), limit=int(limit))
                for r in res:
                    out.append(
                        {
                            "cluster_id": r.get("cluster_id"),
                            "rank": r.get("rank"),
                            "composite_score": r.get("composite_score"),
                            "centrality": r.get("centrality"),
                            "kpi_stance": r.get("kpi_stance"),
                            "kpi_momentum": r.get("kpi_momentum"),
                            "kpi_risk": r.get("kpi_risk"),
                            "recency": r.get("recency"),
                            "investor_quality": r.get("investor_quality"),
                        }
                    )
            except Exception:
                return []
        return out

    def get_recent_articles_for_entity(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent articles mentioning the entity (for citations when KPI edges are sparse)."""
        if not self.driver:
            return []
        query = """
        MATCH (e:Entity {name: $entity_name})<-[r:MENTIONS]-(a:Article)
        RETURN a.title as title,
               a.source as source,
               a.published as published,
               coalesce(a.link, a.canonical_url) as source_url,
               r.stance as stance
        ORDER BY a.published DESC
        LIMIT $limit
        """
        out: List[Dict[str, Any]] = []
        with self.driver.session() as session:
            try:
                res = session.run(query, entity_name=str(entity_name), limit=int(limit))
                for r in res:
                    out.append(
                        {
                            "title": r.get("title"),
                            "source": r.get("source"),
                            "published": r.get("published"),
                            "source_url": r.get("source_url"),
                            "stance": r.get("stance"),
                        }
                    )
            except Exception:
                return []
        return out
    
    def get_investor_quality_for_entity(self, entity_name: str) -> Dict[str, Any]:
        """
        Get investor quality information for an entity.
        Returns list of investors with prestige scores and aggregate quality.
        """
        if not self.driver:
            return {"investors": [], "avg_prestige": 0.0}
        
        query = """
        MATCH (e:Entity {name: $entity_name})-[:FUNDED_BY]->(inv:Investor)
        RETURN inv.name as investor_name,
               inv.prestige as prestige
        ORDER BY inv.prestige DESC
        """
        
        investors = []
        total_prestige = 0.0
        
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name)
            for record in result:
                prestige = float(record['prestige']) if record['prestige'] else 0.0
                investors.append({
                    "name": record['investor_name'],
                    "prestige": prestige
                })
                total_prestige += prestige
        
        avg_prestige = total_prestige / len(investors) if investors else 0.0
        
        return {
            "investors": investors,
            "avg_prestige": avg_prestige,
            "count": len(investors)
        }
    
    def get_entities_in_cluster(self, cluster_id: int) -> List[str]:
        """
        Get all entity names that belong to a specific cluster.
        """
        if not self.driver:
            return []
        
        query = """
        MATCH (e:Entity)-[:RANKED_IN]->(c:Cluster {id: $cluster_id})
        RETURN DISTINCT e.name as entity_name
        """
        
        entity_names = []
        with self.driver.session() as session:
            result = session.run(query, cluster_id=cluster_id)
            for record in result:
                entity_names.append(record['entity_name'])
        
        return entity_names

    def get_entity_context(self, entity_names: List[str]) -> List[str]:
        """Get direct relationships and SWOT data for a list of entities."""
        if not self.driver or not entity_names:
            return []

        if self.has_has_kpi:
            query = """
            UNWIND $names AS name
            MATCH (e:Entity {name: name})<-[r:MENTIONS]-(a:Article)
            OPTIONAL MATCH (a)-[:BELONGS_TO]->(i:Industry)
            OPTIONAL MATCH (e)-[:HAS_KPI]->(k:KPI)<-[:ABOUT]-(s:Snippet)-[:IN]->(a)
            RETURN e.name as entity, i.name as industry, 
                   a.title as title,
                   r.stance as stance,
                   a.swot_Strength as strength,
                   a.swot_Weakness as weakness,
                   a.swot_Opportunity as opportunity,
                   a.swot_Threat as threat,
                   collect(DISTINCT {type: k.type, value: k.value, date: k.date, stance: k.stance}) as kpis
            LIMIT 20
            """
        else:
            query = """
            UNWIND $names AS name
            MATCH (e:Entity {name: name})<-[r:MENTIONS]-(a:Article)
            OPTIONAL MATCH (a)-[:BELONGS_TO]->(i:Industry)
            RETURN e.name as entity, i.name as industry, 
                   a.title as title,
                   r.stance as stance,
                   a.swot_Strength as strength,
                   a.swot_Weakness as weakness,
                   a.swot_Opportunity as opportunity,
                   a.swot_Threat as threat
            LIMIT 20
            """
        
        context = []
        with self.driver.session() as session:
            result = session.run(query, names=entity_names)
            for record in result:
                entity = record['entity']
                # We include the Title to give context to the mention
                info = f"Entity '{entity}' mentioned in article '{record['title']}' (Stance: {record['stance']})."
                if record['industry']:
                    info += f" Industry: {record['industry']}."
                
                # NEW: Add SWOT details if they exist
                swot_parts = []
                if record['strength']: swot_parts.append(f"Strength: {record['strength']}")
                if record['weakness']: swot_parts.append(f"Weakness: {record['weakness']}")
                if record['opportunity']: swot_parts.append(f"Opportunity: {record['opportunity']}")
                if record['threat']: swot_parts.append(f"Threat: {record['threat']}")
                if swot_parts:
                    info += " | SWOT: " + ", ".join(swot_parts)

                if self.has_has_kpi:
                    kpis = [k for k in record.get('kpis', []) if k and k.get('type') and k.get('value')]
                    if kpis:
                        kpis = kpis[:4]
                        k_list = [f"{k['type']}: {k['value']}" for k in kpis]
                        info += f" | Signals: {', '.join(k_list)}"
                
                context.append(info)
        return context

    def get_industry_entities(self, industry_name: str) -> List[str]:
        """Get top entities and trends for an industry."""
        if not self.driver:
            return []
            
        query = """
        MATCH (i:Industry {name: $name})<-[:BELONGS_TO]-(a:Article)-[r:MENTIONS]->(e:Entity)
        RETURN e.name as entity, count(a) as mentions, avg(r.stance) as avg_sentiment
        ORDER BY mentions DESC
        LIMIT 10
        """
        context = []
        with self.driver.session() as session:
            result = session.run(query, name=industry_name)
            entities = []
            for record in result:
                sentiment = record['avg_sentiment']
                sent_str = f"{sentiment:.2f}" if sentiment is not None else "N/A"
                entities.append(f"{record['entity']} (Mentions: {record['mentions']}, Sentiment: {sent_str})")
            
            if entities:
                context.append(f"Top Entities in {industry_name}: {', '.join(entities)}")
                
        # Also get general trends
        query_trends = """
        MATCH (i:Industry {name: $name})<-[:BELONGS_TO]-(a:Article)
        RETURN a.swot_Opportunity as opportunities, a.swot_Threat as threats
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query_trends, name=industry_name)
            for record in result:
                if record['opportunities']:
                    context.append(f"Opportunity in {industry_name}: {', '.join(record['opportunities'])}")
        
        return context

class TrendScoutBackend:
    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.community_analytics = CommunityAnalytics(graph_store=self.graph_store)
        
        # Product-to-Company mapping for validation
        self.product_company_map = {
            'grok': 'xAI',
            'chatgpt': 'OpenAI',
            'gpt-4': 'OpenAI',
            'gpt-4o': 'OpenAI',
            'o1': 'OpenAI',
            'sora': 'OpenAI',
            'dall-e': 'OpenAI',
            'gemini': 'Google',
            'bard': 'Google',
            'palm': 'Google',
            'claude': 'Anthropic',
            'llama': 'Meta',
            'copilot': 'Microsoft',
            'mistral': 'Mistral AI',
            'mixtral': 'Mistral AI'
        }
    
    def cluster_scoped_search(self, query: str, cluster_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform vector search but filter results to only entities in the specified cluster.
        This enables cluster-aware retrieval.
        """
        # Get all entities in cluster
        cluster_entities = self.graph_store.get_entities_in_cluster(cluster_id)
        if not cluster_entities:
            return []
        
        # Do vector search with larger k
        all_results = self.vector_store.search(query, top_k=top_k * 3)
        
        # Filter to only results mentioning cluster entities
        filtered_results = []
        for result in all_results:
            text = result['text'].lower()
            for entity in cluster_entities:
                if entity.lower() in text:
                    result['matched_entity'] = entity
                    filtered_results.append(result)
                    break
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results[:top_k]

    def validate_product_attribution(self, text: str) -> str:
        """
        Post-process the LLM output to catch common attribution errors.
        This is a safety net to fix obvious mistakes like "OpenAI's Grok".
        """
        import re
        
        # Common error patterns to fix
        error_patterns = [
            (r"OpenAI'?s?\s+(Grok|grok)", "xAI's Grok"),
            (r"OpenAI\s+released\s+(Grok|grok)", "xAI released Grok"),
            (r"OpenAI\s+announced\s+(Grok|grok)", "xAI announced Grok"),
            (r"Google'?s?\s+(ChatGPT|chatgpt)", "OpenAI's ChatGPT"),
            (r"Google\s+released\s+(ChatGPT|chatgpt)", "OpenAI released ChatGPT"),
            (r"Anthropic'?s?\s+(Gemini|gemini)", "Google's Gemini"),
            (r"Anthropic\s+released\s+(Gemini|gemini)", "Google released Gemini"),
            (r"xAI'?s?\s+(ChatGPT|GPT-4|GPT-4o|o1)", "OpenAI's \\1"),
            (r"xAI\s+released\s+(ChatGPT|GPT-4|GPT-4o|o1)", "OpenAI released \\1"),
            (r"Meta'?s?\s+(Claude|claude)", "Anthropic's Claude"),
        ]
        
        corrected_text = text
        for pattern, replacement in error_patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text

    def extract_query_intent(self, query: str) -> dict:
        """Parse query to extract entities, industry, intent type, and temporal indicators."""
        q = str(query or "").lower()
        has_why = any(k in q for k in ["why", "reason", "explain"])

        def _match_entities_from_rankings(raw_query: str, limit: int = 3) -> list[str]:
            # Use ranking parquet (fast, local) to match entity mentions in the query.
            try:
                self.community_analytics._ensure_loaded()  # type: ignore[attr-defined]
                df_rank = getattr(self.community_analytics, "df_rank", None)
            except Exception:
                df_rank = None

            ql = str(raw_query or "").lower()
            candidates: list[str] = []

            # Seed with known product->company mappings (high precision).
            for prod, comp in (self.product_company_map or {}).items():
                if prod and prod.lower() in ql and comp:
                    candidates.append(comp)

            if df_rank is not None and not getattr(df_rank, "empty", True):
                names = (
                    df_rank.get("entity_name")
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                    .loc[lambda s: s != ""]
                    .unique()
                    .tolist()
                )
                # Prefer longer names first to avoid partial overlaps.
                names = sorted(names, key=lambda s: len(s), reverse=True)
                for name in names:
                    if name.lower() in ql:
                        candidates.append(name)
                        if len(candidates) >= limit:
                            break

            # De-dupe, preserve order.
            out: list[str] = []
            seen = set()
            for c in candidates:
                if not c:
                    continue
                key = c.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
                if len(out) >= limit:
                    break
            return out

        # Fast path (no LLM): avoid hallucinated entity detection.
        if any(k in q for k in ["competitor", "competition", " vs ", " versus ", "compare"]):
            return {
                "entities": _match_entities_from_rankings(query),
                "industry": None,
                "intent": "comparison",
                "temporal": any(k in q for k in ["new", "recent", "latest", "today"]),
            }
        if any(k in q for k in ["threat", "risks", "risk", "weakness"]):
            return {
                "entities": _match_entities_from_rankings(query),
                "industry": None,
                "intent": "general",
                "temporal": any(k in q for k in ["new", "recent", "latest", "today"]),
            }
        # "Why is X trending?" should be treated as an explanation request, not a list request.
        if has_why:
            return {
                "entities": _match_entities_from_rankings(query),
                "industry": None,
                "intent": "general",
                "temporal": any(k in q for k in ["new", "recent", "latest", "today", "right now", "currently"]),
            }
        # Fast path (no LLM): common list/trending queries should not hallucinate entities.
        if any(k in q for k in ["top", "trending", "list the top", "which startups are trending", "who are the top"]):
            return {"entities": [], "industry": None, "intent": "list", "temporal": any(k in q for k in ["new", "recent", "latest", "today"])}

        if ollama is None:
            return {"entities": [], "industry": None, "intent": "general", "temporal": False}
        
        prompt = f"""
        Analyze this query: "{query}"
        Return JSON with:
        1. "entities": list of specific companies/products mentioned.
        2. "industry": specific industry mentioned (e.g. "Healthcare", "AI").
        3. "intent": "comparison" (if comparing), "list" (if asking for top/new/upcoming), "general".
        4. "temporal": true if asking for "new", "recent", "latest", "upcoming".
        
        JSON ONLY.
        """
        try:
            response = ollama.generate(model=config.LLM_MODEL, prompt=prompt, format='json')
            return json.loads(response['response'])
        except:
            return {"entities": [], "industry": None, "intent": "general", "temporal": False}

    def generate_answer(self, query: str, return_context: bool = False) -> str | dict:
        # Step 1: Figure out what the user wants
        intent_data = self.extract_query_intent(query)
        entities = intent_data.get('entities', [])
        intent = intent_data.get("intent", "general")
        ql = str(query or "").lower()

        # Competitor questions without a target entity are underspecified.
        # Ask for clarification rather than guessing from generic trending lists.
        if ("competitor" in ql or "competitors" in ql) and not entities:
            msg = (
                "Unknown: please specify a company or product (e.g., \"OpenAI competitors\" or \"Anthropic competitors\") "
                "so I can retrieve competitor evidence from the dataset."
            )
            if return_context:
                return {
                    "answer": msg,
                    "vector_context": [],
                    "graph_context": "",
                    "community_context": "",
                    "entity_detected": [],
                    "intent": intent,
                }
            return msg

        def _clean_one_line(s: Any, max_len: int = 260) -> str:
            t = str(s or "")
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) > max_len:
                t = t[: max_len - 1].rstrip() + "…"
            return t

        def _event_mentions_entity(entity_name: str, ev: Dict[str, Any]) -> bool:
            """
            Best-effort check that an evidence event is actually about the entity.
            Prevents mismatched "recent signal" attribution from noisy graph edges.
            """
            en = str(entity_name or "").strip().lower()
            if not en:
                return False
            snippet = str(ev.get("snippet") or ev.get("snippet_text") or "").lower()
            title = str(ev.get("title") or "").lower()
            kpi_value = str(ev.get("kpi_value") or ev.get("value") or "").lower()
            return en in (" ".join([snippet, title, kpi_value]))

        def _event_is_well_attributed(entity_name: str, ev: Dict[str, Any]) -> bool:
            """
            Stronger attribution filter than `_event_mentions_entity`.

            We observed some sources (e.g., TechCrunch) include "related stories" blocks that cause
            unrelated entities to appear in the snippet text, which can misattribute a KPI to
            the wrong company. To reduce this:
            - Prefer events where the article title contains the entity name.
            - Otherwise, require the KPI value itself to mention the entity.
            """
            en = str(entity_name or "").strip().lower()
            if not en:
                return False
            title = str(ev.get("title") or "").strip().lower()
            kpi_value = str(ev.get("kpi_value") or ev.get("value") or "").strip().lower()
            if en and title and en in title:
                return True
            return bool(en and kpi_value and (en in kpi_value))

        def _match_entities_in_text(raw_text: str, limit: int = 5) -> List[str]:
            """
            Extract entity mentions from text using local ranking data + product map (no LLM).
            """
            ql2 = str(raw_text or "").lower()
            candidates: List[str] = []

            for prod, comp in (self.product_company_map or {}).items():
                if prod and prod.lower() in ql2 and comp:
                    candidates.append(comp)

            try:
                self.community_analytics._ensure_loaded()  # type: ignore[attr-defined]
                df_rank = getattr(self.community_analytics, "df_rank", None)
            except Exception:
                df_rank = None

            if df_rank is not None and not getattr(df_rank, "empty", True):
                names = (
                    df_rank.get("entity_name")
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                    .loc[lambda s: s != ""]
                    .unique()
                    .tolist()
                )
                names = sorted(names, key=lambda s: len(s), reverse=True)
                for name in names:
                    if name.lower() in ql2:
                        candidates.append(name)
                        if len(candidates) >= limit:
                            break

            out: List[str] = []
            seen: set[str] = set()
            for c in candidates:
                c2 = str(c or "").strip()
                if not c2:
                    continue
                key = c2.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(c2)
                if len(out) >= limit:
                    break
            return out

        def _normalize_unknown_answer(text: str) -> str:
            """
            If the model indicates "Unknown: ...", return only that line to avoid
            contradictory follow-on text (e.g., extra evidence bullets).
            """
            t = str(text or "").strip()
            if not t:
                return t
            # Prefer explicit Unknown marker.
            m = re.search(r"(?m)^\s*Unknown:\s*[^\n]+", t)
            if m:
                return m.group(0).strip()
            # Common variant: "### Answer Unknown: ..."
            m2 = re.search(r"Unknown:\s*[^\n]+", t)
            if m2:
                return f"Unknown: {m2.group(0).split('Unknown:',1)[1].strip()}".strip()
            # If the model says it's unknown + also says evidence is missing, collapse to Unknown.
            low = t.lower()
            # Common failure mode: "not explicitly stated... however we can infer..."
            if ("not explicitly" in low or "not explicitly stated" in low) and ("infer" in low or "we can infer" in low):
                return "Unknown: not enough evidence in the current dataset."
            # Another common failure mode: "not explicitly stated... may be/could be" (speculation).
            if ("not explicitly" in low or "not explicitly stated" in low) and any(
                p in low for p in ["may be", "might be", "could be", "possible", "potentially"]
            ):
                return "Unknown: not enough evidence in the current dataset."
            if ("unknown" in low) and any(
                p in low
                for p in [
                    "not enough evidence",
                    "not explicitly mentioned",
                    "no direct mention",
                    "no direct mentions",
                    "no specific mention",
                    "no specific mentions",
                    "no mention of",
                    "not mentioned in",
                    "insufficient evidence",
                    "unknown based on",
                ]
            ):
                return "Unknown: not enough evidence in the current dataset."
            return t

        def _has_explicit_threat_evidence(entity_name: str) -> bool:
            """
            For "threat/risk" queries we require explicit evidence.
            If we can't find it, we must not let the model infer.
            """
            en = str(entity_name or "").strip()
            if not en:
                return False

            # 1) KPI evidence: any negative KPI event mentioning the entity.
            try:
                kpis = self.graph_store.get_kpi_breakdown_for_entity(en, limit=20)
            except Exception:
                kpis = {}

            flat: List[Dict[str, Any]] = []
            if isinstance(kpis, dict):
                for _k, vs in kpis.items():
                    if isinstance(vs, list):
                        for ev in vs:
                            if isinstance(ev, dict):
                                flat.append(ev)

            for ev in flat:
                if not _event_mentions_entity(en, ev):
                    continue
                st = ev.get("stance", None)
                try:
                    stf = float(st) if st is not None else 0.0
                except Exception:
                    stf = 0.0
                # Negative stance counts as explicit negative evidence.
                if stf < -0.05:
                    return True
                # Or explicitly threat-like KPI types
                t = str(ev.get("kpi_type") or "").lower()
                if any(x in t for x in ["lawsuit", "security", "breach", "outage", "regulation", "policy", "risk", "threat"]):
                    return True

            # 2) Vector evidence: any retrieved snippet that explicitly frames a threat/risk for the entity.
            for d in (vector_results or [])[:10]:
                txt = str(d.get("text") or "").lower()
                if en.lower() not in txt:
                    continue
                if any(k in txt for k in ["threat", "risk", "lawsuit", "sued", "breach", "hack", "regulation", "ban", "investigation"]):
                    return True

            return False

        def _suggest_followup_queries(
            *,
            question: str,
            entities0: List[str],
            industry0: str | None,
            vector_docs0: List[Dict[str, Any]],
            graph_facts0: List[str],
            community_facts0: List[str],
            max_q: int = 3,
        ) -> List[str]:
            """
            Ask the LLM for additional *search queries* (not answers) to improve recall.
            Returns short queries that we can run through the vector store.
            """
            if ollama is None:
                return []

            # Keep context compact to avoid slowdowns.
            vec_lines: List[str] = []
            for d in (vector_docs0 or [])[:5]:
                title = _clean_one_line(d.get("title") or "", 120)
                src = _clean_one_line(d.get("source") or "", 40)
                pub = _clean_one_line(d.get("published") or "", 30)
                excerpt = _clean_one_line(d.get("text") or "", 220)
                vec_lines.append(f"- {title} ({src} {pub}): {excerpt}")

            g_lines = "\n".join([f"- {_clean_one_line(x, 220)}" for x in (graph_facts0 or [])[:6]])
            c_lines = "\n".join([f"- {_clean_one_line(x, 220)}" for x in (community_facts0 or [])[:4]])

            prompt = f"""
You help a retrieval system. Your job is NOT to answer the question.
Return ONLY JSON with a key "followup_queries": a list of up to {int(max_q)} short search queries (each <= 10 words)
that would help find missing evidence to answer the user's question.

Rules:
- Do not include quotes, citations, or commentary—ONLY JSON.
- If the current evidence already looks sufficient, return an empty list.
- Queries should be specific and evidence-seeking (e.g., "OpenAI funding round", "OpenAI lawsuit", "Anthropic Claude Slack apps").

User question: {question}
Detected entities: {', '.join(entities0) if entities0 else 'None'}
Industry: {industry0 or 'None'}

Current vector evidence:
{chr(10).join(vec_lines) if vec_lines else 'None'}

Current knowledge-graph facts:
{g_lines if g_lines else 'None'}

Current community facts:
{c_lines if c_lines else 'None'}
"""
            try:
                resp = ollama.generate(model=config.LLM_MODEL, prompt=prompt, format="json", options={"temperature": 0.0})
                raw = resp.get("response")
                data = json.loads(raw) if isinstance(raw, str) else {}
                qs = data.get("followup_queries", [])
                if not isinstance(qs, list):
                    return []
                out = []
                seen = set()
                for q in qs:
                    s = _clean_one_line(q, 80)
                    if not s:
                        continue
                    key = s.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(s)
                    if len(out) >= int(max_q):
                        break
                return out
            except Exception:
                return []

        def _wants_negative_explanation(q: str) -> bool:
            q2 = str(q or "").lower()
            return any(k in q2 for k in ["negative", "negatively", "bad pr", "bad press", "down", "declin", "risk"])

        def _format_kpi_event(ev: Dict[str, Any]) -> str:
            kpi_type = str(ev.get("kpi_type") or ev.get("type") or "").strip()
            kpi_value = _clean_one_line(ev.get("kpi_value") or ev.get("value") or "", 220)
            published = str(ev.get("published") or "").strip()
            title = str(ev.get("title") or "").strip()
            source = str(ev.get("source") or "").strip()
            url = str(ev.get("source_url") or "").strip()
            stance = ev.get("stance", None)
            try:
                stance_f = float(stance) if stance is not None else None
            except Exception:
                stance_f = None
            stance_str = f"{stance_f:+.2f}" if stance_f is not None else "N/A"

            bits: list[str] = []
            if kpi_type and kpi_value:
                bits.append(f"{kpi_type}: {kpi_value}")
            elif kpi_value:
                bits.append(kpi_value)
            if published:
                bits.append(published)
            if source:
                bits.append(source)
            if title:
                bits.append(f"\"{_clean_one_line(title, 180)}\"")
            if url:
                bits.append(url)
            bits.append(f"stance={stance_str}")
            return " | ".join(bits)

        def _format_competitor_answer(target: str, comp_events: List[Dict[str, Any]]) -> str:
            """
            Deterministic competitor answer: only uses explicit competition evidence.
            """
            target2 = str(target or "").strip()
            if not target2:
                return "Unknown: not enough evidence in the current dataset."

            competitors: List[str] = []
            for ev in (comp_events or [])[:25]:
                blob = " ".join(
                    [
                        str(ev.get("kpi_value") or ""),
                        str(ev.get("snippet_text") or ""),
                        str(ev.get("title") or ""),
                    ]
                )
                for name in _match_entities_in_text(blob, limit=12):
                    if name.lower() == target2.lower():
                        continue
                    competitors.append(name)
            competitors = _dedupe_keep_order([c for c in competitors if c], limit=8)
            if not competitors:
                return "Unknown: not enough explicit competitor evidence in the current dataset."

            lines = [f"Competitors explicitly referenced alongside {target2} in the dataset:"]
            for c in competitors:
                lines.append(f"- {c}")
            lines.append("")
            lines.append("Evidence:")
            for ev in (comp_events or [])[:3]:
                lines.append(f"- {_format_kpi_event(ev)}")
            return "\n".join(lines).rstrip()

        def _topk_from_query(q: str, default: int = 10) -> int:
            m = re.search(r"(?i)\btop\s+(\d{1,2})\b", str(q or ""))
            if m:
                try:
                    v = int(m.group(1))
                    if 1 <= v <= 50:
                        return v
                except Exception:
                    pass
            return int(default)

        def _llm_reason_with_citations(
            *,
            question: str,
            entity_name: str,
            ranking_row: Dict[str, Any] | None,
            events: List[Dict[str, Any]],
        ) -> str:
            if ollama is None:
                return ""

            # Build a compact evidence list with stable IDs (dedupe to avoid repeated citations).
            ev_lines: List[str] = []
            deduped: List[Dict[str, Any]] = []
            seen_keys: set[str] = set()
            for ev in (events or []):
                url = str(ev.get("source_url") or "").strip()
                title = str(ev.get("title") or "").strip()
                published = str(ev.get("published") or "").strip()
                key = url or f"{title}|{published}"
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                deduped.append(ev)
                if len(deduped) >= 8:
                    break

            def _append_sources_section(md: str) -> str:
                """
                Append a deterministic Sources section.

                Some LLMs occasionally emit "vertical text" when copying long URLs/titles; by
                appending sources ourselves, we keep output readable and stable.
                """
                md2 = str(md or "").rstrip()
                if not md2:
                    return md2
                if re.search(r"(?im)^###\\s+sources\\b", md2):
                    return md2
                out_lines = [md2, "", "### Sources"]
                for i, ev in enumerate(deduped, start=1):
                    title2 = _clean_one_line(ev.get("title") or "", 180)
                    url2 = str(ev.get("source_url") or "").strip()
                    if title2 and url2:
                        out_lines.append(f"[{i}] {title2} — {url2}")
                    elif url2:
                        out_lines.append(f"[{i}] {url2}")
                    elif title2:
                        out_lines.append(f"[{i}] {title2}")
                return "\n".join(out_lines).rstrip()

            for i, ev in enumerate(deduped, start=1):
                kpi_type = str(ev.get("kpi_type") or "").strip()
                kpi_value = _clean_one_line(ev.get("kpi_value") or "", 220)
                published = str(ev.get("published") or "").strip()
                title = _clean_one_line(ev.get("title") or "", 180)
                source = str(ev.get("source") or "").strip()
                url = str(ev.get("source_url") or "").strip()
                stance = ev.get("stance", None)
                try:
                    stance_str = f"{float(stance):+.2f}"
                except Exception:
                    stance_str = "N/A"

                # Avoid pipe characters to keep markdown stable.
                parts = []
                if kpi_type and kpi_value:
                    parts.append(f"{kpi_type}: {kpi_value}")
                elif kpi_value:
                    parts.append(kpi_value)
                if published:
                    parts.append(f"date={published}")
                if source:
                    parts.append(f"source={source}")
                if title:
                    parts.append(f"title={title}")
                if url:
                    parts.append(f"url={url}")
                parts.append(f"stance={stance_str}")
                ev_lines.append(f"[{i}] " + " ; ".join(parts))

            rank_bits: List[str] = []
            if ranking_row:
                try:
                    rank_bits.append(f"composite_score={float(ranking_row.get('composite_score', 0.0) or 0.0):.2f}")
                except Exception:
                    pass
                try:
                    rank_bits.append(f"cluster_id={int(ranking_row.get('cluster_id', 0) or 0)}")
                except Exception:
                    pass
                try:
                    rank_bits.append(f"kpi_stance={float(ranking_row.get('kpi_stance', 0.0) or 0.0):+.2f}")
                except Exception:
                    pass
                try:
                    rank_bits.append(f"kpi_momentum={float(ranking_row.get('kpi_momentum', 0.0) or 0.0):+.2f}")
                except Exception:
                    pass
                try:
                    rank_bits.append(f"kpi_risk={float(ranking_row.get('kpi_risk', 0.0) or 0.0):+.2f}")
                except Exception:
                    pass
                try:
                    rank_bits.append(f"recency={float(ranking_row.get('recency', 0.0) or 0.0):.2f}")
                except Exception:
                    pass

            prompt = f"""
You are TrendScout AI. Answer the user's question about {entity_name} using ONLY the evidence provided below.
If the evidence is insufficient to answer, respond with: "Unknown: not enough evidence in the current dataset."

You MUST:
- Provide a clean, concise explanation (no raw dumps of evidence).
- Use citations like [1], [2] after sentences that rely on evidence.
- Do NOT invent events, numbers, or claims not supported by the evidence.
- Use careful verbs: do not say an entity "built/developed/owns" something unless the evidence explicitly states that.

User question: {question}

Ranking context (may be empty): {"; ".join(rank_bits) if rank_bits else "None"}

Evidence events:
{chr(10).join(ev_lines) if ev_lines else "None"}

Write the answer in markdown with sections:
### Summary
### Key drivers (with citations)
Do NOT include a Sources section; I will append it.
            """
            try:
                resp = ollama.generate(model=config.LLM_MODEL, prompt=prompt, options={"temperature": 0.2})
                out = str(resp.get("response") or "").strip()
                out = _normalize_unknown_answer(out)
                if out.lower().startswith("unknown:"):
                    return out
                return _append_sources_section(out)
            except Exception:
                return ""

        def _compose_general_answer_with_sources(
            *,
            question: str,
            entities: List[str],
            industry: str | None,
            graph_facts: List[str],
            kpi_events_by_entity: Dict[str, List[Dict[str, Any]]],
            vector_docs: List[Dict[str, Any]],
        ) -> str:
            """
            Generic answer path: combine KG + RAG evidence, require citations, and avoid hallucination.
            """
            if ollama is None:
                return "Unknown: LLM backend is not available."

            sources: List[Dict[str, Any]] = []

            # KPI-backed sources (usually have urls/titles)
            for ent in (entities or [])[:3]:
                for ev in (kpi_events_by_entity.get(ent) or [])[:4]:
                    sources.append(
                        {
                            "title": ev.get("title") or "",
                            "source": ev.get("source") or "",
                            "published": ev.get("published") or "",
                            "url": ev.get("source_url") or "",
                            "excerpt": ev.get("snippet") or ev.get("kpi_value") or "",
                            "entity": ent,
                        }
                    )

            # Vector sources (may have urls if the Chroma metadata includes it)
            for d in (vector_docs or [])[:5]:
                sources.append(
                    {
                        "title": d.get("title") or "",
                        "source": d.get("source") or "",
                        "published": d.get("published") or "",
                        "url": d.get("link") or "",
                        "excerpt": d.get("text") or "",
                        "entity": d.get("matched_entity") or "",
                    }
                )

            usable_sources = [s for s in sources if _clean_one_line(s.get("excerpt"), 30)]

            # Dedupe sources by URL/title so the model doesn't cite the same URL multiple times.
            deduped_sources: List[Dict[str, Any]] = []
            seen: set[str] = set()
            for s in usable_sources:
                url = _clean_one_line(s.get("url") or "", 220)
                title = _clean_one_line(s.get("title") or "", 140)
                published = _clean_one_line(s.get("published") or "", 40)
                key = url or f"{title}|{published}"
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped_sources.append(s)
                if len(deduped_sources) >= 12:
                    break

            # Hard guardrail: if we have no sources and no KG facts, do not call the model.
            if not deduped_sources and not graph_facts:
                return "Unknown: not enough evidence in the current dataset."

            src_lines: List[str] = []
            for i, s in enumerate(deduped_sources, start=1):
                title = _clean_one_line(s.get("title") or "Untitled", 140)
                source = _clean_one_line(s.get("source") or "Unknown", 60)
                published = _clean_one_line(s.get("published") or "", 40)
                url = _clean_one_line(s.get("url") or "", 220)
                excerpt = _clean_one_line(s.get("excerpt") or "", 260)
                meta_bits = [source]
                if published:
                    meta_bits.append(published)
                if url:
                    meta_bits.append(url)
                src_lines.append(f"[{i}] {title} — {' | '.join(meta_bits)} — {excerpt}")

            graph_lines = "\n".join([f"- {_clean_one_line(x, 260)}" for x in (graph_facts or [])[:10]])
            ent_str = ", ".join(entities) if entities else "None"
            ind_str = industry or "None"

            prompt = f"""
You are TrendScout AI. Answer the user's question using ONLY the evidence below.
If the evidence is insufficient, respond exactly with: "Unknown: not enough evidence in the current dataset."

Rules:
- Do not invent facts, dates, companies, or numbers.
- Cite sources like [1], [2] for any factual statement.
- Prefer KPI-linked evidence over general claims.
- Use careful verbs: do not say an entity "built/developed/owns" something unless the evidence explicitly states that.
- First, identify what the user is asking (intent + entity scope). Then answer directly.

User question: {question}
Detected entities: {ent_str}
Industry (if any): {ind_str}

Knowledge-graph facts (may be empty):
{graph_lines if graph_lines else "None"}

Evidence sources:
{chr(10).join(src_lines) if src_lines else "None"}

Return markdown with:
### Answer
### Evidence (bullets, each with citations)
Do NOT include a Sources section; I will append it.
"""
            def _append_sources_section_for_cited(md: str) -> str:
                md2 = str(md or "").rstrip()
                if not md2:
                    return md2
                if re.search(r"(?im)^###\\s+sources\\b", md2):
                    return md2

                used = []
                for m in re.finditer(r"\\[(\\d{1,2})\\]", md2):
                    try:
                        used.append(int(m.group(1)))
                    except Exception:
                        continue
                used = sorted(set([u for u in used if 1 <= u <= len(deduped_sources)]))
                if not used:
                    # No citations? Keep output as-is (avoid implying sourcing).
                    return md2

                out_lines = [md2, "", "### Sources"]
                for idx in used:
                    s = deduped_sources[idx - 1]
                    title2 = _clean_one_line(s.get("title") or "Untitled", 140)
                    url2 = _clean_one_line(s.get("url") or "", 220)
                    if url2:
                        out_lines.append(f"[{idx}] {title2} — {url2}")
                    else:
                        out_lines.append(f"[{idx}] {title2}")
                return "\\n".join(out_lines).rstrip()
            try:
                resp = ollama.generate(model=config.LLM_MODEL, prompt=prompt, options={"temperature": 0.2})
                out = str(resp.get("response") or "").strip()
                out = _normalize_unknown_answer(out)
                if out.lower().startswith("unknown:"):
                    return out
                return _append_sources_section_for_cited(out) or "Unknown: not enough evidence in the current dataset."
            except Exception:
                return "Unknown: not enough evidence in the current dataset."

        # Deterministic list answers for accuracy (avoid hallucinated entity names).
        if intent == "list" and not entities and any(k in ql for k in ["trending", "ranking", "rank", "top"]):
            top_k = _topk_from_query(query, default=10)
            trending = self.community_analytics.get_top_trending_entities(top_k=max(top_k, 10))
            if trending:
                want_startups = "startup" in ql or "startups" in ql or "companies" in ql
                # Filter out obvious non-companies when user asks for startups/companies.
                bad_name_substrings = [
                    "techcrunch",
                    "wired",
                    "the verge",
                    "venturebeat",
                    "crunchbase",
                    "mergermarket",
                    "white house",
                ]
                type_blacklist = {"Media", "Government", "Person", "Country", "Product"}

                names = [str(r.get("entity_name") or "") for r in trending if r.get("entity_name")]
                types = self.graph_store.get_entity_types(names) if want_startups else {}

                filtered = []
                for row in trending:
                    name = str(row.get("entity_name") or "").strip()
                    if not name:
                        continue
                    if want_startups and any(b in name.lower() for b in bad_name_substrings):
                        continue
                    if want_startups:
                        t = (types.get(name) or "").strip()
                        if t in type_blacklist:
                            continue
                    filtered.append(row)
                    if len(filtered) >= top_k:
                        break

                lines = ["Top trending startups right now:" if want_startups else "Top trending entities right now:"]
                for row in filtered:
                    score = row.get("score")
                    try:
                        score_str = f"{float(score):.2f}"
                    except Exception:
                        score_str = "N/A"
                    extra = ""
                    if row.get("cluster_id") is not None:
                        extra = f", cluster={row.get('cluster_id')}"

                    name = row.get("entity_name")
                    brief_reason = "Unknown: no KPI-linked signal found."
                    try:
                        kpis = self.graph_store.get_kpi_breakdown_for_entity(str(name), limit=2)
                    except Exception:
                        kpis = {}
                    flat_events: list[dict[str, Any]] = []
                    if isinstance(kpis, dict):
                        for _k, vs in kpis.items():
                            if isinstance(vs, list):
                                for ev in vs:
                                    if isinstance(ev, dict):
                                        flat_events.append(ev)
                    flat_events = [e for e in flat_events if (e.get("kpi_value") or e.get("value"))]
                    # Avoid mismatched attribution: keep only events that mention the entity.
                    flat_events = [e for e in flat_events if _event_mentions_entity(str(name or ""), e)]
                    flat_events = [e for e in flat_events if _event_is_well_attributed(str(name or ""), e)]
                    if flat_events:
                        t = str(flat_events[0].get("kpi_type") or "").strip()
                        v = _clean_one_line(flat_events[0].get("kpi_value") or "", 140)
                        url = str(flat_events[0].get("source_url") or "").strip()
                        date = str(flat_events[0].get("published") or "").strip()
                        if t and v:
                            brief_reason = f"recent signal: {t}: {v}"
                        elif v:
                            brief_reason = f"recent signal: {v}"
                        if date:
                            brief_reason += f" ({date})"
                        if url:
                            brief_reason += f" [source: {url}]"
                    else:
                        parts = []
                        try:
                            if row.get("recency") is not None:
                                parts.append(f"recency={float(row.get('recency')):.2f}")
                        except Exception:
                            pass
                        try:
                            if row.get("centrality") is not None:
                                parts.append(f"centrality={float(row.get('centrality')):.2f}")
                        except Exception:
                            pass
                        if parts:
                            brief_reason = "drivers: " + ", ".join(parts)

                    lines.append(f"- #{row.get('rank')}: {name} (score={score_str}{extra}) — {brief_reason}")
                if return_context:
                    return {
                        "answer": "\n".join(lines),
                        "vector_context": [],
                        "graph_context": "",
                        "community_context": "",
                        "entity_detected": [],
                        "intent": intent,
                    }
                return "\n".join(lines)

        # Competitor questions WITH a target entity: require explicit competition evidence.
        # Do not guess competitors from generic "AI space" lists or mere co-mentions.
        if ("competitor" in ql or "competitors" in ql or "competition" in ql) and entities:
            target = str(entities[0])
            comp_events: List[Dict[str, Any]] = []
            try:
                breakdown = self.graph_store.get_kpi_breakdown_for_entity(target, limit=30)
            except Exception:
                breakdown = {}
            if isinstance(breakdown, dict):
                for k, vs in breakdown.items():
                    if not isinstance(vs, list):
                        continue
                    for ev in vs:
                        if not isinstance(ev, dict):
                            continue
                        kpi_type = str(ev.get("kpi_type") or k or "").lower()
                        blob = " ".join(
                            [
                                str(ev.get("kpi_value") or ""),
                                str(ev.get("snippet_text") or ""),
                                str(ev.get("title") or ""),
                            ]
                        ).lower()
                        has_comp_kw = any(w in (kpi_type + " " + blob) for w in ["compet", "rival", " vs ", " versus "])
                        if not has_comp_kw:
                            continue
                        if not _event_mentions_entity(target, ev):
                            continue
                        # Stronger attribution: avoid "related stories" contamination.
                        if not _event_is_well_attributed(target, ev):
                            continue
                        comp_events.append(ev)

            if not comp_events:
                msg = "Unknown: not enough explicit competitor evidence in the current dataset."
                if return_context:
                    return {
                        "answer": msg,
                        "vector_context": [],
                        "graph_context": "",
                        "community_context": "",
                        "entity_detected": [target],
                        "intent": "comparison",
                    }
                return msg

            answer_text = _format_competitor_answer(target, comp_events)
            if return_context:
                return {
                    "answer": answer_text,
                    "vector_context": [],
                    "graph_context": "",
                    "community_context": "",
                    "entity_detected": [target],
                    "intent": "comparison",
                }
            return answer_text

        # "Why" questions: use evidence + LLM synthesis with citations (no hallucinations).
        if "why" in ql and entities:
            primary_entity = str(entities[0])
            wants_neg = _wants_negative_explanation(query)

            # Pull best ranking row (parquet) if available.
            best_rank = None
            try:
                self.community_analytics._ensure_loaded()  # type: ignore[attr-defined]
                df_rank = getattr(self.community_analytics, "df_rank", None)
                if df_rank is not None and not df_rank.empty and "entity_name" in df_rank.columns:
                    sub = df_rank[df_rank["entity_name"].astype(str).str.lower() == primary_entity.lower()]
                    if not sub.empty:
                        best_rank = sub.sort_values("composite_score", ascending=False).iloc[0].to_dict()
            except Exception:
                best_rank = None

            kpis = self.graph_store.get_kpi_breakdown_for_entity(primary_entity, limit=12)
            flat_events: list[dict[str, Any]] = []
            if isinstance(kpis, dict):
                for _k, vs in kpis.items():
                    if isinstance(vs, list):
                        for ev in vs:
                            if isinstance(ev, dict):
                                flat_events.append(ev)

            flat_events = [e for e in flat_events if (e.get("kpi_value") or e.get("value") or e.get("snippet_text"))]
            # Avoid "related stories" misattribution: only keep events that are clearly about the entity.
            flat_events = [e for e in flat_events if _event_mentions_entity(primary_entity, e)]
            flat_events = [e for e in flat_events if _event_is_well_attributed(primary_entity, e)]
            neg_events: list[dict[str, Any]] = []
            pos_events: list[dict[str, Any]] = []
            for ev in flat_events:
                st = ev.get("stance", None)
                try:
                    stf = float(st) if st is not None else 0.0
                except Exception:
                    stf = 0.0
                if stf < -0.05:
                    neg_events.append(ev)
                elif stf > 0.05:
                    pos_events.append(ev)

            if not flat_events and best_rank is None:
                msg = (
                    f"Unknown: I don’t have enough extracted evidence to explain why {primary_entity} is trending in the current dataset "
                    "(no ranking row and no KPI events linked to the entity)."
                )
                if return_context:
                    return {
                        "answer": msg,
                        "vector_context": [],
                        "graph_context": "",
                        "community_context": "",
                        "entity_detected": [primary_entity],
                        "intent": "general",
                    }
                return msg

            chosen = (neg_events if wants_neg else (neg_events + pos_events))[:8]
            if wants_neg and not chosen:
                msg = f"Unknown: no negative KPI-linked evidence was found for {primary_entity} in the current dataset."
                if return_context:
                    return {
                        "answer": msg,
                        "vector_context": [],
                        "graph_context": "",
                        "community_context": "",
                        "entity_detected": [primary_entity],
                        "intent": "general",
                    }
                return msg

            llm_answer = _llm_reason_with_citations(
                question=query,
                entity_name=primary_entity,
                ranking_row=best_rank if isinstance(best_rank, dict) else None,
                events=chosen,
            )
            if not llm_answer:
                # Fallback: simple, readable deterministic answer (still no hallucination).
                lines: list[str] = []
                lines.append(f"Why {primary_entity} is trending right now:" if not wants_neg else f"Why {primary_entity} is getting negative attention:")
                if best_rank is not None:
                    try:
                        sc = float(best_rank.get("composite_score", 0.0) or 0.0)
                    except Exception:
                        sc = 0.0
                    cid = best_rank.get("cluster_id", None)
                    lines.append(f"- Ranking: score={sc:.2f}" + (f", cluster={int(cid)}" if cid is not None else ""))
                if chosen:
                    lines.append("- Evidence (most recent signals):")
                    for ev in chosen[:3]:
                        lines.append(f"  - {_format_kpi_event(ev)}")
                else:
                    lines.append("- Unknown: no KPI-linked evidence was found for this entity in Neo4j.")
                llm_answer = "\n".join(lines)

            answer_text = _normalize_unknown_answer(llm_answer)
            if return_context:
                return {
                    "answer": answer_text,
                    "vector_context": [],
                    "graph_context": "",
                    "community_context": "",
                    "entity_detected": [primary_entity],
                    "intent": "general",
                }
            return answer_text

        if ollama is None:
            raise RuntimeError("Ollama is required to generate answers. Install dependencies and ensure Ollama is running.")
        
        # Sanitize entities: ensure they are strings
        clean_entities = []
        for e in entities:
            if isinstance(e, str):
                clean_entities.append(e)
            elif isinstance(e, dict):
                # Handle case where LLM returns objects like {"name": "OpenAI"}
                # We'll try to find a likely key
                for key in ['name', 'entity', 'company', 'product']:
                    if key in e:
                        clean_entities.append(e[key])
                        break
        entities = clean_entities

        industry = intent_data.get('industry')
        intent = intent_data.get('intent')
        is_temporal = intent_data.get('temporal', False)

        # Step 2: Search vector database with recency bias if temporal query
        vector_results = self.vector_store.search(query, top_k=5, recency_bias=is_temporal)
        vector_context = "\n".join([f"- {r['text']} (Source: {r['source']}, Date: {r['published']})" for r in vector_results])

        # Step 3: Retrieve knowledge graph and community analytics data
        graph_context = []
        community_context = []

        # Get detailed stats for specific entities mentioned
        if entities:
            graph_context.extend(self.graph_store.get_entity_context(entities))

            # Get community context for primary entity
            primary_entity = entities[0]
            comm_ids = self.community_analytics.get_entity_communities(primary_entity)
            if comm_ids:
                cid = comm_ids[0]
                top_ents = self.community_analytics.get_top_entities_in_community(cid)
                swot = self.community_analytics.get_swot_for_community(cid)
                temporal = self.community_analytics.get_temporal_for_entity(primary_entity, community_id=cid)

                if top_ents:
                    # Enhanced: Show ranking scores if available
                    ent_strs = []
                    for e in top_ents:
                        if 'rank' in e:
                            ent_str = f"{e['entity_name']} (Rank #{e['rank']}, Score={e['score']:.2f}"
                            if 'investor_quality' in e and e['investor_quality'] > 0:
                                ent_str += f", Investor Quality={e['investor_quality']:.2f}"
                            ent_str += ")"
                        else:
                            ent_str = f"{e['entity_name']} (mentions={e['mentions']}, avg_stance={e.get('avg_stance', 'N/A')})"
                        ent_strs.append(ent_str)
                    
                    community_context.append(
                        f"Community {cid} key entities: {'; '.join(ent_strs)}"
                    )
                    
                if swot:
                    swot_str = ", ".join([f"{k}: {v}" for k, v in swot.items()])
                    community_context.append(
                        f"Community {cid} SWOT distribution (counts): {swot_str}"
                    )
                if temporal:
                    # Show last 6 months of temporal data
                    temp_str = ", ".join([
                        f"{t['year_month']}: {t['entity_mentions']} mentions"
                        for t in temporal[-6:]
                    ])
                    community_context.append(
                        f"Community {cid} recent temporal trend (last up to 6 periods): {temp_str}"
                    )
                
                # NEW: Add Forecast if temporal intent
                if is_temporal:
                    forecast = self.community_analytics.get_community_forecast(cid)
                    if forecast:
                        community_context.append(
                            f"Community {cid} FORECAST (Next Month): Predicted Mentions = {forecast['predicted_next_month']:.1f} (Trend Slope: {forecast['slope']:.2f})"
                        )
            
            # NEW: Get detailed KPI breakdown for primary entity
            kpi_breakdown = self.graph_store.get_kpi_breakdown_for_entity(primary_entity, limit=5)
            if any(kpi_breakdown.values()):
                all_events: list[dict[str, Any]] = []
                for kpi_type, kpis in kpi_breakdown.items():
                    for k in kpis:
                        all_events.append({"type": kpi_type, **(k or {})})

                pos = sum(1 for e in all_events if float(e.get("stance", 0.0) or 0.0) > 0.25)
                neg = sum(1 for e in all_events if float(e.get("stance", 0.0) or 0.0) < -0.25)
                neu = max(0, len(all_events) - pos - neg)

                kpi_summary = []
                for kpi_type, kpis in kpi_breakdown.items():
                    if kpis:
                        kpi_summary.append(f"{kpi_type.upper()}: {len(kpis)}")
                if kpi_summary:
                    community_context.append(
                        f"Recent event signals for {primary_entity}: {', '.join(kpi_summary)} (pos={pos}, neg={neg}, neutral={neu})"
                    )

                def _fmt_driver(e: dict) -> str:
                    stance = float(e.get("stance", 0.0) or 0.0)
                    tag = "POS" if stance > 0.25 else ("NEG" if stance < -0.25 else "NEU")
                    src = e.get("source") or "Source"
                    pub = e.get("published") or "date"
                    val = str(e.get("kpi_value") or "").replace("\n", " ").strip()
                    url = str(e.get("source_url") or "").strip()
                    if len(val) > 140:
                        val = val[:137] + "..."
                    suffix = f" ({src}, {pub})"
                    if url:
                        suffix += f" [{url}]"
                    return f"- [{tag}] {str(e.get('type', 'kpi')).upper()}: {val}{suffix}"

                drivers = all_events[:]
                try:
                    drivers.sort(key=lambda x: str(x.get("published") or ""), reverse=True)
                except Exception:
                    pass
                drivers = drivers[:6]
                if drivers:
                    community_context.append("Event drivers (latest):\n" + "\n".join(_fmt_driver(d) for d in drivers))
            
            # NEW: Get investor quality data
            investor_info = self.graph_store.get_investor_quality_for_entity(primary_entity)
            if investor_info['investors']:
                top_investors = [inv['name'] for inv in investor_info['investors'][:3]]
                community_context.append(
                    f"Top Investors for {primary_entity}: {', '.join(top_investors)} (Avg Prestige: {investor_info['avg_prestige']:.2f})"
                )
            
            # NEW: Get quantifiable SWOT metrics
            quant_swot = self.community_analytics.get_quantifiable_swot_for_entity(primary_entity)
            if quant_swot:
                swot_metrics_str = []
                if 'kpi_stance_score' in quant_swot:
                    swot_metrics_str.append(f"Overall Sentiment Score: {quant_swot['kpi_stance_score']:.2f}")
                if 'kpi_polarity_delta' in quant_swot:
                    swot_metrics_str.append(f"Sentiment Trend (30d Delta): {quant_swot['kpi_polarity_delta']:.2f}")
                community_context.append(f"Quantifiable SWOT for {primary_entity}: {', '.join(swot_metrics_str)}")

        # Get top players for mentioned industry
        if industry:
            graph_context.extend(self.graph_store.get_industry_entities(industry))

        # NEW: Check for "Trending" or "Top" intent to inject Ranking Data
        if intent in ['list', 'general'] or any(w in query.lower() for w in ['trending', 'top', 'hot', 'ranking']):
            trending = self.community_analytics.get_top_trending_entities(top_k=10)
            if trending:
                # Format as a CSV for the LLM to easily parse into a table
                csv_header = "Rank,Company Name,Score,Growth Slope"
                csv_rows = [csv_header]
                for t in trending:
                    try:
                        score = float(t.get("score", 0.0) or 0.0)
                    except Exception:
                        score = 0.0
                    gs = t.get("growth_slope")
                    if gs is None:
                        gs = t.get("recency")
                    try:
                        gs_f = float(gs or 0.0)
                    except Exception:
                        gs_f = 0.0
                    csv_rows.append(f"{t.get('rank')},{t.get('entity_name')},{score:.1f},{gs_f:.1f}")
                
                # Wrap in a special tag so the UI can render it as a dataframe
                community_context.append(f"COMMUNITY-LEVEL ANALYSIS (TOP 10 TRENDING):\n<csv_table>\n" + "\n".join(csv_rows) + "\n</csv_table>")

        graph_context_str = "\n".join(graph_context)
        community_context_str = "\n".join(community_context)

        # Hard "no hallucination" guardrail: if we have no evidence, say unknown instead of calling the LLM.
        if not vector_results and not graph_context and not community_context and not industry and not entities:
            msg = "Unknown: I don’t have enough context in the current dataset to answer that question."
            if return_context:
                return {
                    "answer": msg,
                    "vector_context": [],
                    "graph_context": "",
                    "community_context": "",
                    "entity_detected": [],
                    "intent": intent,
                }
            return msg

        # Unified answer path: use evidence + citations, and avoid hallucination for any query type.
        # (The longer prompt below remains for reference but is bypassed.)
        # Iterative retrieval: ask the LLM for follow-up search queries to improve recall,
        # then run an extra vector search pass and enrich entity detection.
        followups = _suggest_followup_queries(
            question=query,
            entities0=list(entities or []),
            industry0=industry,
            vector_docs0=list(vector_results or []),
            graph_facts0=list(graph_context or []),
            community_facts0=list(community_context or []),
            max_q=3,
        )
        if followups:
            extra_docs: List[Dict[str, Any]] = []
            for fq in followups:
                try:
                    extra_docs.extend(self.vector_store.search(fq, top_k=3, recency_bias=is_temporal))
                except Exception:
                    continue

            # De-dupe docs by snippet_id if present, else by text.
            merged: List[Dict[str, Any]] = []
            seen_doc: set[str] = set()
            for d in list(vector_results or []) + extra_docs:
                sid = str(d.get("snippet_id") or "").strip()
                key = sid or _clean_one_line(d.get("text") or "", 220).lower()
                if not key or key in seen_doc:
                    continue
                seen_doc.add(key)
                merged.append(d)
            vector_results = merged[:10]

            # Expand entity detection from followup queries (still no LLM entity extraction).
            extra_entities = _match_entities_in_text(" ".join(followups), limit=3)
            for e in extra_entities:
                if str(e).strip() and str(e).strip() not in (entities or []):
                    entities.append(str(e).strip())
            if extra_entities:
                try:
                    graph_context.extend(self.graph_store.get_entity_context(extra_entities))
                except Exception:
                    pass
                graph_context_str = "\n".join(graph_context)

        # Threat/risk questions: do NOT allow inference when we lack explicit evidence,
        # even after iterative retrieval.
        if any(k in ql for k in ["threat", "threats", "risk", "risks"]) and entities:
            primary = str(entities[0])
            if not _has_explicit_threat_evidence(primary):
                msg = "Unknown: not enough evidence in the current dataset."
                if return_context:
                    return {
                        "answer": msg,
                        "vector_context": vector_results,
                        "graph_context": graph_context_str,
                        "community_context": community_context_str,
                        "entity_detected": [primary],
                        "intent": intent,
                    }
                return msg

        kpi_events_by_entity: Dict[str, List[Dict[str, Any]]] = {}
        for ent in (entities or [])[:3]:
            flat: List[Dict[str, Any]] = []
            try:
                breakdown = self.graph_store.get_kpi_breakdown_for_entity(ent, limit=10)
            except Exception:
                breakdown = {}
            if isinstance(breakdown, dict):
                for k, vs in breakdown.items():
                    if isinstance(vs, list):
                        for ev in vs:
                            if isinstance(ev, dict):
                                flat.append(
                                    {
                                        "kpi_type": ev.get("kpi_type") or k,
                                        "kpi_value": ev.get("kpi_value"),
                                        "published": ev.get("published"),
                                        "source_url": ev.get("source_url"),
                                        "title": ev.get("title"),
                                        "source": ev.get("source"),
                                        "stance": ev.get("stance"),
                                        "snippet": ev.get("snippet"),
                                    }
                                )
            if not flat:
                try:
                    for a in self.graph_store.get_recent_articles_for_entity(ent, limit=3):
                        if not isinstance(a, dict):
                            continue
                        flat.append(
                            {
                                "kpi_type": "Mention",
                                "kpi_value": "Mentioned in recent coverage",
                                "published": a.get("published"),
                                "source_url": a.get("source_url"),
                                "title": a.get("title"),
                                "source": a.get("source"),
                                "stance": a.get("stance"),
                                "snippet": a.get("title"),
                            }
                        )
                except Exception:
                    pass
            kpi_events_by_entity[ent] = flat

        evidence_facts: List[str] = []
        evidence_facts.extend(graph_context or [])
        evidence_facts.extend(community_context or [])

        answer = _compose_general_answer_with_sources(
            question=query,
            entities=entities,
            industry=industry,
            graph_facts=evidence_facts,
            kpi_events_by_entity=kpi_events_by_entity,
            vector_docs=vector_results,
        )
        answer = _normalize_unknown_answer(self.validate_product_attribution(answer))

        if return_context:
            return {
                "answer": answer,
                "vector_context": vector_results,
                "graph_context": graph_context_str,
                "community_context": community_context_str,
                "entity_detected": entities,
                "intent": intent,
            }
        return answer

        # Step 4: Put it all together and ask the LLM for the final answer
        system_instruction = """You are TrendScout AI, an advanced market intelligence analyst. 
        Your goal is to provide deep, synthesized insights by connecting dots across multiple data sources.
        
        DATA SOURCES AVAILABLE:
        1. **Recent News (Vector)**: Specific articles and snippets with semantic relevance. Use this for specific details, quotes, and recent events.
        2. **Knowledge Graph (Structured)**: Entity relationships, industries, direct connections, and SWOT themes. Use this to understand who competes with whom, what industry they belong to, and their strategic context.
        3. **Community Analysis (Aggregated)**: High-level trends, rankings, SWOT counts, and growth forecasts. Use this for "big picture" trends and statistical backing.

        CORE INSTRUCTIONS:
        1. **Synthesize, Don't Just List**: Weave facts into a coherent narrative. Treat rankings as *conversation intensity*, not automatically "good" performance.
           - Example: Instead of "The ranking score is X," say "The company is being discussed a lot right now; the drivers look mostly positive/negative based on the cited events."
           - IMPORTANT: "Trending" can be driven by negative attention (risk). Separate Attention vs Momentum (positive events) vs Risk (negative events).
        2. **Evidence-Based and Attributed**: When making a claim (e.g., "OpenAI is trending"), cite the specific metric (mentions/score/slope) and also the *direction* (positive vs negative) from the extracted event stances and/or Quantifiable SWOT.
        3. **Structure Your Answer**: Start with a high-level summary. Then, use markdown headings (e.g., `### Key Developments`, `### Competitive Landscape`, `### SWOT Analysis`) to organize your answer.
        4. **Use Tables for Comparisons**: When comparing multiple entities, use a markdown table to present the data clearly. If you have ranking data, present it in a table.
        5. **Quantify SWOT**: When performing a SWOT analysis, use the provided "Quantifiable SWOT" metrics (Sentiment Score, Sentiment Trend) to support your claims about whether the company's position is strong, weak, improving, or declining.
        5. **Gap Analysis**: If the user asks about a topic not fully covered, explicitly state what the data shows and where the gaps are, rather than making up an answer.
        """

        if intent == 'comparison':
            system_instruction += "\nTASK: Provide a comparative analysis (SWOT) of the entities mentioned."
        elif intent == 'list':
            system_instruction += "\nTASK: Provide a structured list or table of the relevant companies/trends. Highlight recent developments."
        
        fact_check_instruction = """
        CRITICAL WRITING STYLE INSTRUCTIONS:
        1. **Natural, Conversational Tone**: Write like a market analyst speaking to a colleague, not a robot reading data.
           - [NO] AVOID: "Based on the context", "According to the data", "The analysis shows", "Community-Level Analysis", "Knowledge Graph"
           - [OK] USE: "Recent developments show", "The market is seeing", "Companies are focusing on", "Industry data reveals"
        
        2. **Seamless Integration**: Blend data into your narrative naturally.
           - [NO] BAD: "OpenAI has a Growth Slope of 410.0 according to Community Analysis"
           - [OK] GOOD: "OpenAI is drawing unusually high attention right now, and most of the recent drivers are risk/negative vs growth/positive (cite the events)."
        
        3. **Human-Readable Insights**: Translate technical metrics into business insights.
           - Turn "score: 4374.5, slope: 410.0" into "attention is high and increasing quickly" (but do not claim business growth unless the snippet evidence supports it)
           - Turn "mentions: 218" into "dominating industry conversations" or "leading the narrative"
           - Turn "rank: 1" into "the center of the current conversation" (not "market leader")
        
        [WARN] CRITICAL PRODUCT ATTRIBUTION RULES (MUST FOLLOW EXACTLY):
        
        **STEP 1: IDENTIFY THE COMPANY FROM CONTEXT**
        - Look for explicit statements like "xAI released Grok", "OpenAI announced GPT-4", "Google unveiled Gemini"
        - The company name MUST appear in the same sentence or immediately adjacent to the product name
        - If you see "Grok" mentioned alone, look for "xAI" or "Elon Musk" nearby - NOT OpenAI
        
        **STEP 2: VERIFY AGAINST THIS MASTER LIST**
        - **OpenAI** (Sam Altman): ChatGPT, GPT-4, GPT-4o, GPT-4o mini, o1, o1-mini, o1-preview, Sora, DALL-E, Codex
        - **xAI** (Elon Musk): Grok, Grok 2, Grok 3, Grok 4, Grok 4.1, Grok 4.1 Fast - ALL GROK VERSIONS BELONG TO xAI
        - **Google/DeepMind**: Gemini, Gemini Pro, Gemini Ultra, Bard, PaLM, PaLM 2, Imagen, Gemini Flash
        - **Anthropic**: Claude, Claude 2, Claude 3, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku, Claude 3.5
        - **Meta**: Llama, Llama 2, Llama 3, Llama 3.1, Llama 3.2, Code Llama
        - **Microsoft**: Copilot, Bing Chat (uses OpenAI models but is Microsoft's product)
        - **Mistral AI**: Mistral, Mixtral, Mistral Large
        
        **STEP 3: CROSS-CHECK FOR COMMON ERRORS**
        - [NO] WRONG: "OpenAI's Grok" (Grok is xAI's product)
        - [NO] WRONG: "Google's ChatGPT" (ChatGPT is OpenAI's product)
        - [NO] WRONG: "Anthropic's Gemini" (Gemini is Google's product)
        - [OK] CORRECT: "xAI's Grok 4.1 Fast achieved exceptional performance"
        - [OK] CORRECT: "OpenAI's ChatGPT continues to dominate"
        - [OK] CORRECT: "Google's Gemini competes with OpenAI's GPT-4"
        
        **STEP 4: IF UNSURE, USE NEUTRAL LANGUAGE**
        - Instead of "OpenAI released Grok", say "Grok was released" (then identify xAI if context allows)
        - Better yet: Skip the detail if attribution is unclear
        
        **ENFORCEMENT**: If you attribute a product to the wrong company, the entire response is INVALID.
        """

        industry_filter_instruction = ""
        if industry:
            industry_filter_instruction = f"""
        3. **Industry Relevance (STRICT)**: The user is asking specifically about the '{industry}' industry. 
           - **ONLY** include entities and trends that are explicitly mentioned in the context as being related to {industry}. 
           - **EXCLUDE** general AI news (e.g., generic model releases from xAI, OpenAI, Google) unless the text *explicitly* mentions their application in {industry}.
           - If the context contains no specific companies for {industry}, state that no specific {industry} startups were found in the recent news, and only discuss the general trends found in the Knowledge Graph.
           - **CRITICAL**: Do not list companies like Anthropic, xAI, or Google just because they are in the text. They must be doing something IN {industry}.
           - **CRITICAL**: Do not hallucinate relationships. If the text says "Anthropic developed Grok", it is WRONG. Ignore it. If the text says "xAI released Grok", that is correct.
            """

        final_prompt = f"""
        {system_instruction}
        {fact_check_instruction}
        
        CRITICAL FINAL VERIFICATION CHECKLIST (Check before responding):
        
        [OK] STEP 1: For EVERY product mentioned, verify the company:
           - Grok -> Must be xAI (NOT OpenAI)
           - ChatGPT/GPT-4/o1 -> Must be OpenAI (NOT xAI)
           - Gemini -> Must be Google (NOT OpenAI or Anthropic)
           - Claude -> Must be Anthropic (NOT OpenAI or Google)
        
        [OK] STEP 2: Check the source text for explicit attribution:
           - Look for "xAI released Grok" [OK] CORRECT
           - Reject "OpenAI's Grok" [FAIL] WRONG - Fix to "xAI's Grok"
           - If you see only "Grok 4.1 Fast" without company, search context for xAI/Elon Musk mentions
        
        [OK] STEP 3: Red flags to watch for:
           - Multiple products from different companies in one sentence (risk of mixing them up)
           - Headlines that juxtapose companies (e.g., "OpenAI vs Grok" - doesn't mean OpenAI owns Grok)
           - Table data or pricing lists (often garbled - ignore if unclear)
        
        [OK] STEP 4: Before finalizing, ask yourself:
           - "Did I correctly attribute EVERY product to the right company?"
           - "Did I verify against the master product list above?"
           - "If I'm unsure about a product's owner, did I use neutral language or omit it?"
        
        FORMATTING INSTRUCTIONS:
        1. **Formatting**:
           - Provide the narrative answer in standard text.
           - IF you need to present a table (for comparisons, lists, or data), DO NOT use Markdown tables.
           - Instead, output the table data in CSV format wrapped in <csv_table> tags.
           - The CSV should have a header row.
           - Use a comma (,) as the delimiter.
           - Quote fields if they contain commas.
        {industry_filter_instruction}
        
        Here is the context data I have gathered. Use it to answer the user's question.
        
        --- CONTEXT ---
        
        **Recent News Snippets:**
        {vector_context}
        
        **Key Facts from Knowledge Graph:**
        {graph_context_str}
        
        **High-Level Market Analysis & Rankings:**
        {community_context_str}
        
        --- QUESTION ---
        {query}
        
        Answer:
        """
        
        response = ollama.chat(model=config.LLM_MODEL, messages=[{'role': 'user', 'content': final_prompt}])
        answer = response['message']['content']
        
        # Apply post-processing validation to catch attribution errors
        answer = self.validate_product_attribution(answer)
        
        if return_context:
            return {
                "answer": answer,
                "vector_context": vector_results,
                "graph_context": graph_context_str,
                "community_context": community_context_str,
                "entity_detected": entities,
                "intent": intent
            }
        return answer

if __name__ == "__main__":
    # Test the backend with multiple queries
    backend = TrendScoutBackend()
    
    test_queries = [
        "Which startups are trending right now?",  # Tests Ranking
        "What is the growth forecast for OpenAI?", # Tests Forecasting
        "Who are the top competitors in the AI space?", # Tests Graph/Ranking
        "What are the emerging threats for OpenAI?", # Tests SWOT + Community
        "List the top trending entities." # Explicit ranking request
    ]
    
    print(f"\n{'='*50}")
    print("RUNNING RETRIEVAL VERIFICATION SUITE")
    print(f"{'='*50}\n")
    
    for q in test_queries:
        print(f"QUERY: {q}")
        print("-" * 20)
        
        try:
            result = backend.generate_answer(q, return_context=True)
            
            print(f"1. DETECTED ENTITIES: {result['entity_detected']}")
            print(f"2. INTENT: {result['intent']}")
            
            print(f"3. GRAPH CONTEXT (First 200 chars):")
            if result['graph_context']:
                print(f"   {result['graph_context'][:200].replace(chr(10), ' ')}...")
            else:
                print("   [No Graph Context Found]")

            print(f"4. COMMUNITY CONTEXT (First 200 chars):")
            if result.get('community_context'):
                print(f"   {result['community_context'][:200].replace(chr(10), ' ')}...")
            else:
                print("   [No Community Context Found]")
                
            print(f"5. VECTOR CONTEXT (Sources):")
            for doc in result['vector_context']:
                print(f"   - {doc['source']} (Score: {doc['score']:.4f})")
                
            print(f"5. FINAL ANSWER:")
            print(f"   {result['answer'][:300].replace(chr(10), ' ')}...")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"\n{'='*50}\n")
