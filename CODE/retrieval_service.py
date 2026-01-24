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
        rank_path = os.path.join(base, "entity_ranking.parquet")
        temp_feat_path = os.path.join(base, "temporal_features.parquet")
        forecast_path = os.path.join(base, "community_forecast.parquet")

        if not (os.path.exists(comm_path) and os.path.exists(ent_path)):
            # Minimal requirement: communities + entity summary. Others are optional.
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
        self.df_ent = pd.read_parquet(ent_path)
        self.df_swot = pd.read_parquet(swot_path) if os.path.exists(swot_path) else None
        self.df_temp = pd.read_parquet(temp_path) if os.path.exists(temp_path) else None
        self.df_rank = pd.read_parquet(rank_path) if os.path.exists(rank_path) else None
        self.df_temp_feat = pd.read_parquet(temp_feat_path) if os.path.exists(temp_feat_path) else None
        self.df_forecast = pd.read_parquet(forecast_path) if os.path.exists(forecast_path) else None
        self._loaded = True

    def get_entity_communities(self, entity_name: str) -> List[int]:
        """Get community IDs where the specified entity appears."""
        
        self._ensure_loaded()
        if self.df_ent is None:
            return []
        sub = self.df_ent[self.df_ent["entity_name"].str.lower() == entity_name.lower()]
        return sorted(sub["community_id"].dropna().unique().tolist())

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
            entity_rank_data = self.df_rank[self.df_rank['entity_name'].str.lower() == entity_name.lower()]
            if not entity_rank_data.empty:
                # Take the highest score if entity is in multiple clusters
                top_rank = entity_rank_data.sort_values('score', ascending=False).iloc[0]
                metrics['kpi_stance_score'] = top_rank.get('avg_stance', 0.0)

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
        
        sub = self.df_rank.sort_values("rank").head(top_k)
        return [
            {
                "rank": int(row["rank"]),
                "entity_name": row["entity_name"],
                "score": float(row["score"]),
                "mentions": int(row["mentions"]),
                "growth_slope": float(row["slope"])
            }
            for _, row in sub.iterrows()
        ]

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
                "published": published_str
            })

        # Sort by final score and take top_k
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        return processed_results[:top_k]

class GraphStore:
    def __init__(self):
        self.driver = None
        if GraphDatabase is None:
            print("Graph store unavailable: 'neo4j' is not installed.")
            return
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            )
            self.verify_connection()
        except Exception as e:
            print(f"Graph Store connection failed: {e}")

    def verify_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")

    def close(self):
        if self.driver:
            self.driver.close()

    def get_kpi_breakdown_for_entity(self, entity_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get detailed KPI information for an entity from the Snippet-KPI graph.
        Returns structured KPIs organized by type: funding, hiring, partnership, product.
        """
        if not self.driver:
            return {"funding": [], "hiring": [], "partnership": [], "product": []}
        
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
        if ollama is None:
            raise RuntimeError("Ollama is required to generate answers. Install dependencies and ensure Ollama is running.")
        # Step 1: Figure out what the user wants
        intent_data = self.extract_query_intent(query)
        entities = intent_data.get('entities', [])
        
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
                    csv_rows.append(f"{t['rank']},{t['entity_name']},{t['score']:.1f},{t['growth_slope']:.1f}")
                
                # Wrap in a special tag so the UI can render it as a dataframe
                community_context.append(f"COMMUNITY-LEVEL ANALYSIS (TOP 10 TRENDING):\n<csv_table>\n" + "\n".join(csv_rows) + "\n</csv_table>")

        graph_context_str = "\n".join(graph_context)
        community_context_str = "\n".join(community_context)

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
