import os
import pandas as pd
import numpy as np
import ollama
from neo4j import GraphDatabase
import config
from typing import List, Dict, Any
import json
from datetime import datetime

import re


class CommunityAnalytics:
    """
    I'm the bridge to our community analysis data.
    I read the parquet files we generated earlier to answer questions about
    which communities entities belong to, what the SWOT breakdown is, and how things change over time.
    I don't need to talk to the graph database directly; I just use the pre-computed stats.
    """

    def __init__(self):
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        base = config.DATA_DIR
        comm_path = os.path.join(base, "article_communities.parquet")
        ent_path = os.path.join(base, "community_entity_summary.parquet")
        swot_path = os.path.join(base, "community_swot_summary.parquet")
        temp_path = os.path.join(base, "community_temporal_summary.parquet")

        if not (os.path.exists(comm_path) and os.path.exists(ent_path)):
            # Minimal requirement: communities + entity summary. Others are optional.
            self._loaded = True
            self.df_comm = None
            self.df_ent = None
            self.df_swot = None
            self.df_temp = None
            return

        self.df_comm = pd.read_parquet(comm_path)
        self.df_ent = pd.read_parquet(ent_path)
        self.df_swot = pd.read_parquet(swot_path) if os.path.exists(swot_path) else None
        self.df_temp = pd.read_parquet(temp_path) if os.path.exists(temp_path) else None
        self._loaded = True

    def get_entity_communities(self, entity_name: str) -> List[int]:
        """
        Let's find out which neighborhoods (communities) this entity hangs out in.
        I'll look up the entity name and return the IDs of the communities where it appears.
        """
        self._ensure_loaded()
        if self.df_ent is None:
            return []
        sub = self.df_ent[self.df_ent["entity_name"].str.lower() == entity_name.lower()]
        return sorted(sub["community_id"].dropna().unique().tolist())

    def get_top_entities_in_community(self, community_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Who are the big players in this community?
        I'll grab the top entities based on how often they're mentioned.
        """
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
        """
        What's the strategic outlook for this community?
        I'll sum up the Strengths, Weaknesses, Opportunities, and Threats counts.
        """
        self._ensure_loaded()
        if self.df_swot is None:
            return {}
        sub = self.df_swot[self.df_swot["community_id"] == community_id]
        if sub.empty:
            return {}
        agg = sub.groupby("swot_type")["count"].sum().to_dict()
        return {str(k): int(v) for k, v in agg.items()}

    def get_temporal_for_entity(self, entity_name: str, community_id: int | None = None) -> List[Dict[str, Any]]:
        """
        Let's see how the conversation volume has changed over time.
        I'm looking at the monthly mention counts for the community to give a sense of momentum.
        """
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

class VectorStore:
    def __init__(self):
        self.embeddings_path = os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")
        self.df = None
        self.embeddings_matrix = None
        self.load_data()

    def clean_text(self, text: str) -> str:
        """
        Sometimes the text is a mess, especially with tables and prices.
        I'm going to clean it up so the LLM doesn't get confused by a soup of numbers.
        """
        # Pattern: Multiple occurrences of prices ($X.XX) in close proximity
        # If a segment has more than 3 prices, it's likely a flattened table.
        if text.count('$') > 2:
            # Aggressive regex to catch "Company Model $Price" repeating patterns
            # Matches: (Words/Symbols) + $Price + (Words/Symbols) + $Price ...
            # We look for at least 2 occurrences of "Words $Price" to identify a table row/col sequence
            pattern = r'((?:[A-Za-z0-9\.\(\)\-\>\<\=]+\s+){1,8}\$\d+\.\d+(?:\s+|$)){2,}'
            text = re.sub(pattern, ' [TABLE DATA REMOVED] ', text)
            
            # Fallback: if there are still too many prices, just truncate or warn
            if text.count('$') > 4:
                 text = re.sub(r'\$.*?\$', ' [PRICING] ', text) # Remove text between dollar signs
        return text

    def load_data(self):
        if os.path.exists(self.embeddings_path):
            self.df = pd.read_parquet(self.embeddings_path)
            # Convert list of floats to numpy matrix
            if 'embedding' in self.df.columns and not self.df.empty:
                # Ensure embeddings are lists, then stack
                valid_df = self.df.dropna(subset=['embedding'])
                self.embeddings_matrix = np.vstack(valid_df['embedding'].values)
                self.df = valid_df
                
                # Parse dates for temporal scoring
                self.df['parsed_date'] = pd.to_datetime(self.df['published'], errors='coerce', utc=True)
                
                print(f"Vector Store loaded: {len(self.df)} documents.")
        else:
            print("Vector Store not found. Run rag_index.py first.")

    def search(self, query: str, top_k: int = 5, recency_bias: bool = False) -> List[Dict[str, Any]]:
        if self.df is None or self.embeddings_matrix is None:
            return []

        # Embed query
        response = ollama.embeddings(model=config.LLM_MODEL, prompt=query)
        query_embedding = np.array(response['embedding'])

        # Cosine similarity
        norm_query = np.linalg.norm(query_embedding)
        norm_matrix = np.linalg.norm(self.embeddings_matrix, axis=1)
        
        dot_products = np.dot(self.embeddings_matrix, query_embedding)
        similarities = dot_products / (norm_matrix * norm_query)
        
        # Apply Recency Bias if requested
        final_scores = similarities
        if recency_bias:
            now = pd.Timestamp.now(tz='UTC')
            # Calculate days old (fill NaT with 365 days)
            days_old = (now - self.df['parsed_date']).dt.days.fillna(365)
            # Decay function: score = similarity * (1 / (1 + 0.01 * days_old))
            # This means a 100-day old article has half the weight of a new one
            decay_factor = 1 / (1 + 0.005 * days_old.values)
            final_scores = similarities * decay_factor

        # Get top K indices
        top_indices = np.argsort(final_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            cleaned_text = self.clean_text(row['text'])
            results.append({
                "score": float(final_scores[idx]),
                "text": cleaned_text,
                "source": row.get('source', 'Unknown'),
                "published": row.get('published', '')
            })
        return results

class GraphStore:
    def __init__(self):
        self.driver = None
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

    def get_entity_context(self, entity_names: List[str]) -> List[str]:
        """Get direct relationships and SWOT data for a list of entities."""
        if not self.driver or not entity_names:
            return []

        query = """
        UNWIND $names AS name
        MATCH (e:Entity {name: name})<-[r:MENTIONS]-(a:Article)
        OPTIONAL MATCH (a)-[:BELONGS_TO]->(i:Industry)
        OPTIONAL MATCH (a)-[rm:REPORTED_METRIC]->(m:Metric)
        RETURN e.name as entity, i.name as industry, 
               a.title as title,
               r.stance as stance,
               collect(DISTINCT {name: m.name, value: rm.value}) as metrics
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
                
                # Removed generic SWOT attribution to prevent hallucination (e.g. attributing Article strengths to all mentioned entities)
                
                metrics = [m for m in record['metrics'] if m['name'] is not None]
                if metrics:
                    m_list = [f"{m['name']}: {m['value']}" for m in metrics]
                    info += f" | Metrics: {', '.join(m_list)}"
                
                context.append(info)
        return context

    def get_industry_entities(self, industry_name: str) -> List[str]:
        """Get top entities and trends for an industry."""
        if not self.driver:
            return []
            
        query = """
        MATCH (i:Industry {name: $name})<-[:BELONGS_TO]-(a:Article)-[:MENTIONS]->(e:Entity)
        RETURN e.name as entity, count(a) as mentions, avg(a.sentiment) as avg_sentiment
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
        self.community_analytics = CommunityAnalytics()

    def extract_query_intent(self, query: str) -> dict:
        """
        I need to understand what the user is really asking.
        I'll ask the LLM to break down the query into entities, industries, and the type of answer they want (list, comparison, etc.).
        """
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
        industry = intent_data.get('industry')
        intent = intent_data.get('intent')
        is_temporal = intent_data.get('temporal', False)

        # Step 2: Search the vector database for relevant articles
        # If the user asked for "recent" news, I'll prioritize newer stuff.
        vector_results = self.vector_store.search(query, top_k=5, recency_bias=is_temporal)
        vector_context = "\n".join([f"- {r['text']} (Source: {r['source']}, Date: {r['published']})" for r in vector_results])

        # Step 3: Dig into the Knowledge Graph and Community Analytics
        graph_context = []
        community_context = []

        # If we found specific companies, let's get their detailed stats
        if entities:
            graph_context.extend(self.graph_store.get_entity_context(entities))

            # I'll also pull in the community data for the main entity mentioned.
            # This gives us the "neighborhood" context—who else is in their cluster, what are the SWOTs, etc.
            primary_entity = entities[0]
            comm_ids = self.community_analytics.get_entity_communities(primary_entity)
            if comm_ids:
                cid = comm_ids[0]
                top_ents = self.community_analytics.get_top_entities_in_community(cid)
                swot = self.community_analytics.get_swot_for_community(cid)
                temporal = self.community_analytics.get_temporal_for_entity(primary_entity, community_id=cid)

                if top_ents:
                    ent_str = "; ".join([
                        f"{e['entity_name']} (mentions={e['mentions']}, avg_stance={e['avg_stance']})"
                        for e in top_ents
                    ])
                    community_context.append(
                        f"Community {cid} key entities: {ent_str}"
                    )
                if swot:
                    swot_str = ", ".join([f"{k}: {v}" for k, v in swot.items()])
                    community_context.append(
                        f"Community {cid} SWOT distribution (counts): {swot_str}"
                    )
                if temporal:
                    # I'll just show the last 6 months of data to keep it concise
                    temp_str = ", ".join([
                        f"{t['year_month']}: {t['entity_mentions']} mentions"
                        for t in temporal[-6:]
                    ])
                    community_context.append(
                        f"Community {cid} recent temporal trend (last up to 6 periods): {temp_str}"
                    )

        # If the user mentioned an industry, I'll get the top players in that field
        if industry:
            graph_context.extend(self.graph_store.get_industry_entities(industry))

        graph_context_str = "\n".join(graph_context)
        community_context_str = "\n".join(community_context)

        # Step 4: Put it all together and ask the LLM for the final answer
        system_instruction = "You are TrendScout AI, a market intelligence analyst."
        if intent == 'comparison':
            system_instruction += " Provide a comparative analysis (SWOT) of the entities mentioned."
        elif intent == 'list':
            system_instruction += " Provide a list or table of the relevant companies/trends. Highlight recent developments."
        
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
        
        CRITICAL INSTRUCTIONS:
        1. **Attribution Accuracy**: When attributing products, releases, or actions to companies, verify strictly against the provided context. 
           - Example: If text says "xAI released Grok to compete with OpenAI", do NOT say OpenAI released Grok.
           - **WARNING**: The context may contain placeholders like `[TABLE DATA REMOVED]`. This indicates malformed text. **DO NOT** try to interpret the words immediately surrounding these placeholders as valid statements of fact.
           - **WARNING**: Be careful with headlines like "Anthropic vs Grok" or "OpenAI on Gemini". Do not assume the first entity owns the second product. Only attribute ownership if the text explicitly says "released", "launched", or "developed".
           - If the context is ambiguous or looks like a jumbled list of names and prices, **IGNORE IT**.
        2. **Formatting**:
           - Provide the narrative answer in standard text.
           - IF you need to present a table (for comparisons, lists, or data), DO NOT use Markdown tables.
           - Instead, output the table data in CSV format wrapped in <csv_table> tags.
           - The CSV should have a header row.
           - Use a comma (,) as the delimiter.
           - Quote fields if they contain commas.
        {industry_filter_instruction}
        
        Example Table:
        <csv_table>
        Entity,Strengths,Weaknesses
        "Company A","Strong Brand","High Cost"
        "Company B","Innovation","Small Market Share"
        </csv_table>
        
        Context:
        --- RECENT NEWS (Vector) ---
        {vector_context}
        
        --- KNOWLEDGE GRAPH (Structured) ---
        {graph_context_str}
        
        --- COMMUNITY-LEVEL ANALYSIS (Structured Aggregates) ---
        {community_context_str}
        
        --- QUESTION ---
        {query}
        
        Answer:
        """
        
        response = ollama.chat(model=config.LLM_MODEL, messages=[{'role': 'user', 'content': final_prompt}])
        answer = response['message']['content']
        
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
        "What is the sentiment around OpenAI?",
        "What are the opportunities in the Generative AI industry?",
        "Tell me about the new coding models released.",
        "What are the threats to AI startups?",
        "Compare OpenAI and Google in terms of AI dominance.",
        "List the top AI startups in Healthcare."
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
