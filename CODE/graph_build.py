import os
import json
import pandas as pd
from neo4j import GraphDatabase
import config
from ast import literal_eval
import networkx as nx
from datetime import datetime
from collections import defaultdict
import numpy as np

class GraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.verify_connection()

    def verify_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                print(f"Connected to Neo4j: {result.single()['num'] == 1}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create graph database constraints and indexes for unique nodes and performance."""
        
        # Drop incorrect KPI constraint on 'type' property if exists
        with self.driver.session() as session:
            try:
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    record_dict = dict(record)
                    constraint_name = record_dict.get('name', '')
                    labels = record_dict.get('labelsOrTypes', [])
                    properties = record_dict.get('properties', [])
                    
                    # Drop KPI constraint on 'type' property (incorrect)
                    if 'KPI' in labels and 'type' in properties:
                        try:
                            session.run(f"DROP CONSTRAINT {constraint_name}")
                            print(f"Dropped incorrect constraint: {constraint_name} (KPI.type)")
                        except Exception as e:
                            print(f"Could not drop {constraint_name}: {e}")
            except Exception as e:
                print(f"Error checking constraints: {e}")
        
        # Create correct constraints
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Snippet) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:KPI) REQUIRE k.id IS UNIQUE",
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    print(f"Warning creating constraint: {e}")
        print("Constraints and indexes created.")

    def load_data(self):
        """Load processed data from parquet files: KPI entities, snippets, and entity mapping."""
        
        print("Loading data...")
        
        # Load extraction results
        kpi_path = config.KPI_ENTITIES_FILE
        if not os.path.exists(kpi_path):
            print(f"No extracted data found at {kpi_path}")
            return None, None, None
            
        df_kpi = pd.read_parquet(kpi_path)
        
        # Load snippets for metadata
        snippets_path = config.SNIPPETS_FILE
        if not os.path.exists(snippets_path):
            print(f"No snippets data found at {snippets_path}")
            return None, None, None
            
        df_snippets = pd.read_parquet(snippets_path)
        
        # Load entity map
        map_path = config.ENTITY_MAP_FILE
        if os.path.exists(map_path):
            df_map = pd.read_parquet(map_path)
            # Create a dictionary for fast lookup
            entity_map = dict(zip(df_map['raw_name'], df_map['canonical_name']))
        else:
            entity_map = {}
            
        return df_kpi, df_snippets, entity_map

    def build_graph(self):
        """Build knowledge graph by creating nodes and edges from extracted data.
        Creates Article, Entity, Industry, Snippet, and KPI nodes with relationships.
        """
        
        df_kpi, df_snippets, entity_map = self.load_data()
        if df_kpi is None:
            return

        print(f"Building graph from {len(df_kpi)} extraction records and {len(df_snippets)} snippets...")
        
        # Index snippets by id for fast lookup
        snippets_dict = df_snippets.set_index('snippet_id').to_dict('index')
        
        # Group extractions by snippet_id
        grouped = df_kpi.groupby('snippet_id')
        
        # Track entities per article for co-occurrence calculation
        article_entities = {}  # {article_id: set(entity_names)}
        article_dates = {}  # {article_id: datetime}
        
        with self.driver.session() as session:
            count = 0
            for snippet_id, group in grouped:
                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} articles...")
                    
                # Get article metadata from snippets
                meta = snippets_dict.get(snippet_id, {})
                title = meta.get('title', 'Unknown Title')
                source = meta.get('source', 'Unknown Source')
                link = meta.get('link', '')
                published = str(meta.get('published', ''))
                
                # Determine industry and collect SWOT data
                industries = group['industry'].dropna().unique()
                main_industry = industries[0] if len(industries) > 0 else "Unclassified"
                
                # Collect SWOT
                swot_data = {'Strength': [], 'Weakness': [], 'Opportunity': [], 'Threat': []}
                swot_rows = group[group['category'] == 'SWOT']
                for _, row in swot_rows.iterrows():
                    dtype = row['detail_type']
                    dval = row['detail_value']
                    if dtype in swot_data and dval:
                        swot_data[dtype].append(dval)
                
                # Prepare article properties for Neo4j
                article_props = {
                    'id': snippet_id,
                    'title': title,
                    'source': source,
                    'link': link,
                    'published': published,
                    'swot_Strength': swot_data['Strength'],
                    'swot_Weakness': swot_data['Weakness'],
                    'swot_Opportunity': swot_data['Opportunity'],
                    'swot_Threat': swot_data['Threat']
                }
                
                # Create Article and Industry nodes with relationship
                session.run("""
                    MERGE (a:Article {id: $id})
                    SET a.title = $title,
                        a.source = $source,
                        a.link = $link,
                        a.published = $published,
                        a.swot_Strength = $swot_Strength,
                        a.swot_Weakness = $swot_Weakness,
                        a.swot_Opportunity = $swot_Opportunity,
                        a.swot_Threat = $swot_Threat
                    
                    MERGE (i:Industry {name: $industry})
                    MERGE (a)-[:BELONGS_TO]->(i)
                """, {**article_props, 'industry': main_industry})
                
                # Separate entity mentions from KPIs
                entity_rows = group[group['category'] == 'Entity']
                kpi_rows = group[group['category'] == 'KPI']
                
                # Prepare entities using canonical name mapping
                entities_to_batch = []
                entity_names = set()
                
                for _, row in entity_rows.iterrows():
                    raw_name = row['entity_name']
                    if not raw_name: continue
                    
                    name = entity_map.get(raw_name, raw_name)
                    etype = row['entity_type'] if row['entity_type'] else 'Unknown'
                    stance = row['stance']
                    
                    entities_to_batch.append({
                        'name': name,
                        'type': etype,
                        'stance': stance
                    })
                    entity_names.add(name)
                
                # Store entities and date for co-occurrence calculation
                article_entities[snippet_id] = entity_names
                try:
                    article_dates[snippet_id] = datetime.fromisoformat(published.replace('Z', '+00:00')) if published else datetime.now()
                except:
                    article_dates[snippet_id] = datetime.now()
                
                # Sending the entities to the graph in a batch is much faster than doing them one by one.
                if entities_to_batch:
                    session.run("""
                        MATCH (a:Article {id: $article_id})
                        UNWIND $entities AS ent
                        MERGE (e:Entity {name: ent.name})
                        ON CREATE SET e.type = ent.type
                        MERGE (a)-[r:MENTIONS]->(e)
                        SET r.stance = ent.stance
                    """, {'article_id': snippet_id, 'entities': entities_to_batch})

                # Create KPI nodes and relationships
                kpis_to_batch = []
                for _, row in kpi_rows.iterrows():
                    k_name = row['detail_type']
                    k_value = row['detail_value']
                    if k_name and k_value:
                        kpis_to_batch.append({'name': k_name, 'value': k_value})
                        
                if kpis_to_batch:
                    session.run("""
                        MATCH (a:Article {id: $article_id})
                        UNWIND $kpis AS k
                        MERGE (m:Metric {name: k.name})
                        MERGE (a)-[r:REPORTED_METRIC]->(m)
                        SET r.value = k.value
                    """, {'article_id': snippet_id, 'kpis': kpis_to_batch})
                
                # Create Snippet nodes with KPI stance (for KPI graph Gᵏ)
                snippets_to_batch = []
                for idx, row in kpi_rows.iterrows():
                    kpi_type = row['detail_type']
                    kpi_value = row['detail_value']
                    # Determine stance/polarity from context
                    stance_value = self._determine_kpi_stance(kpi_type, kpi_value)
                    
                    if kpi_type and kpi_value:
                        # Create a unique ID for the KPI node - use row index to ensure uniqueness
                        kpi_node_id = f"{snippet_id}_{kpi_type}_{idx}"
                        snippets_to_batch.append({
                            'id': snippet_id, # This is the Snippet ID
                            'text': kpi_value,
                            'date': published,
                            'kpi': {'id': kpi_node_id, 'type': kpi_type},
                            'stance': stance_value
                        })
                
                if snippets_to_batch:
                    session.run("""
                        MATCH (a:Article {id: $article_id})
                        UNWIND $snippets AS s
                        MERGE (sn:Snippet {id: s.id}) // Use the actual snippet_id
                        SET sn.text = s.text, sn.date = s.date
                        MERGE (sn)-[:IN]->(a)
                        
                        MERGE (k:KPI {id: s.kpi.id})
                        SET k.type = s.kpi.type, k.stance = s.stance, k.date = s.date
                        MERGE (sn)-[:ABOUT]->(k)
                    """, {'article_id': snippet_id, 'snippets': snippets_to_batch})

        print("Graph build phase 1 complete. Now creating Article-Article edges...")
        self._create_article_colinks(article_entities, article_dates)
        print("Graph build complete.")
    
    def _determine_kpi_stance(self, kpi_type: str, kpi_value: str) -> float:
        """Determine positive/negative/neutral stance from KPI context."""
        kpi_value_lower = kpi_value.lower() if kpi_value else ""
        
        # Positive indicators
        positive_words = ['raised', 'secured', 'growth', 'increase', 'expansion', 'launched', 
                         'acquired', 'partnership', 'funding', 'investment', 'hired']
        # Negative indicators  
        negative_words = ['layoff', 'shutdown', 'decline', 'loss', 'lawsuit', 'breach', 
                         'failure', 'delay', 'issue', 'problem', 'risk']
        
        pos_count = sum(1 for word in positive_words if word in kpi_value_lower)
        neg_count = sum(1 for word in negative_words if word in kpi_value_lower)
        
        if pos_count > neg_count:
            return 1.0
        elif neg_count > pos_count:
            return -1.0
        else:
            return 0.0
    
    def _create_article_colinks(self, article_entities: dict, article_dates: dict):
        """Create weighted Article-Article edges based on Jaccard similarity of shared entities."""
        print("Calculating Jaccard similarity for article pairs...")
        
        article_ids = list(article_entities.keys())
        edges_to_create = []
        tau = 30  # Recency decay parameter (days)
        now = pd.Timestamp.now(tz='UTC')  # Use timezone-aware timestamp
        
        for i, aid1 in enumerate(article_ids):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(article_ids)} articles...")
            
            entities1 = article_entities[aid1]
            if len(entities1) == 0:
                continue
            
            date1 = article_dates.get(aid1, now)
            # Ensure date1 is timezone-aware
            if date1.tzinfo is None:
                date1 = pd.Timestamp(date1, tz='UTC')
            
            # Only compare with subsequent articles to avoid duplicates
            for aid2 in article_ids[i+1:]:
                entities2 = article_entities[aid2]
                if len(entities2) == 0:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(entities1 & entities2)
                if intersection == 0:
                    continue
                    
                union = len(entities1 | entities2)
                jaccard = intersection / union
                
                # Apply recency decay
                date2 = article_dates.get(aid2, now)
                # Ensure date2 is timezone-aware
                if date2.tzinfo is None:
                    date2 = pd.Timestamp(date2, tz='UTC')
                    
                avg_date = date1 if date1 > date2 else date2  # Use newer date
                days_old = (now - avg_date).days
                recency_weight = np.exp(-days_old / tau)
                
                # Final weight
                weight = jaccard * recency_weight
                
                if weight > 0.1:  # Threshold to avoid too many weak edges
                    edges_to_create.append({
                        'aid1': aid1,
                        'aid2': aid2,
                        'weight': weight,
                        'shared_entities': intersection
                    })
        
        print(f"Creating {len(edges_to_create)} CO_LINK relationships...")
        
        with self.driver.session() as session:
            # Batch create edges
            for i in range(0, len(edges_to_create), 500):
                batch = edges_to_create[i:i+500]
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (a1:Article {id: edge.aid1})
                    MATCH (a2:Article {id: edge.aid2})
                    MERGE (a1)-[r:CO_LINK]-(a2)
                    SET r.weight = edge.weight, r.shared_entities = edge.shared_entities
                """, {'edges': batch})
                if i % 1000 == 0 and i > 0:
                    print(f"  Created {i} edges...")
        
        print(f"✅ Created {len(edges_to_create)} weighted Article-Article edges")

if __name__ == "__main__":
    builder = GraphBuilder()
    try:
        builder.create_constraints()
        builder.build_graph()
    finally:
        builder.close()
