import os
import json
import pandas as pd
from neo4j import GraphDatabase
import config
from ast import literal_eval
import networkx as nx

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
        """
        I'm setting up the rules for our graph database.
        We need to make sure articles, entities, and industries are unique so we don't have duplicates messing things up.
        I'll also add some indexes to make searching faster.
        """
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("Constraints and indexes created.")

    def load_data(self):
        """
        I'm pulling in all the data we've processed so far.
        I need the extracted entities/KPIs, the original article snippets, and our entity mapping file to clean up names.
        """
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
        """
        This is the main event. I'm going to iterate through every article and build our Knowledge Graph.
        I'll create nodes for Articles, Industries, Entities, and Metrics, and connect them all together.
        """
        df_kpi, df_snippets, entity_map = self.load_data()
        if df_kpi is None:
            return

        print(f"Building graph from {len(df_kpi)} extraction records and {len(df_snippets)} snippets...")
        
        # Index snippets by id for fast lookup
        snippets_dict = df_snippets.set_index('snippet_id').to_dict('index')
        
        # Group extractions by snippet_id
        grouped = df_kpi.groupby('snippet_id')
        
        with self.driver.session() as session:
            count = 0
            for snippet_id, group in grouped:
                count += 1
                if count % 50 == 0:
                    print(f"Processed {count} articles...")
                    
                # First, let's grab the basic info about the article (title, source, date).
                meta = snippets_dict.get(snippet_id, {})
                title = meta.get('title', 'Unknown Title')
                source = meta.get('source', 'Unknown Source')
                link = meta.get('link', '')
                published = str(meta.get('published', ''))
                
                # Now I'll figure out which industry this article belongs to and collect all the SWOT points.
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
                
                # I'm packing all the article details into a dictionary so I can send it to Neo4j.
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
                
                # Time to create the nodes! I'll make the Article node and the Industry node, and link them up.
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
                
                # Now for the interesting stuff: the companies and the numbers.
                # I'll separate the entity mentions from the key performance indicators.
                entity_rows = group[group['category'] == 'Entity']
                kpi_rows = group[group['category'] == 'KPI']
                
                # I'm getting the list of entities ready.
                # I'll use our canonical name map to fix any typos or variations (like 'Google Inc' -> 'Google').
                entities_to_batch = []
                
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

                # Finally, let's add the metrics (Revenue, Funding, etc.) and link them to the article.
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

        print("Graph build complete.")

if __name__ == "__main__":
    builder = GraphBuilder()
    try:
        builder.create_constraints()
        builder.build_graph()
    finally:
        builder.close()
