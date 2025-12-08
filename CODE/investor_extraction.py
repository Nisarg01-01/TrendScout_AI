"""
Investor Extraction and Quality Scoring
Extracts investor names from funding KPIs and assigns prestige scores
"""

import os
import pandas as pd
import json
from neo4j import GraphDatabase
import config
from typing import Dict, List

# Prestige scores for well-known investors (0-1 scale)
INVESTOR_PRESTIGE = {
    # Tier 1: Top-tier VCs
    'Sequoia Capital': 1.0,
    'Andreessen Horowitz': 1.0,
    'a16z': 1.0,
    'Benchmark': 1.0,
    'Accel': 0.95,
    'Greylock': 0.95,
    'Kleiner Perkins': 0.95,
    'Lightspeed': 0.9,
    'Index Ventures': 0.9,
    'Founders Fund': 0.9,
    
    # Tier 2: Strong VCs
    'NEA': 0.85,
    'General Catalyst': 0.85,
    'Insight Partners': 0.85,
    'Bessemer': 0.85,
    'Battery Ventures': 0.8,
    'Redpoint': 0.8,
    'GV': 0.8,  # Google Ventures
    'Microsoft Ventures': 0.8,
    
    # Tier 3: Corporate VCs / Good firms
    'Intel Capital': 0.75,
    'Salesforce Ventures': 0.75,
    'Google Ventures': 0.8,
    'Amazon Alexa Fund': 0.75,
    'Samsung NEXT': 0.7,
    'Qualcomm Ventures': 0.7,
    
    # Tier 4: Active but less prestigious
    'Y Combinator': 0.7,  # Accelerator, different model
    'Techstars': 0.65,
    '500 Startups': 0.6,
}


class InvestorExtractor:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def extract_investors_from_kpis(self) -> List[Dict]:
        """Extract investor names from funding KPIs."""
        print("Extracting investors from funding KPIs...")
        
        kpi_file = os.path.join(config.DATA_DIR, "kpi_entities.parquet")
        if not os.path.exists(kpi_file):
            print("⚠️ No KPI data found")
            return []
        
        df_kpi = pd.read_parquet(kpi_file)
        
        # Filter funding KPIs with investors
        df_funding = df_kpi[
            (df_kpi['category'] == 'KPI') &
            (df_kpi['detail_type'] == 'Funding')
        ].copy()
        
        if 'kpi_investors' not in df_funding.columns:
            print("⚠️ No structured investor data found")
            return []
        
        investors_data = []
        
        for _, row in df_funding.iterrows():
            investors_json = row.get('kpi_investors')
            if not investors_json:
                continue
            
            try:
                investors = json.loads(investors_json)
                for investor_name in investors:
                    if investor_name:
                        # Assign prestige score
                        prestige = self._get_investor_prestige(investor_name)
                        
                        investors_data.append({
                            'name': investor_name,
                            'prestige': prestige,
                            'snippet_id': row['snippet_id']
                        })
            except:
                pass
        
        print(f"✅ Extracted {len(investors_data)} investor mentions")
        return investors_data
    
    def _get_investor_prestige(self, investor_name: str) -> float:
        """Get prestige score for investor, with fuzzy matching."""
        # Exact match
        if investor_name in INVESTOR_PRESTIGE:
            return INVESTOR_PRESTIGE[investor_name]
        
        # Fuzzy matching (contains)
        investor_lower = investor_name.lower()
        for known_investor, score in INVESTOR_PRESTIGE.items():
            if known_investor.lower() in investor_lower or investor_lower in known_investor.lower():
                return score
        
        # Unknown investor: default mid-tier score
        return 0.5
    
    def create_investor_nodes(self, investors_data: List[Dict]):
        """Create :Investor nodes in Neo4j."""
        if not investors_data:
            return
        
        print("Creating :Investor nodes in Neo4j...")
        
        # Deduplicate investors
        investors_unique = {}
        for inv in investors_data:
            name = inv['name']
            if name not in investors_unique:
                investors_unique[name] = {
                    'name': name,
                    'prestige': inv['prestige'],
                    'mention_count': 0
                }
            investors_unique[name]['mention_count'] += 1
        
        # Create nodes
        with self.driver.session() as session:
            investors_list = list(investors_unique.values())
            session.run("""
                UNWIND $investors AS inv
                MERGE (i:Investor {name: inv.name})
                SET i.prestige = inv.prestige,
                    i.mention_count = inv.mention_count
            """, {'investors': investors_list})
        
        print(f"✅ Created {len(investors_unique)} Investor nodes")
    
    def link_investors_to_entities(self):
        """Create relationships: Entity-[FUNDED_BY]->Investor."""
        print("Linking investors to entities...")
        
        kpi_file = os.path.join(config.DATA_DIR, "kpi_entities.parquet")
        df_kpi = pd.read_parquet(kpi_file)
        
        # Get snippets file for entity mapping
        snippets_file = os.path.join(config.DATA_DIR, "snippets.parquet")
        if not os.path.exists(snippets_file):
            print("⚠️ Snippets file not found")
            return
        
        df_snippets = pd.read_parquet(snippets_file)
        
        # Get entity extractions
        df_entities = df_kpi[df_kpi['category'] == 'Entity'].copy()
        
        # Get funding KPIs with investors
        df_funding = df_kpi[
            (df_kpi['category'] == 'KPI') &
            (df_kpi['detail_type'] == 'Funding')
        ].copy()
        
        if 'kpi_investors' not in df_funding.columns:
            return
        
        relationships = []
        
        for _, funding_row in df_funding.iterrows():
            snippet_id = funding_row['snippet_id']
            investors_json = funding_row.get('kpi_investors')
            
            if not investors_json:
                continue
            
            try:
                investors = json.loads(investors_json)
            except:
                continue
            
            # Get entities mentioned in same snippet
            entities_in_snippet = df_entities[df_entities['snippet_id'] == snippet_id]['entity_name'].unique()
            
            for entity_name in entities_in_snippet:
                for investor_name in investors:
                    if entity_name and investor_name:
                        relationships.append({
                            'entity': entity_name,
                            'investor': investor_name,
                            'snippet_id': snippet_id
                        })
        
        if relationships:
            with self.driver.session() as session:
                session.run("""
                    UNWIND $rels AS rel
                    MATCH (e:Entity {name: rel.entity})
                    MATCH (i:Investor {name: rel.investor})
                    MERGE (e)-[r:FUNDED_BY]->(i)
                    ON CREATE SET r.mention_count = 1
                    ON MATCH SET r.mention_count = r.mention_count + 1
                """, {'rels': relationships})
            
            print(f"✅ Created {len(relationships)} FUNDED_BY relationships")
    
    def run_full_pipeline(self):
        """Execute complete investor extraction pipeline."""
        print("\n" + "="*80)
        print("INVESTOR EXTRACTION & QUALITY SCORING")
        print("="*80 + "\n")
        
        investors_data = self.extract_investors_from_kpis()
        
        if not investors_data:
            print("⚠️ No investors found in KPI data")
            print("   Make sure to run the updated extract_llm.py first")
            return
        
        self.create_investor_nodes(investors_data)
        self.link_investors_to_entities()
        
        print("\n" + "="*80)
        print("✅ INVESTOR EXTRACTION COMPLETE")
        print("="*80 + "\n")


def main():
    extractor = InvestorExtractor()
    try:
        extractor.run_full_pipeline()
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
