"""
Temporal Features Module
Implements rolling window calculations for:
- Funding cadence (30/90/180 days)
- Hiring velocity (30/90/180 days)
- Buzz momentum (article/mention counts)
- Source diversity
"""

import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import config
from typing import Dict, List
import json

class TemporalFeatureEngine:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.windows = [30, 90, 180]  # days
        
    def close(self):
        self.driver.close()
    
    def get_all_entities(self) -> List[str]:
        """Get list of all entities in the graph."""
        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name as name")
            return [record['name'] for record in result]
    
    def calculate_funding_cadence(self, entity_name: str, window_days: int) -> Dict:
        """
        Calculate funding metrics in given time window.
        Returns: count, total_amount, average_amount, stages
        """
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=window_days)
        
        with self.driver.session() as session:
            # Get snippet IDs related to the entity
            result = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(e:Entity {name: $entity_name})
                MATCH (a)<-[:IN]-(s:Snippet)
                RETURN s.id as snippet_id
            """, {'entity_name': entity_name})
            
            # Get a set of relevant snippet IDs for this entity
            relevant_snippet_ids = {record['snippet_id'] for record in result}
        
        # Load structured KPI data from parquet
        kpi_file = os.path.join(config.DATA_DIR, "kpi_entities.parquet")
        if not os.path.exists(kpi_file):
            return {
                'count': 0,
                'total_amount': 0.0,
                'avg_amount': 0.0,
                'stages': []
            }
        
        df_kpi = pd.read_parquet(kpi_file)
        
        # Filter funding events in window
        df_funding = df_kpi[
            (df_kpi['category'] == 'KPI') &
            (df_kpi['detail_type'] == 'Funding') &
            # CRITICAL FIX: Filter by snippets that mention the entity
            (df_kpi['snippet_id'].isin(relevant_snippet_ids))
        ].copy()
        
        if 'kpi_amount' in df_funding.columns:
            # Convert date strings to datetime objects for filtering
            df_funding['date'] = pd.to_datetime(
                df_funding['detail_value'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0],
                errors='coerce',
                utc=True
            )
            funding_in_window = df_funding[df_funding['date'] >= cutoff]
            funding_in_window = funding_in_window[funding_in_window['kpi_amount'] > 0]
            
            count = len(funding_in_window)
            total_amount = funding_in_window['kpi_amount'].sum()
            avg_amount = funding_in_window['kpi_amount'].mean() if count > 0 else 0.0
            
            stages = []
            if 'kpi_stage' in funding_in_window.columns:
                stages = funding_in_window['kpi_stage'].dropna().unique().tolist()
            
            return {
                'count': int(count),
                'total_amount': float(total_amount),
                'avg_amount': float(avg_amount),
                'stages': stages
            }
        
        return {
            'count': 0,
            'total_amount': 0.0,
            'avg_amount': 0.0,
            'stages': []
        }
    
    def calculate_hiring_velocity(self, entity_name: str, window_days: int) -> Dict:
        """
        Calculate hiring metrics in given time window.
        Returns: total_count, unique_roles, unique_skills
        """
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=window_days)
        
        # Get snippet IDs related to the entity from Neo4j
        relevant_snippet_ids = set()
        with self.driver.session() as session:
            # Get snippet IDs related to the entity
            result = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(e:Entity {name: $entity_name})
                MATCH (a)<-[:IN]-(s:Snippet)
                RETURN s.id as snippet_id
            """, {'entity_name': entity_name})
            
            # Get a set of relevant snippet IDs for this entity
            relevant_snippet_ids = {record['snippet_id'] for record in result}
        
        # Load structured KPI data
        kpi_file = os.path.join(config.DATA_DIR, "kpi_entities.parquet")
        if not os.path.exists(kpi_file):
            return {
                'total_count': 0,
                'unique_roles': [],
                'unique_skills': []
            }
        
        df_kpi = pd.read_parquet(kpi_file)
        
        # Filter hiring events
        df_hiring = df_kpi[
            (df_kpi['category'] == 'KPI') &
            (df_kpi['detail_type'] == 'Hiring') &
            # CRITICAL FIX: Filter by snippets that mention the entity
            (df_kpi['snippet_id'].isin(relevant_snippet_ids))
        ].copy()
        
        if 'kpi_count' not in df_hiring.columns:
            return {
                'total_count': 0,
                'unique_roles': [],
                'unique_skills': [],
                'velocity_per_month': 0.0,
            }
        
        # Filter by date window
        df_hiring['date'] = pd.to_datetime(
            df_hiring['detail_value'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0],
            errors='coerce',
            utc=True
        )
        df_hiring = df_hiring[df_hiring['date'] >= cutoff]

        total_count = int(df_hiring['kpi_count'].sum())
        
        # Extract roles and skills
        unique_roles = set()
        unique_skills = set()
        
        if 'kpi_roles' in df_hiring.columns:
            for roles_json in df_hiring['kpi_roles'].dropna():
                try:
                    roles = json.loads(roles_json)
                    unique_roles.update(roles)
                except:
                    pass
        
        if 'kpi_skills' in df_hiring.columns:
            for skills_json in df_hiring['kpi_skills'].dropna():
                try:
                    skills = json.loads(skills_json)
                    unique_skills.update(skills)
                except:
                    pass
        
        return {
            'total_count': total_count,
            'unique_roles': list(unique_roles),
            'unique_skills': list(unique_skills),
            'velocity_per_month': total_count / (window_days / 30.0) if window_days > 0 else 0
        }
    
    def calculate_buzz_momentum(self, entity_name: str, window_days: int) -> Dict:
        """
        Calculate article/mention counts and source diversity.
        Returns: article_count, mention_count, unique_sources
        """
        cutoff = datetime.now() - timedelta(days=window_days)
        cutoff_str = cutoff.isoformat()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(e:Entity {name: $entity_name})
                WHERE a.published >= $cutoff
                RETURN count(DISTINCT a) as article_count,
                       count(*) as mention_count,
                       collect(DISTINCT a.source) as sources
            """, {'entity_name': entity_name, 'cutoff': cutoff_str})
            
            record = result.single()
            if record:
                return {
                    'article_count': record['article_count'],
                    'mention_count': record['mention_count'],
                    'unique_sources': len(record['sources']),
                    'source_list': record['sources'],
                    'momentum_per_month': record['article_count'] / (window_days / 30.0) if window_days > 0 else 0
                }
        
        return {
            'article_count': 0,
            'mention_count': 0,
            'unique_sources': 0,
            'source_list': [],
            'momentum_per_month': 0
        }
    
    def calculate_kpi_polarity_delta(self, entity_name: str) -> Dict:
        """
        Calculate KPI polarity changes: last 30 days vs previous 30 days.
        Returns: current_polarity, previous_polarity, delta
        """
        now = datetime.now()
        window1_start = now - timedelta(days=30)
        window2_start = now - timedelta(days=60)
        window2_end = window1_start
        
        with self.driver.session() as session:
            # Current 30 days
            result1 = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(e:Entity {name: $entity_name})
                MATCH (a)<-[:IN]-(s:Snippet)-[:ABOUT]->(k:KPI)
                WHERE k.date >= $start1
                RETURN avg(k.stance) as avg_stance
            """, {
                'entity_name': entity_name,
                'start1': window1_start.isoformat()
            })
            
            current_polarity = result1.single()['avg_stance'] or 0.0
            
            # Previous 30 days
            result2 = session.run("""
                MATCH (a:Article)-[:MENTIONS]->(e:Entity {name: $entity_name})
                MATCH (a)<-[:IN]-(s:Snippet)-[:ABOUT]->(k:KPI)
                WHERE k.date >= $start2 AND k.date < $end2
                RETURN avg(k.stance) as avg_stance
            """, {
                'entity_name': entity_name,
                'start2': window2_start.isoformat(),
                'end2': window2_end.isoformat()
            })
            
            previous_polarity = result2.single()['avg_stance'] or 0.0
        
        return {
            'current_polarity_30d': float(current_polarity),
            'previous_polarity_30d': float(previous_polarity),
            'polarity_delta': float(current_polarity - previous_polarity)
        }
    
    def compute_features_for_entity(self, entity_name: str) -> Dict:
        """Compute all temporal features for one entity."""
        features = {
            'entity_name': entity_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Funding cadence for each window
        for window in self.windows:
            funding = self.calculate_funding_cadence(entity_name, window)
            features[f'funding_count_{window}d'] = funding['count']
            features[f'funding_total_{window}d'] = funding['total_amount']
            features[f'funding_avg_{window}d'] = funding['avg_amount']
            features[f'funding_stages_{window}d'] = json.dumps(funding['stages'])
        
        # Hiring velocity for each window
        for window in self.windows:
            hiring = self.calculate_hiring_velocity(entity_name, window)
            features[f'hiring_count_{window}d'] = hiring['total_count']
            features[f'hiring_velocity_{window}d'] = hiring['velocity_per_month']
            features[f'hiring_roles_{window}d'] = json.dumps(hiring['unique_roles'])
            features[f'hiring_skills_{window}d'] = json.dumps(hiring['unique_skills'])
        
        # Buzz momentum for each window
        for window in self.windows:
            buzz = self.calculate_buzz_momentum(entity_name, window)
            features[f'article_count_{window}d'] = buzz['article_count']
            features[f'mention_count_{window}d'] = buzz['mention_count']
            features[f'source_diversity_{window}d'] = buzz['unique_sources']
            features[f'buzz_momentum_{window}d'] = buzz['momentum_per_month']
        
        # KPI polarity delta
        polarity = self.calculate_kpi_polarity_delta(entity_name)
        features['kpi_polarity_current'] = polarity['current_polarity_30d']
        features['kpi_polarity_previous'] = polarity['previous_polarity_30d']
        features['kpi_polarity_delta'] = polarity['polarity_delta']
        
        return features
    
    def compute_all_features(self) -> pd.DataFrame:
        """Compute temporal features for all entities."""
        print("\n" + "="*80)
        print("COMPUTING TEMPORAL FEATURES")
        print("="*80 + "\n")
        
        entities = self.get_all_entities()
        print(f"Computing features for {len(entities)} entities...")
        
        all_features = []
        for i, entity in enumerate(entities):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(entities)} entities...")
            
            features = self.compute_features_for_entity(entity)
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        
        # Export to parquet
        out_path = os.path.join(config.DATA_DIR, "temporal_features.parquet")
        df.to_parquet(out_path, index=False)
        
        print(f"\n[OK] Computed {len(df)} feature rows")
        print(f"[OK] Saved to {out_path}")
        
        # Show sample
        print("\n--- Sample Features ---")
        sample_cols = ['entity_name', 'funding_count_30d', 'hiring_count_30d', 
                       'article_count_30d', 'buzz_momentum_30d', 'kpi_polarity_delta']
        available_cols = [c for c in sample_cols if c in df.columns]
        print(df[available_cols].head(5).to_string(index=False))
        print("-" * 80 + "\n")
        
        return df


def main():
    engine = TemporalFeatureEngine()
    try:
        df_features = engine.compute_all_features()
        print(f"[OK] Temporal feature extraction complete: {len(df_features)} entities")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
