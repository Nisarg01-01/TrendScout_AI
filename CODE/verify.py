#!/usr/bin/env python3
"""
Quick verification script for new TrendScout AI components.
Tests graph schema, data files, and basic functionality.
"""

import os
import sys
import pandas as pd
from utils.neo4j_utils import get_neo4j_driver
import config

def color_print(text, color='green'):
    """Print colored text to terminal."""
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def test_graph_schema():
    """Test Neo4j graph schema for new nodes and relationships."""
    color_print("\n=== Testing Graph Schema ===", 'blue')
    
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Test KPICluster nodes
            result = session.run("MATCH (kc:KPICluster) RETURN count(kc) as count")
            kpi_count = result.single()['count']
            if kpi_count > 0:
                color_print(f"✓ KPICluster nodes: {kpi_count}", 'green')
            else:
                color_print(f"⚠ No KPICluster nodes found", 'yellow')
            
            # Test Investor nodes
            result = session.run("MATCH (i:Investor) RETURN count(i) as count")
            inv_count = result.single()['count']
            if inv_count > 0:
                color_print(f"✓ Investor nodes: {inv_count}", 'green')
            else:
                color_print(f"⚠ No Investor nodes found", 'yellow')
            
            # Test SIMILAR_TO edges
            result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count")
            sim_count = result.single()['count']
            if sim_count > 0:
                color_print(f"✓ SIMILAR_TO relationships: {sim_count}", 'green')
            else:
                color_print(f"⚠ No SIMILAR_TO relationships found", 'yellow')
            
            # Test FUNDED_BY relationships
            result = session.run("MATCH ()-[r:FUNDED_BY]->() RETURN count(r) as count")
            fund_count = result.single()['count']
            if fund_count > 0:
                color_print(f"✓ FUNDED_BY relationships: {fund_count}", 'green')
            else:
                color_print(f"⚠ No FUNDED_BY relationships found", 'yellow')
            
            # Test enhanced RANKED_IN with investor_quality
            result = session.run("""
                MATCH (e:Entity)-[r:RANKED_IN]->(c:Cluster)
                WHERE r.investor_quality IS NOT NULL
                RETURN count(r) as count
            """)
            rank_count = result.single()['count']
            if rank_count > 0:
                color_print(f"✓ RANKED_IN with investor_quality: {rank_count}", 'green')
            else:
                color_print(f"⚠ No RANKED_IN relationships with investor_quality", 'yellow')
            
            # Show sample ranking data
            result = session.run("""
                MATCH (e:Entity)-[r:RANKED_IN]->(c:Cluster)
                WHERE r.investor_quality IS NOT NULL
                RETURN e.name as entity, r.rank as rank, r.score as score,
                       r.centrality as centrality, r.kpi_stance as kpi_stance,
                       r.recency as recency, r.investor_quality as investor_quality
                ORDER BY r.rank ASC
                LIMIT 5
            """)
            
            records = list(result)
            if records:
                color_print("\n📊 Sample Rankings:", 'blue')
                for rec in records:
                    print(f"  #{rec['rank']} {rec['entity']}: Score={rec['score']:.2f} "
                          f"(C={rec['centrality']:.2f}, K={rec['kpi_stance']:.2f}, "
                          f"R={rec['recency']:.2f}, I={rec['investor_quality']:.2f})")
        
        driver.close()
        return True
    
    except Exception as e:
        color_print(f"✗ Graph schema test failed: {e}", 'red')
        return False

def test_temporal_features():
    """Test temporal_features.parquet file."""
    color_print("\n=== Testing Temporal Features ===", 'blue')
    
    file_path = os.path.join(config.DATA_DIR, 'temporal_features.parquet')
    
    if not os.path.exists(file_path):
        color_print(f"⚠ temporal_features.parquet not found at {file_path}", 'yellow')
        return False
    
    try:
        df = pd.read_parquet(file_path)
        color_print(f"✓ Temporal features file loaded: {len(df)} entities", 'green')
        
        # Check expected columns
        expected_cols = ['entity_name', 'funding_30d', 'funding_90d', 'funding_180d', 
                        'hiring_velocity', 'buzz_momentum']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if not missing_cols:
            color_print(f"✓ All expected columns present", 'green')
        else:
            color_print(f"⚠ Missing columns: {missing_cols}", 'yellow')
        
        # Show sample
        color_print("\n📊 Sample Temporal Data:", 'blue')
        print(df.head(3).to_string())
        
        return True
    
    except Exception as e:
        color_print(f"✗ Temporal features test failed: {e}", 'red')
        return False

def test_kpi_extraction():
    """Test enhanced KPI extraction in kpi_entities.parquet."""
    color_print("\n=== Testing KPI Extraction ===", 'blue')
    
    file_path = config.KPI_ENTITIES_FILE
    
    if not os.path.exists(file_path):
        color_print(f"⚠ kpi_entities.parquet not found at {file_path}", 'yellow')
        return False
    
    try:
        df = pd.read_parquet(file_path)
        color_print(f"✓ KPI entities file loaded: {len(df)} KPIs", 'green')
        
        # Check for structured fields
        if 'funding_amount' in df.columns:
            funding_count = df['funding_amount'].notna().sum()
            color_print(f"✓ Structured funding amounts: {funding_count} entries", 'green')
        else:
            color_print(f"⚠ No funding_amount column found", 'yellow')
        
        if 'hiring_count' in df.columns:
            hiring_count = df['hiring_count'].notna().sum()
            color_print(f"✓ Structured hiring counts: {hiring_count} entries", 'green')
        else:
            color_print(f"⚠ No hiring_count column found", 'yellow')
        
        # Show sample funding KPIs
        if 'kpi_type' in df.columns:
            funding_kpis = df[df['kpi_type'] == 'funding'].head(3)
            if not funding_kpis.empty:
                color_print("\n📊 Sample Funding KPIs:", 'blue')
                for _, row in funding_kpis.iterrows():
                    print(f"  Entity: {row.get('entity_name', 'N/A')}")
                    print(f"    Amount: ${row.get('funding_amount', 0):,.0f}")
                    print(f"    Stage: {row.get('funding_stage', 'N/A')}")
                    print(f"    Investors: {row.get('funding_investors', [])[:3]}")
                    print()
        
        return True
    
    except Exception as e:
        color_print(f"✗ KPI extraction test failed: {e}", 'red')
        return False

def test_retrieval_service():
    """Test retrieval service with new features."""
    color_print("\n=== Testing Retrieval Service ===", 'blue')
    
    try:
        from retrieval_service import TrendScoutBackend
        
        backend = TrendScoutBackend()
        color_print("✓ TrendScoutBackend initialized", 'green')
        
        # Test cluster_scoped_search
        if hasattr(backend, 'cluster_scoped_search'):
            color_print("✓ cluster_scoped_search method available", 'green')
        else:
            color_print("⚠ cluster_scoped_search method not found", 'yellow')
        
        # Test graph_store methods
        if hasattr(backend.graph_store, 'get_kpi_breakdown_for_entity'):
            color_print("✓ get_kpi_breakdown_for_entity method available", 'green')
        else:
            color_print("⚠ get_kpi_breakdown_for_entity method not found", 'yellow')
        
        if hasattr(backend.graph_store, 'get_investor_quality_for_entity'):
            color_print("✓ get_investor_quality_for_entity method available", 'green')
        else:
            color_print("⚠ get_investor_quality_for_entity method not found", 'yellow')
        
        if hasattr(backend.graph_store, 'get_entities_in_cluster'):
            color_print("✓ get_entities_in_cluster method available", 'green')
        else:
            color_print("⚠ get_entities_in_cluster method not found", 'yellow')
        
        return True
    
    except Exception as e:
        color_print(f"✗ Retrieval service test failed: {e}", 'red')
        return False

def test_module_imports():
    """Test that new modules can be imported."""
    color_print("\n=== Testing Module Imports ===", 'blue')
    
    modules = [
        'kpi_clustering',
        'temporal_features',
        'investor_extraction'
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            color_print(f"✓ {module_name}.py imports successfully", 'green')
        except Exception as e:
            color_print(f"✗ {module_name}.py import failed: {e}", 'red')
            all_ok = False
    
    return all_ok

def main():
    """Run all verification tests."""
    color_print("\n" + "="*60, 'blue')
    color_print("  TrendScout AI - Component Verification", 'blue')
    color_print("="*60, 'blue')
    
    results = {
        "Module Imports": test_module_imports(),
        "Graph Schema": test_graph_schema(),
        "Temporal Features": test_temporal_features(),
        "KPI Extraction": test_kpi_extraction(),
        "Retrieval Service": test_retrieval_service()
    }
    
    # Summary
    color_print("\n" + "="*60, 'blue')
    color_print("  Verification Summary", 'blue')
    color_print("="*60, 'blue')
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_status in results.items():
        status_color = 'green' if passed_status else 'red'
        status_text = 'PASS' if passed_status else 'FAIL'
        color_print(f"{test_name}: {status_text}", status_color)
    
    color_print(f"\nTotal: {passed}/{total} tests passed", 
                'green' if passed == total else 'yellow')
    
    if passed == total:
        color_print("\n✅ All components verified successfully!", 'green')
        return 0
    elif passed > 0:
        color_print("\n⚠ Some components need attention. Check warnings above.", 'yellow')
        color_print("   This may be normal if pipeline hasn't been run yet.", 'yellow')
        return 0
    else:
        color_print("\n✗ Verification failed. Please check errors above.", 'red')
        return 1

if __name__ == "__main__":
    sys.exit(main())
