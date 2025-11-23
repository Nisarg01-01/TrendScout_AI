import pandas as pd
import os
import config

def verify_parquet(name, path, columns_to_show=None, n=3):
    print(f"\n{'='*20} {name} {'='*20}")
    if not os.path.exists(path):
        print(f"FILE NOT FOUND: {path}")
        return

    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if not df.empty:
        print(f"\n--- First {n} rows ---")
        if columns_to_show:
            # Filter columns that actually exist
            cols = [c for c in columns_to_show if c in df.columns]
            print(df[cols].head(n).to_string())
        else:
            print(df.head(n).to_string())
            
        # Specific checks
        if 'swot' in df.columns:
            print("\n--- SWOT Sample ---")
            print(df['swot'].head(n).values)
            
        if 'entities' in df.columns:
            print("\n--- Entities Sample ---")
            print(df['entities'].head(n).values)

def main():
    # 1. Ingestion
    verify_parquet("Articles Raw", config.ARTICLES_FILE, ['source', 'title', 'published'])
    
    # 2. Preprocessing
    verify_parquet("Snippets", config.SNIPPETS_FILE, ['snippet_id', 'text'])
    
    # 3. Extraction
    verify_parquet("KPI & Entities", config.KPI_ENTITIES_FILE, ['chunk_id', 'industry', 'sentiment', 'swot', 'entities'])
    
    # 4. Deduplication
    verify_parquet("Entity Map", config.ENTITY_MAP_FILE, ['raw_name', 'canonical_name'])

if __name__ == "__main__":
    main()
