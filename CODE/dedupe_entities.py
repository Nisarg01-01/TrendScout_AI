import pandas as pd
import os
import config
from rapidfuzz import process, fuzz
from tqdm import tqdm

def normalize_name(name: str) -> str:
    """Basic normalization: lowercase, strip, remove common suffixes."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [" inc", " inc.", " llc", " ltd", " corp", " corporation", " ai"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name.title() # Return Title Case for display

def deduplicate_entities():
    if not os.path.exists(config.KPI_ENTITIES_FILE):
        print("No extracted entities found. Run extract_llm.py first.")
        return

    df = pd.read_parquet(config.KPI_ENTITIES_FILE)
    
    # Filter for just entities
    entities_df = df[df["category"] == "Entity"].copy()
    
    if entities_df.empty:
        print("No entities to deduplicate.")
        return

    # Get unique raw names
    raw_names = entities_df["entity_name"].dropna().unique().tolist()
    print(f"Found {len(raw_names)} unique raw entity names.")

    # Create mapping: Raw Name -> Canonical Name
    # Use greedy clustering approach with fuzzy matching
    # Sort by frequency to use most common version as canonical
    name_counts = entities_df["entity_name"].value_counts()
    sorted_names = name_counts.index.tolist()
    
    canonical_map = {}
    processed_names = set()

    for name in tqdm(sorted_names, desc="Deduplicating"):
        if name in processed_names:
            continue
            
        # This name becomes a new canonical root
        canonical_map[name] = name
        processed_names.add(name)
        
        # Find matches in the remaining list
        # We only look at names that haven't been processed yet
        # Optimization: RapidFuzz process.extract is fast
        
        # Get candidates (all unprocessed names)
        candidates = [n for n in sorted_names if n not in processed_names]
        if not candidates:
            break
            
        matches = process.extract(
            name, 
            candidates, 
            scorer=fuzz.token_sort_ratio, 
            limit=None, 
            score_cutoff=config.FUZZY_THRESHOLD
        )
        
        for match_name, score, _ in matches:
            canonical_map[match_name] = name
            processed_names.add(match_name)

    # Save the map
    map_df = pd.DataFrame(list(canonical_map.items()), columns=["raw_name", "canonical_name"])
    map_df.to_parquet(config.ENTITY_MAP_FILE, index=False)
    
    print(f"Mapped {len(raw_names)} raw names to {map_df['canonical_name'].nunique()} canonical entities.")
    print(f"Saved entity map to {config.ENTITY_MAP_FILE}")

    # Verification Output
    print("\n--- Sample Output (Deduplication) ---")
    print(map_df.head(5).to_string())
    print("-------------------------------------\n")

def main():
    deduplicate_entities()

if __name__ == "__main__":
    main()
