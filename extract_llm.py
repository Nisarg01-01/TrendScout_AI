import pandas as pd
import ollama
import json
import os
import config
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Output File
KPI_ENTITIES_FILE = os.path.join(config.DATA_DIR, "kpi_entities.parquet")

PROMPT_TEMPLATE = """
You are a market intelligence analyst. Analyze the following text snippet and extract structured data.
Return ONLY a valid JSON object with no markdown formatting.

Text: "{text}"

Extract:
1. "entities": List of companies/organizations mentioned.
   - "name": Canonical name.
   - "type": "Startup", "VC", "Big Tech", "Research Lab", etc.
2. "sector": The broad market sector (e.g., "Fintech", "Healthcare", "Enterprise Software", "E-commerce").
   - Example: If JP Morgan launches an AI agent, the sector is "Fintech".
   - Example: If a hospital uses AI, the sector is "Healthcare".
3. "industry": The specific technology or niche (e.g., "Generative AI", "Robotics", "Cybersecurity").
4. "kpis": List of key performance indicators or metrics.
   - "name": e.g., "Funding", "Revenue", "Growth", "Valuation".
   - "value": e.g., "$10M", "50%", "Series A".
5. "swot": List of SWOT elements relevant to the entities.
   - "type": "Strength", "Weakness", "Opportunity", "Threat".
   - "description": Brief description.
6. "stance": Sentiment towards the main entity (-1.0 to 1.0).

JSON Structure:
{{
  "entities": [{{ "name": "...", "type": "..." }}],
  "sector": "...",
  "industry": "...",
  "kpis": [{{ "name": "...", "value": "..." }}],
  "swot": [{{ "type": "...", "description": "..." }}],
  "stance": 0.0
}}
"""

def clean_json_response(response_text):
    """
    Sometimes the LLM wraps the JSON in markdown or adds extra text.
    I'm going to strip all that away and just get the raw JSON object.
    """
    # Try to find JSON block with regex
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        return match.group(0)
    
    # Fallback to markdown stripping
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    return response_text.strip()

def extract_from_snippet(text: str, model: str):
    """
    I'm sending the text to Ollama to get the structured data we need.
    If it fails or gives me bad JSON, I'll give it a couple more tries.
    """
    prompt = PROMPT_TEMPLATE.format(text=text)
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Use format="json" to force JSON mode in Ollama
            response = ollama.generate(model=model, prompt=prompt, format="json", options={"temperature": 0.1})
            raw_response = response['response']
            cleaned_text = clean_json_response(raw_response)
            return json.loads(cleaned_text)
        except Exception as e:
            if attempt == max_retries - 1:
                # print(f"Failed to extract JSON after {max_retries} attempts: {e}")
                return None
            time.sleep(0.5)
    return None

def process_single_row(row):
    """
    Here's where the magic happens for each article.
    I take the text, get the extraction, and then break it down into rows for our dataframe.
    I'll separate out the Entities, KPIs, and SWOT points.
    """
    snippet_id = row["snippet_id"]
    text = row["text"]
    local_results = []
    
    data = extract_from_snippet(text, config.LLM_MODEL)
    
    if not data:
        return []

    industry_raw = data.get("industry", "Unknown")
    sector_raw = data.get("sector", "")
    
    # I want to make sure we have a good high-level category for the industry.
    # If the LLM gave us a broad 'Sector', I'll use that because it's usually cleaner (e.g., 'Fintech' vs 'AI for Banks').
    if sector_raw and sector_raw != "Unknown":
        industry = sector_raw
    else:
        industry = industry_raw

    stance = data.get("stance", 0.0)
    
    # Add Entities
    for ent in data.get("entities", []):
        if isinstance(ent, str):
            ent_name = ent
            ent_type = "Unknown"
        else:
            ent_name = ent.get("name")
            ent_type = ent.get("type")

        local_results.append({
            "snippet_id": snippet_id,
            "entity_name": ent_name,
            "entity_type": ent_type,
            "industry": industry,
            "category": "Entity",
            "detail_type": None,
            "detail_value": None,
            "stance": stance,
            "confidence": 1.0
        })
        
    # Add KPIs
    for kpi in data.get("kpis", []):
        if isinstance(kpi, str):
            kpi_name = "General"
            kpi_value = kpi
        else:
            kpi_name = kpi.get("name")
            kpi_value = kpi.get("value")

        local_results.append({
            "snippet_id": snippet_id,
            "entity_name": None,
            "entity_type": None,
            "industry": industry,
            "category": "KPI",
            "detail_type": kpi_name,
            "detail_value": kpi_value,
            "stance": stance,
            "confidence": 1.0
        })

    # Add SWOT
    for swot in data.get("swot", []):
        if isinstance(swot, str):
            swot_type = "General"
            swot_desc = swot
        else:
            swot_type = swot.get("type", "General")
            swot_desc = swot.get("description", "")

        local_results.append({
            "snippet_id": snippet_id,
            "entity_name": None,
            "entity_type": None,
            "industry": industry,
            "category": "SWOT",
            "detail_type": swot_type,
            "detail_value": swot_desc,
            "stance": stance,
            "confidence": 1.0
        })
        
    return local_results

def process_snippets():
    """
    Main loop to go through all our news snippets.
    I'll check which ones we haven't processed yet so we don't waste time re-doing work.
    """
    if not os.path.exists(config.SNIPPETS_FILE):
        print("No snippets file found. Run preprocess.py first.")
        return

    df = pd.read_parquet(config.SNIPPETS_FILE)
    print(f"Loaded {len(df)} snippets.")

    # Incremental Processing Logic
    existing_ids = set()
    if os.path.exists(KPI_ENTITIES_FILE):
        try:
            existing_df = pd.read_parquet(KPI_ENTITIES_FILE)
            if 'snippet_id' in existing_df.columns:
                existing_ids = set(existing_df['snippet_id'].unique())
                print(f"Found {len(existing_ids)} already processed snippets.")
        except Exception as e:
            print(f"Could not read existing file: {e}. Starting fresh.")

    # Filter for new snippets
    df_new = df[~df['snippet_id'].isin(existing_ids)]
    
    if df_new.empty:
        print("All snippets have already been processed.")
        return

    print(f"Processing {len(df_new)} new snippets...")

    results = []
    
    # Check if Ollama is reachable
    try:
        ollama.list()
    except Exception:
        print("Error: Ollama is not running. Please install and start Ollama.")
        return

    # Parallel Processing
    # RTX 4050 6GB: Llama 3.2 (4 workers), Llama 3.1 8B (2-3 workers)
    MAX_WORKERS = 3
    
    print(f"Starting extraction with {MAX_WORKERS} parallel workers (Model: {config.LLM_MODEL})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_row, row) for _, row in df_new.iterrows()]
        
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Intelligence"):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                # print(f"Task failed: {e}")
                pass

    if not results:
        print("No intelligence extracted.")
        return

    new_results_df = pd.DataFrame(results)
    
    # Append to existing if available
    if os.path.exists(KPI_ENTITIES_FILE) and not existing_ids:
         # If file exists but we couldn't read IDs (edge case), overwrite or append? 
         # Let's just overwrite if we started fresh, or append if we filtered.
         # Actually, if existing_ids is empty but file exists, it might be corrupt or empty.
         pass

    if os.path.exists(KPI_ENTITIES_FILE):
        try:
            existing_df = pd.read_parquet(KPI_ENTITIES_FILE)
            final_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        except:
            final_df = new_results_df
    else:
        final_df = new_results_df

    final_df.to_parquet(KPI_ENTITIES_FILE, index=False)
    print(f"Saved {len(final_df)} extraction records to {KPI_ENTITIES_FILE} ({len(new_results_df)} new)")

    # Verification Output
    print("\n--- Sample Output (Extraction) ---")
    cols_to_show = [c for c in ['snippet_id', 'entity_name', 'industry', 'category'] if c in final_df.columns]
    print(final_df[cols_to_show].tail(3).to_string())
    print("----------------------------------\n")

def main():
    process_snippets()

if __name__ == "__main__":
    main()
