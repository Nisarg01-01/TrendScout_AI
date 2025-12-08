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

def parse_funding_amount(text: str) -> float:
    """Extract funding amount from text like '$10M' or '$2.5B' → number."""
    if not text:
        return 0.0
    
    text = text.upper().replace(',', '')
    
    # Match patterns like $10M, $2.5B, 10 million, etc.
    patterns = [
        r'\$?([0-9.]+)\s*B(?:ILLION)?',  # Billions
        r'\$?([0-9.]+)\s*M(?:ILLION)?',  # Millions
        r'\$?([0-9.]+)\s*K(?:THOUSAND)?', # Thousands
        r'\$([0-9.]+)',                   # Raw dollar amount
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            if 'B' in pattern:
                return num * 1_000_000_000
            elif 'M' in pattern:
                return num * 1_000_000
            elif 'K' in pattern:
                return num * 1_000
            else:
                return num
    
    return 0.0

def parse_hiring_count(text: str) -> int:
    """Extract hiring count from text like '50 engineers' → 50."""
    if not text:
        return 0
    
    # Match patterns like "50 engineers", "hire 20 people", etc.
    patterns = [
        r'(\d+)\s+(?:engineer|developer|employee|people|position|role|hire)',
        r'hire\s+(\d+)',
        r'hiring\s+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    
    return 0

PROMPT_TEMPLATE = """
You are a market intelligence analyst. Analyze the following text snippet and extract structured data.
Return ONLY a valid JSON object with no markdown formatting.

Text: "{text}"

Extract:
1. "entities": List of companies/organizations mentioned.
   - "name": Canonical name.
   - "type": "Startup", "VC", "Big Tech", "Research Lab", etc.
2. "sector": The broad market sector (e.g., "Fintech", "Healthcare", "Enterprise Software").
3. "industry": The specific technology or niche (e.g., "Generative AI", "Robotics").
4. "kpis": List of key performance indicators with structured fields:
   - For Funding:
     * "type": "Funding"
     * "amount": number in dollars (extract from "$10M" → 10000000)
     * "stage": "Seed", "Series A", "Series B", "Series C", "IPO", etc.
     * "investors": list of investor names
     * "value_text": original text
   - For Hiring:
     * "type": "Hiring"
     * "count": number of positions (extract from "50 engineers" → 50)
     * "roles": list of role names
     * "skills": list of required skills
     * "value_text": original text
   - For Partnerships:
     * "type": "Partnership"
     * "partner": partner company name
     * "description": brief description
     * "value_text": original text
   - For Product:
     * "type": "Product"
     * "name": product name
     * "description": brief description
     * "value_text": original text
   - For other metrics (Revenue, Growth, etc.):
     * "type": metric type
     * "value_text": the value as text
5. "swot": List of SWOT elements relevant to the entities.
   - "type": "Strength", "Weakness", "Opportunity", "Threat".
   - "description": Brief description.
6. "stance": Sentiment towards the main entity (-1.0 to 1.0).

JSON Structure:
{{
  "entities": [{{ "name": "...", "type": "..." }}],
  "sector": "...",
  "industry": "...",
  "kpis": [
    {{ "type": "Funding", "amount": 10000000, "stage": "Series A", "investors": ["Sequoia"], "value_text": "..." }},
    {{ "type": "Hiring", "count": 50, "roles": ["Engineer"], "skills": ["Python"], "value_text": "..." }},
    {{ "type": "Partnership", "partner": "Google", "description": "...", "value_text": "..." }},
    {{ "type": "Product", "name": "GPT-5", "description": "...", "value_text": "..." }},
    {{ "type": "Revenue", "value_text": "$1B ARR" }}
  ],
  "swot": [{{ "type": "...", "description": "..." }}],
  "stance": 0.0
}}
"""

def clean_json_response(response_text):
    """Extract JSON content from LLM response, removing markdown wrappers."""
    
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
    """Extract structured data from text using LLM with optimized settings."""
    
    prompt = PROMPT_TEMPLATE.format(text=text)
    
    try:
        # Use format="json" to force JSON mode, lower temperature for consistency
        # num_predict limits output length for speed
        response = ollama.generate(
            model=model, 
            prompt=prompt, 
            format="json", 
            options={
                "temperature": 0.1,
                "num_predict": 1024,  # Limit output length
                "top_p": 0.9,
            }
        )
        raw_response = response['response']
        cleaned_text = clean_json_response(raw_response)
        return json.loads(cleaned_text)
    except Exception as e:
        # Single retry on failure
        try:
            time.sleep(0.3)
            response = ollama.generate(model=model, prompt=prompt, format="json", options={"temperature": 0.1})
            raw_response = response['response']
            cleaned_text = clean_json_response(raw_response)
            return json.loads(cleaned_text)
        except:
            return None

def process_single_row(row):
    """Process article text to extract entities, KPIs, and SWOT data."""
    
    snippet_id = row["snippet_id"]
    text = row["text"]
    local_results = []
    
    data = extract_from_snippet(text, config.LLM_MODEL)
    
    if not data:
        return []

    industry_raw = data.get("industry", "Unknown")
    sector_raw = data.get("sector", "")
    
    # Prefer broader sector classification over specific industry
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
        
    # Add KPIs with structured fields
    for kpi in data.get("kpis", []):
        if isinstance(kpi, str):
            # Fallback for simple string KPIs
            kpi_type = "General"
            kpi_value = kpi
            kpi_data = {"value_text": kpi}
        else:
            kpi_type = kpi.get("type", "General")
            kpi_value = kpi.get("value_text", str(kpi))
            kpi_data = kpi.copy()
        
        # Parse structured fields based on type
        if kpi_type == "Funding":
            amount_raw = kpi_data.get("amount", 0)
            if isinstance(amount_raw, str):
                amount = parse_funding_amount(amount_raw)
            else:
                amount = float(amount_raw) if amount_raw else 0.0
            
            # Store structured funding data
            local_results.append({
                "snippet_id": snippet_id,
                "entity_name": None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Funding",
                "detail_value": kpi_value,
                "stance": stance,
                "confidence": 1.0,
                "kpi_amount": amount,
                "kpi_stage": kpi_data.get("stage", ""),
                "kpi_investors": json.dumps(kpi_data.get("investors", [])),
            })
            
        elif kpi_type == "Hiring":
            count_raw = kpi_data.get("count", 0)
            if isinstance(count_raw, str):
                count = parse_hiring_count(count_raw)
            else:
                count = int(count_raw) if count_raw else 0
            
            local_results.append({
                "snippet_id": snippet_id,
                "entity_name": None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Hiring",
                "detail_value": kpi_value,
                "stance": stance,
                "confidence": 1.0,
                "kpi_count": count,
                "kpi_roles": json.dumps(kpi_data.get("roles", [])),
                "kpi_skills": json.dumps(kpi_data.get("skills", [])),
            })
            
        elif kpi_type == "Partnership":
            local_results.append({
                "snippet_id": snippet_id,
                "entity_name": None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Partnership",
                "detail_value": kpi_value,
                "stance": stance,
                "confidence": 1.0,
                "kpi_partner": kpi_data.get("partner", ""),
                "kpi_description": kpi_data.get("description", ""),
            })
            
        elif kpi_type == "Product":
            local_results.append({
                "snippet_id": snippet_id,
                "entity_name": None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Product",
                "detail_value": kpi_value,
                "stance": stance,
                "confidence": 1.0,
                "kpi_product_name": kpi_data.get("name", ""),
                "kpi_description": kpi_data.get("description", ""),
            })
            
        else:
            # Generic KPI
            local_results.append({
                "snippet_id": snippet_id,
                "entity_name": None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": kpi_type,
                "detail_value": kpi_value,
                "stance": stance,
                "confidence": 1.0,
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
    """Process all snippets to extract structured data, filtering already processed ones."""
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
    # Optimized for RTX 4050: llama3.1 can handle 4 workers
    MAX_WORKERS = 4
    
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
