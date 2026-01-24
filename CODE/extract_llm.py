import pandas as pd
import json
import os
import config
import shutil
from datetime import datetime, timezone
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse
from typing import Any
import hashlib
from typing import Dict, List, Optional

try:
    from jsonschema import validate as jsonschema_validate  # type: ignore
except Exception:  # pragma: no cover
    jsonschema_validate = None

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None

# Output File
KPI_ENTITIES_FILE = os.path.join(config.DATA_DIR, "kpi_entities.parquet")

GENERIC_ENTITY_NAME_STOPLIST = {
    "ai",
    "artificial intelligence",
    "generative ai",
    "genai",
    "ai systems",
    "systems",
    "business",
    "businesses",
    "company",
    "companies",
    "the company",
    "your company",
    "social media company",
    "tech company",
    "tech companies",
    "data center",
    "data centers",
    "startup",
    "startups",
    "investor",
    "investors",
    "unknown",
}


def _backup_file_copy(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = f"{path}.bak_{stamp}"
    shutil.copy2(path, backup_path)
    return backup_path

def parse_funding_amount(text: str) -> float:
    """Extract funding amount from text like '$10M' or '$2.5B' -> number."""
    if not text:
        return 0.0
    
    t = text.upper().replace(",", "")

    # $10M / 2.5B / 750K
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(B|M|K)\b", t)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        if unit == "B":
            return num * 1_000_000_000
        if unit == "M":
            return num * 1_000_000
        if unit == "K":
            return num * 1_000

    # "10 million" / "2.5 billion"
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(BILLION|MILLION|THOUSAND)\b", t)
    if m2:
        num = float(m2.group(1))
        unit = m2.group(2)
        if unit == "BILLION":
            return num * 1_000_000_000
        if unit == "MILLION":
            return num * 1_000_000
        if unit == "THOUSAND":
            return num * 1_000

    # Raw $5000000
    m3 = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)\b", t)
    if m3:
        return float(m3.group(1))

    return 0.0

def parse_hiring_count(text: str) -> int:
    """Extract hiring count from text like '50 engineers' -> 50."""
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
You are a market intelligence analyst.
Analyze the following text snippet and extract structured, evidence-based signals.
Return ONLY a valid JSON object with no markdown formatting.
If a field is not explicitly supported by the text, leave it empty or null.
Do NOT guess funding, hiring counts, investors, or acquisitions.

Text: "{text}"

Extract:
1) "primary_entity": The single company/org the snippet is mainly about (or null if unclear).
2) "entities": List of companies/organizations explicitly mentioned.
   - "name": string
   - "type": "Startup" | "Investor" | "Big Tech" | "Government" | "Research Lab" | "Media" | "Other" | "Unknown"
3) "sector": broad sector (e.g., "Fintech", "Healthcare", "Enterprise Software") or "Unknown"
4) "industry": specific niche (e.g., "Generative AI", "Robotics") or "Unknown"
5) "kpis": List of explicit events/claims. Each item MUST include an evidence excerpt that appears verbatim in the snippet.
   - "type": One of:
     "Funding" | "Acquisition" | "Partnership" | "Product" | "Hiring" | "Layoffs" | "Regulation" | "Lawsuit" | "Security" | "Outage" | "Pricing" | "Policy" | "Competition" | "Other"
   - "entity": The entity this KPI is about (usually the primary_entity)
   - "value_text": A short evidence excerpt copied from the snippet (must be a substring)
   - Optional structured fields by type:
     Funding: "amount" (string or number), "stage" (string), "investors" (list of strings)
     Hiring/Layoffs: "count" (string or number), "roles" (list), "skills" (list)
     Partnership: "partner" (string), "description" (string)
     Product: "name" (string), "description" (string)
     Acquisition: "target" (string), "description" (string)
     Regulation/Policy: "description" (string)
     Competition: "competitor" (string), "description" (string)
   - "polarity": -1 | 0 | 1 (good/bad/neutral for the entity)
   - "confidence": number between 0 and 1 (how explicit the claim is)
6) "swot": List of SWOT elements (only if explicitly stated or strongly implied).
   - "type": "Strength" | "Weakness" | "Opportunity" | "Threat"
   - "description": string
   - "entity": string (defaults to primary_entity)
7) "stance": Overall sentiment toward primary_entity (-1.0 to 1.0). If primary_entity is null, use 0.0.

JSON Structure:
{{
  "primary_entity": "...",
  "entities": [{{ "name": "...", "type": "..." }}],
  "sector": "...",
  "industry": "...",
  "kpis": [
    {{ "type": "Funding", "entity": "X", "amount": 10000000, "stage": "Series A", "investors": ["Sequoia"], "value_text": "raised $10M", "polarity": 1, "confidence": 0.9 }},
    {{ "type": "Hiring", "entity": "X", "count": 50, "roles": ["Engineer"], "skills": ["Python"], "value_text": "hiring 50 engineers", "polarity": 1, "confidence": 0.8 }}
  ],
  "swot": [{{ "type": "...", "description": "...", "entity": "X" }}],
  "stance": 0.0
}}
"""

# A more compact prompt that significantly reduces input tokens per call (faster),
# while relying on downstream evidence guardrails for safety.
COMPACT_PROMPT_TEMPLATE = """
Extract structured, evidence-based signals from the snippet below.
Return ONLY valid JSON (no markdown). Do not guess. If unsure, output empty lists and stance=0.

Snippet:
\"\"\"{text}\"\"\"

Return JSON with keys:
{{
  \"primary_entity\": string|null,
  \"entities\": [{{\"name\": string, \"type\": string}}],
  \"sector\": string|null,
  \"industry\": string|null,
  \"kpis\": [
    {{
      \"type\": string,
      \"entity\": string|null,
      \"value_text\": string,
      \"polarity\": -1|0|1,
      \"confidence\": number
    }}
  ],
  \"swot\": [{{\"type\": string, \"description\": string, \"entity\": string|null}}],
  \"stance\": number
}}
"""

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "primary_entity": {"type": ["string", "null"]},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "type": {"type": ["string", "null"]},
                },
            },
        },
        "sector": {"type": ["string", "null"]},
        "industry": {"type": ["string", "null"]},
        "kpis": {
            "type": "array",
            "items": {"type": ["object", "string"]},
        },
        "swot": {
            "type": "array",
            "items": {"type": ["object", "string"]},
        },
        "stance": {"type": ["number", "string", "null"]},
    },
    "required": ["entities", "kpis", "swot", "stance"],
}


def _validate_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if jsonschema_validate is None:
        return True
    try:
        jsonschema_validate(instance=payload, schema=EXTRACTION_SCHEMA)
        return True
    except Exception:
        return False


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _norm_entity_name(name: Any) -> str:
    s = _as_str(name)
    # Trim trivial quotes/backticks.
    s = s.strip("`'\"")
    return s.strip()

def _is_junk_entity_name(name: str) -> bool:
    n = _as_str(name).strip()
    if not n:
        return True
    low = n.lower().strip()
    if low in GENERIC_ENTITY_NAME_STOPLIST:
        return True
    # Very short tokens are almost always noise (e.g., "AI", "US").
    if len(n) <= 2:
        return True
    # Generic phrases that show up as placeholders in snippets.
    if low.endswith(" company") or low.endswith(" companies"):
        return True
    # Purely numeric/punctuation strings.
    if not re.search(r"[a-zA-Z]", n):
        return True
    return False


def _pick_primary_entity(data: Dict[str, Any]) -> str:
    pe = _norm_entity_name(data.get("primary_entity"))
    if pe:
        return pe

    entities = data.get("entities") or []
    names: List[str] = []
    for ent in entities:
        if isinstance(ent, str):
            n = _norm_entity_name(ent)
        elif isinstance(ent, dict):
            n = _norm_entity_name(ent.get("name"))
        else:
            n = ""
        if n:
            names.append(n)

    if len(names) == 1:
        return names[0]

    return ""


def _value_text_supported(snippet_text: str, value_text: str) -> bool:
    vt = _as_str(value_text)
    if not vt:
        return False
    return vt.lower() in _as_str(snippet_text).lower()


def _kpi_has_evidence(snippet_text: str, kpi_type: str, value_text: str) -> bool:
    t = _as_str(snippet_text).lower()
    vt = _as_str(value_text).lower()

    # Require that the LLM-provided excerpt actually exists in the snippet.
    if vt and vt not in t:
        return False

    # Type-specific guardrails to reduce false positives on numbers/dates.
    if kpi_type == "Funding":
        has_funding_kw = any(
            kw in t
            for kw in [
                "raised",
                "raise",
                "funding",
                "round",
                "seed",
                "series ",
                "valuation",
                "invested",
                "backed",
                "financing",
            ]
        )
        has_amount_hint = bool(re.search(r"\$\s*[0-9]", snippet_text)) or ("million" in t) or ("billion" in t)
        return has_funding_kw and has_amount_hint

    if kpi_type == "Hiring":
        return any(kw in t for kw in ["hiring", "hire", "recruit", "headcount", "engineer", "role", "positions"])

    if kpi_type == "Layoffs":
        return any(kw in t for kw in ["layoff", "laid off", "job cuts", "cuts", "furlough"])

    if kpi_type == "Acquisition":
        return any(kw in t for kw in ["acquired", "acquisition", "bought", "buys", "purchase", "purchased"])

    if kpi_type == "Partnership":
        return any(kw in t for kw in ["partner", "partnership", "collaborat", "teamed up", "joined forces"])

    if kpi_type == "Product":
        return any(kw in t for kw in ["launch", "launched", "release", "released", "unveil", "announced", "rollout"])

    if kpi_type == "Regulation":
        return any(kw in t for kw in ["regulat", "ban", "policy", "compliance", "law", "government"])

    if kpi_type == "Lawsuit":
        return any(kw in t for kw in ["lawsuit", "sued", "litigation", "complaint", "court"])

    if kpi_type == "Security":
        return any(kw in t for kw in ["breach", "hack", "leak", "security", "vulnerab"])

    if kpi_type == "Outage":
        return any(kw in t for kw in ["outage", "downtime", "incident"])

    if kpi_type == "Pricing":
        return any(kw in t for kw in ["price", "pricing", "cost", "fees"])

    if kpi_type == "Policy":
        return any(kw in t for kw in ["policy", "policies", "rule", "rules", "guideline"])

    if kpi_type == "Competition":
        return any(kw in t for kw in ["compete", "competitor", "rival", "ahead", "behind", "vs"])

    return True

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

def extract_from_snippet(text: str, model: str, num_predict: int = 512, prompt_style: str = "full"):
    """Extract structured data from text using LLM with optimized settings."""
    
    tpl = PROMPT_TEMPLATE if (prompt_style or "full") == "full" else COMPACT_PROMPT_TEMPLATE
    prompt = tpl.format(text=text)
    
    try:
        # Use format="json" to force JSON mode, lower temperature for consistency
        # num_predict limits output length for speed
        if ollama is None:
            return None

        response = ollama.generate(
            model=model, 
            prompt=prompt, 
            format="json", 
            options={
                "temperature": 0.1,
                "num_predict": int(num_predict),  # Limit output length
                "top_p": 0.9,
            }
        )
        raw_response = response['response']
        cleaned_text = clean_json_response(raw_response)
        payload = json.loads(cleaned_text)
        if not _validate_payload(payload):
            return None
        return payload
    except Exception as e:
        # Single retry on failure
        try:
            time.sleep(0.3)
            if ollama is None:
                return None
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.1, "num_predict": int(num_predict)},
            )
            raw_response = response['response']
            cleaned_text = clean_json_response(raw_response)
            payload = json.loads(cleaned_text)
            if not _validate_payload(payload):
                return None
            return payload
        except:
            return None

def _looks_kpi_relevant(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    # High-recall filter for speed when running in "demo/smoke" mode.
    keywords = [
        "raised",
        "funding",
        "series ",
        "seed round",
        "valuation",
        "$",
        "million",
        "billion",
        "hiring",
        "hire",
        "layoff",
        "partnership",
        "partnered",
        "customer",
        "launched",
        "release",
        "announced",
        "acquired",
        "acquisition",
        "regulation",
        "lawsuit",
        "outage",
        "breach",
    ]
    return any(k in t for k in keywords)


def process_single_row(row, model: str, num_predict: int, prompt_style: str = "full"):
    """Process article text to extract entities, KPIs, and SWOT data."""
    
    snippet_id = row["snippet_id"]
    text = row["text"]
    snippet_text_hash = _as_str(row.get("text_hash"))
    local_results = []
    
    # Be tolerant to older monkeypatches/tests that don't accept the new kwarg.
    try:
        data = extract_from_snippet(text, model, num_predict=num_predict, prompt_style=prompt_style)
    except TypeError:
        data = extract_from_snippet(text, model, num_predict=num_predict)
    
    if not data:
        return [
            {
                "snippet_id": snippet_id,
                "entity_name": None,
                "entity_type": None,
                "industry": "Unknown",
                "category": "Meta",
                "detail_type": "Error",
                "detail_value": "no_json",
                "stance": 0.0,
                "confidence": 0.0,
            }
        ]

    industry_raw = data.get("industry", "Unknown")
    sector_raw = data.get("sector", "")
    
    # Prefer broader sector classification over specific industry
    if sector_raw and sector_raw != "Unknown":
        industry = sector_raw
    else:
        industry = industry_raw

    stance = data.get("stance", 0.0)
    try:
        stance = float(stance) if stance is not None else 0.0
    except Exception:
        stance = 0.0

    primary_entity = _pick_primary_entity(data)
    
    # Add Entities
    for ent in data.get("entities", []):
        if isinstance(ent, str):
            ent_name = _norm_entity_name(ent)
            ent_type = "Unknown"
        else:
            ent_name = _norm_entity_name(ent.get("name"))
            ent_type = _as_str(ent.get("type")) or "Unknown"

        if not ent_name or _is_junk_entity_name(ent_name):
            continue

        local_results.append({
            "snippet_id": snippet_id,
            "snippet_text_hash": snippet_text_hash or None,
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
            kpi_data: Dict[str, Any] = {"value_text": kpi}
        else:
            kpi_type = _as_str(kpi.get("type")) or "Other"
            kpi_value = _as_str(kpi.get("value_text")) or _as_str(kpi)
            kpi_data = dict(kpi)

        subject_entity = _norm_entity_name(kpi_data.get("entity")) or primary_entity

        kpi_polarity = kpi_data.get("polarity", None)
        try:
            kpi_polarity_f = float(kpi_polarity) if kpi_polarity is not None else None
        except Exception:
            kpi_polarity_f = None
        if kpi_polarity_f is not None:
            # Keep KPI polarity bounded.
            if kpi_polarity_f > 1.0:
                kpi_polarity_f = 1.0
            elif kpi_polarity_f < -1.0:
                kpi_polarity_f = -1.0
        kpi_row_stance = kpi_polarity_f if kpi_polarity_f is not None else stance
        confidence = kpi_data.get("confidence", 1.0)
        try:
            confidence = float(confidence) if confidence is not None else 1.0
        except Exception:
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))

        # Evidence guardrail: ensure the excerpt exists and the type has keyword support in the text.
        if not _value_text_supported(text, kpi_value):
            continue
        if not _kpi_has_evidence(text, kpi_type, kpi_value):
            # If it's not supported, drop it rather than polluting downstream graph/ranking.
            continue
        
        # Parse structured fields based on type
        if kpi_type == "Funding":
            amount_raw = kpi_data.get("amount", 0)
            if isinstance(amount_raw, str):
                amount = parse_funding_amount(amount_raw)
            else:
                try:
                    amount = float(amount_raw) if amount_raw else 0.0
                except Exception:
                    amount = 0.0
            
            # Store structured funding data
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Funding",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_amount": amount,
                "kpi_stage": _as_str(kpi_data.get("stage")),
                "kpi_investors": json.dumps(kpi_data.get("investors", []) or []),
            })
            
        elif kpi_type == "Hiring":
            count_raw = kpi_data.get("count", 0)
            if isinstance(count_raw, str):
                count = parse_hiring_count(count_raw)
            else:
                try:
                    count = int(count_raw) if count_raw else 0
                except Exception:
                    count = 0
            
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Hiring",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_count": count,
                "kpi_roles": json.dumps(kpi_data.get("roles", []) or []),
                "kpi_skills": json.dumps(kpi_data.get("skills", []) or []),
            })

        elif kpi_type == "Layoffs":
            count_raw = kpi_data.get("count", 0)
            if isinstance(count_raw, str):
                count = parse_hiring_count(count_raw)
            else:
                try:
                    count = int(count_raw) if count_raw else 0
                except Exception:
                    count = 0

            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Layoffs",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_count": count,
                "kpi_roles": json.dumps(kpi_data.get("roles", []) or []),
                "kpi_skills": json.dumps(kpi_data.get("skills", []) or []),
            })
            
        elif kpi_type == "Partnership":
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Partnership",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_partner": _as_str(kpi_data.get("partner")),
                "kpi_description": _as_str(kpi_data.get("description")),
            })

        elif kpi_type == "Acquisition":
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Acquisition",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_target": _as_str(kpi_data.get("target")),
                "kpi_description": _as_str(kpi_data.get("description")),
            })

        elif kpi_type == "Product":
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Product",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_product_name": _as_str(kpi_data.get("name")),
                "kpi_description": _as_str(kpi_data.get("description")),
            })

        elif kpi_type == "Competition":
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": "Competition",
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_competitor": _as_str(kpi_data.get("competitor")),
                "kpi_description": _as_str(kpi_data.get("description")),
            })

        elif kpi_type in {"Regulation", "Policy", "Lawsuit", "Security", "Outage", "Pricing"}:
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": kpi_type,
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_description": _as_str(kpi_data.get("description")),
            })
            
        else:
            # Generic KPI
            local_results.append({
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": subject_entity or None,
                "entity_type": None,
                "industry": industry,
                "category": "KPI",
                "detail_type": kpi_type,
                "detail_value": kpi_value,
                "stance": kpi_row_stance,
                "confidence": confidence,
                "kpi_description": _as_str(kpi_data.get("description")),
            })

    # Add SWOT
    for swot in data.get("swot", []):
        if isinstance(swot, str):
            swot_type = "General"
            swot_desc = swot
            swot_entity = primary_entity
        else:
            swot_type = _as_str(swot.get("type")) or "General"
            swot_desc = _as_str(swot.get("description"))
            swot_entity = _norm_entity_name(swot.get("entity")) or primary_entity

        if not swot_desc:
            continue

        local_results.append({
            "snippet_id": snippet_id,
            "snippet_text_hash": snippet_text_hash or None,
            "entity_name": swot_entity or None,
            "entity_type": None,
            "industry": industry,
            "category": "SWOT",
            "detail_type": swot_type,
            "detail_value": swot_desc,
            "stance": stance,
            "confidence": 1.0
        })
        
    if not local_results:
        return [
            {
                "snippet_id": snippet_id,
                "snippet_text_hash": snippet_text_hash or None,
                "entity_name": None,
                "entity_type": None,
                "industry": industry,
                "category": "Meta",
                "detail_type": "Empty",
                "detail_value": "",
                "stance": stance,
                "confidence": 1.0,
            }
        ]

    return local_results

def _write_checkpoint(existing_df: pd.DataFrame, new_results_df: pd.DataFrame, out_path: str) -> int:
    if new_results_df.empty and existing_df.empty:
        return 0
    if not existing_df.empty:
        final_df = pd.concat([existing_df, new_results_df], ignore_index=True)
    else:
        final_df = new_results_df
    final_df.to_parquet(out_path, index=False)
    return len(final_df)


def process_snippets(
    model: str,
    out_path: str,
    max_workers: int = 2,
    max_snippets: Optional[int] = None,
    num_predict: int = 512,
    filter_kpi: bool = False,
    save_every: int = 25,
    force: bool = False,
    prompt_style: str = "full",
):
    """Process all snippets to extract structured data, filtering already processed ones."""
    if not os.path.exists(config.SNIPPETS_FILE):
        print("No snippets file found. Run preprocess.py first.")
        return

    df = pd.read_parquet(config.SNIPPETS_FILE)
    print(f"Loaded {len(df)} snippets.")
    if "text_hash" not in df.columns and "text" in df.columns:
        # Backfill for older snippets.parquet files.
        df = df.copy()
        df["text_hash"] = df["text"].astype(str).map(lambda s: hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest())
    current_snippet_ids = set(df["snippet_id"].astype(str)) if "snippet_id" in df.columns else set()

    # Incremental Processing Logic
    existing_ids = set()
    existing_df = pd.DataFrame()
    if os.path.exists(out_path):
        if force:
            bak = _backup_file_copy(out_path)
            if bak:
                print(f"[INFO] --force enabled. Backup written: {bak}")
            existing_ids = set()
            existing_df = pd.DataFrame()
        else:
            try:
                existing_df = pd.read_parquet(out_path)
                if 'snippet_id' in existing_df.columns:
                    existing_ids = set(existing_df['snippet_id'].astype(str).unique())
                    # If we have text hashes, re-process snippet_ids whose content changed.
                    if "snippet_text_hash" in existing_df.columns and "text_hash" in df.columns:
                        cur_hash = dict(zip(df["snippet_id"].astype(str), df["text_hash"].astype(str)))
                        stale_ids = set()
                        for r in existing_df[["snippet_id", "snippet_text_hash"]].dropna().itertuples(index=False):
                            sid = str(r.snippet_id)
                            if sid in cur_hash and str(r.snippet_text_hash) != str(cur_hash[sid]):
                                stale_ids.add(sid)
                        if stale_ids:
                            print(f"[WARN] {len(stale_ids)} snippets changed text since last extraction; re-processing them.")
                            existing_ids = existing_ids - stale_ids
                    stale = bool(current_snippet_ids) and any(sid not in current_snippet_ids for sid in existing_ids)
                    if stale:
                        bak = _backup_file_copy(out_path)
                        if bak:
                            print(f"[WARN] Existing extraction is stale vs current snippets. Backup written: {bak}")
                        existing_ids = set()
                        existing_df = pd.DataFrame()
                        print("[INFO] Starting fresh extraction to avoid mixing old snippet_ids.")
                    else:
                        print(f"Found {len(existing_ids)} already processed snippets.")
            except Exception as e:
                print(f"Could not read existing file: {e}. Starting fresh.")
                existing_df = pd.DataFrame()

    # Filter for new snippets
    df_new = df[~df['snippet_id'].astype(str).isin(existing_ids)]
    if filter_kpi:
        before = len(df_new)
        df_new = df_new[df_new["text"].astype(str).map(_looks_kpi_relevant)]
        print(f"[INFO] KPI filter enabled: {before} -> {len(df_new)} snippets")

    if max_snippets is not None and max_snippets > 0:
        df_new = df_new.head(int(max_snippets))
    
    if df_new.empty:
        print("All snippets have already been processed.")
        return

    print(f"Processing {len(df_new)} new snippets...")

    results = []
    failures = 0
    meta_only = 0
    
    # Check if Ollama is reachable
    if ollama is None:
        print("Error: Python package `ollama` is not installed. Install it in your environment.")
        return

    try:
        ollama.list()
    except Exception:
        print("Error: Ollama is not running. Please install and start Ollama.")
        return

    max_workers = int(max_workers) if max_workers and int(max_workers) > 0 else 1
    num_predict = int(num_predict) if num_predict and int(num_predict) > 0 else 256
    
    print(f"Starting extraction with {max_workers} workers (Model: {model}, num_predict={num_predict})...")
    started = time.time()
    completed = 0
    checkpoint_rows = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_single_row, row, model, num_predict, prompt_style)
            for _, row in df_new.iterrows()
        ]
        
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Intelligence"):
            try:
                batch_results = future.result()
                if (
                    len(batch_results) == 1
                    and isinstance(batch_results[0], dict)
                    and batch_results[0].get("category") == "Meta"
                ):
                    meta_only += 1
                results.extend(batch_results)
                completed += 1
                if save_every and completed % int(save_every) == 0:
                    new_results_df = pd.DataFrame(results)
                    checkpoint_rows = _write_checkpoint(existing_df, new_results_df, out_path)
                    elapsed = max(time.time() - started, 1e-6)
                    rate = completed / elapsed
                    remaining = len(df_new) - completed
                    eta_min = (remaining / rate) / 60.0 if rate > 0 else float("inf")
                    print(
                        f"[INFO] checkpoint: {completed}/{len(df_new)} snippets, rows={checkpoint_rows}, "
                        f"failures={failures}, meta_only={meta_only}, eta~{eta_min:.1f}m"
                    )
            except Exception as e:
                failures += 1

    if not results:
        print("No intelligence extracted.")
        return

    new_results_df = pd.DataFrame(results)

    final_rows = _write_checkpoint(existing_df, new_results_df, out_path)
    print(f"Saved {final_rows} extraction records to {out_path} ({len(new_results_df)} new, failures={failures})")

    # Verification Output
    print("\n--- Sample Output (Extraction) ---")
    final_df = pd.read_parquet(out_path)
    cols_to_show = [c for c in ['snippet_id', 'entity_name', 'industry', 'category'] if c in final_df.columns]
    print(final_df[cols_to_show].tail(3).to_string())
    print("----------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Extract structured KPIs/entities/SWOT from snippets using Ollama.")
    parser.add_argument("--model", default=config.LLM_MODEL, help="Ollama model name (default: config.LLM_MODEL).")
    parser.add_argument("--out", default=KPI_ENTITIES_FILE, help="Output parquet path (default: DATA/kpi_entities.parquet).")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2).")
    parser.add_argument("--max-snippets", type=int, default=0, help="Process only first N new snippets (0 = all).")
    parser.add_argument("--num-predict", type=int, default=512, help="Max tokens to generate (default: 512).")
    parser.add_argument("--filter-kpi", action="store_true", help="Only process KPI-looking snippets (faster, higher recall than precision).")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint after every N snippets (default: 25).")
    parser.add_argument("--force", action="store_true", help="Re-extract all snippets into a fresh output file (backs up existing parquet).")
    parser.add_argument(
        "--prompt-style",
        choices=["full", "compact"],
        default="full",
        help="Prompt verbosity: compact is faster and often more reliable for JSON.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Convenience preset: --prompt-style compact --filter-kpi, and reduces num-predict.",
    )
    args = parser.parse_args()

    max_snippets = args.max_snippets if args.max_snippets and args.max_snippets > 0 else None
    prompt_style = args.prompt_style
    filter_kpi = bool(args.filter_kpi)
    num_predict = args.num_predict
    workers = args.workers
    save_every = args.save_every

    if args.fast:
        prompt_style = "compact"
        filter_kpi = True
        num_predict = min(int(num_predict), 192)
        workers = min(int(workers), 2)
        save_every = max(int(save_every), 100)

    process_snippets(
        model=args.model,
        out_path=args.out,
        max_workers=workers,
        max_snippets=max_snippets,
        num_predict=num_predict,
        filter_kpi=filter_kpi,
        save_every=save_every,
        force=bool(args.force),
        prompt_style=prompt_style,
    )

if __name__ == "__main__":
    main()
