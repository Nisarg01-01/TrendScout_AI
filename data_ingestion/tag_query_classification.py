from openai import OpenAI
import os
import json
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_tags(query: str):
    prompt = f"""
    Extract all relevant TechCrunch tags from the user query.

    Output only a JSON object with two arrays:

    {{
      "companies": [],
      "topics": []
    }}

    Rules:
    - "companies": list every company mentioned (apple, nvidia, meta)
    - "topics": list topics/technologies (virtual reality, cloud computing)
    - Convert every element to valid TechCrunch tag format:
        - lowercase
        - hyphens instead of spaces
        - no punctuation
    - If nothing found, return empty arrays.

    User query: "{query}"
    """

    response = gpt_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content.strip()
    
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"companies": [], "topics": []}





def classify_query(query: str) -> str:
    """
    Classifies a user query into either 'jobs' or 'news'.
    Returns: "jobs" or "news"
    """
    prompt = f"""
    You are a classifier. You must reply with exactly one word: "jobs" or "news".

    If the user is asking about hiring, openings, roles, workers, recruiting,
    companies hiring, placements, internships or job lists → reply "jobs".

    Otherwise, if the user is asking about events, updates, trends, releases,
    companies, business changes, product launches → reply "news".

    Query: "{query}"
    Answer (one word only):
    """

    response = gpt_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip().lower()
