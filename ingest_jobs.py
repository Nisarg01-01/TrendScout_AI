import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

# Configuration
RAW_URL = "https://raw.githubusercontent.com/vanshb03/New-Grad-2026/main/README.md"
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "jobs_raw.parquet")

def extract_jobs_from_table():
    """Fetch and parse jobs from the GitHub README."""
    try:
        response = requests.get(RAW_URL)
        response.raise_for_status()
        md = response.text

        pattern = re.compile(
            r"\| Company \| Role \| Location \| Application/Link \| Date Posted \|\s*\|[-| :]+\|\s*((?:\|.*\|\s*)+)",
            re.MULTILINE
        )

        match = pattern.search(md)
        if not match:
            print("No job table found.")
            return []

        table_block = match.group(1)
        jobs = []

        for line in table_block.strip().split("\n"):
            line = line.strip()
            if not line.startswith("|"):
                continue

            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) < 5:
                continue

            company, role, location, apply_html, date_posted = cols[:5]

            soup = BeautifulSoup(apply_html, "html.parser")
            link_tag = soup.find("a")
            apply_link = link_tag["href"] if link_tag else ""

            jobs.append({
                "company": company,
                "title": role,
                "location": location,
                "url": apply_link,
                "posted_at": date_posted,
                "fetched_at": datetime.utcnow().isoformat()
            })

        return pd.DataFrame(jobs)
    except Exception as e:
        print(f"Error extracting jobs: {e}")
        return pd.DataFrame()

def save_jobs(new_df: pd.DataFrame):
    """Append new jobs to Parquet file, avoiding duplicates."""
    if new_df.empty:
        print("No jobs found.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_parquet(OUTPUT_FILE)
        # Deduplicate against existing data using URL
        # Note: Some jobs might not have a URL, so we might need a composite key, 
        # but for now URL is the best unique ID we have.
        existing_urls = set(existing_df["url"])
        new_df = new_df[~new_df["url"].isin(existing_urls)]
        
        if new_df.empty:
            print("No new unique jobs to append.")
            return
            
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(new_df)} new jobs. Total: {len(combined_df)}")

def main():
    df = extract_jobs_from_table()
    save_jobs(df)

if __name__ == "__main__":
    main()
