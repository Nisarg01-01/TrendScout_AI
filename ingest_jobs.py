import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os
import config

def extract_jobs_from_table():
    """
    I'm going to scrape that GitHub README to find job listings.
    It's a markdown table, so I'll use some regex magic to pull out the rows.
    """
    try:
        response = requests.get(config.JOBS_URL)
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
        total_rows = 0
        passed_filter = 0

        for line in table_block.strip().split("\n"):
            line = line.strip()
            if not line.startswith("|"):
                continue

            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) < 5:
                continue
            
            total_rows += 1
            company, role, location, apply_html, date_posted = cols[:5]

            # I only care about AI jobs, so I'll check the company and role against our keywords.
            combined_text = (company + " " + role).lower()
            if not any(k in combined_text for k in config.AI_KEYWORDS):
                continue
            
            passed_filter += 1
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
        
        print(f"Found {total_rows} job rows, {passed_filter} kept (AI keywords).")
        return pd.DataFrame(jobs)

        return pd.DataFrame(jobs)
    except Exception as e:
        print(f"Error extracting jobs: {e}")
        return pd.DataFrame()

def save_jobs(new_df: pd.DataFrame):
    """
    Saving the jobs to our parquet file.
    Standard procedure: check for duplicates based on the URL so we don't double-count.
    """
    if new_df.empty:
        print("No jobs found.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.JOBS_FILE):
        existing_df = pd.read_parquet(config.JOBS_FILE)
        # Deduplicate against existing data using URL
        existing_urls = set(existing_df["url"])
        initial_count = len(new_df)
        new_df = new_df[~new_df["url"].isin(existing_urls)]
        print(f"  Deduplication: {initial_count} candidates -> {len(new_df)} unique new jobs.")
        
        if new_df.empty:
            print("No new unique jobs to append.")
            return
            
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        print(f"Created new storage with {len(new_df)} jobs.")

    combined_df.to_parquet(config.JOBS_FILE, index=False)
    print(f"Saved {len(new_df)} new jobs. Total: {len(combined_df)}")

def main():
    df = extract_jobs_from_table()
    save_jobs(df)

if __name__ == "__main__":
    main()
