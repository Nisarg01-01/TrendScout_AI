"""
ingest_tc.py
-------------
Fetches startup-related RSS feeds (TechCrunch, VentureBeat, Wired, etc.),
filters only AI-related articles, cleans text, saves locally, and optionally uploads to Google Drive.

Legal compliance: all sources expose public RSS feeds intended for aggregation.
Full article text is not scraped — only metadata and feed summaries are stored.
"""

import os
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# Optional: Drive upload
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    GoogleAuth = None
    GoogleDrive = None

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

# RSS feeds (public, legal)
FEEDS = [
    "https://techcrunch.com/startups/feed/",
    "https://venturebeat.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://news.crunchbase.com/feed/"
]

# Keywords to keep only AI-related articles
AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning",
    "generative ai", "large language model", "llm", "computer vision",
    "nlp", "autonomous", "ai startup", "ai company", "ai tool"
]

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "articles_raw.parquet")

# Google Drive folder id (from .env)
GDRIVE_FOLDER_ID = os.getenv("DATA_DRIVE_FOLDER_ID")  # optional
GDRIVE_CREDS = "gdrive_creds.json"                    # service account JSON file

# -------------------------------------------------------------------
# FEED PROCESSING
# -------------------------------------------------------------------


def parse_feed(url: str):
    """Parse one RSS feed → list of dicts."""
    parsed = feedparser.parse(url)
    articles = []
    for entry in parsed.entries:
        title = entry.get("title", "")
        summary_html = entry.get("summary", "")
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(" ", strip=True)
        combined = (title + " " + text).lower()

        # Filter for AI relevance
        if not any(k in combined for k in AI_KEYWORDS):
            continue

        articles.append({
            "source": parsed.feed.get("title", "Unknown"),
            "title": title,
            "link": entry.get("link"),
            "published": entry.get("published", ""),
            "summary": text,
            "fetched_at": datetime.utcnow().isoformat()
        })
    return articles


def fetch_all_feeds():
    """Fetch all configured feeds and merge results."""
    all_articles = []
    for url in FEEDS:
        try:
            print(f"Fetching {url}")
            articles = parse_feed(url)
            print(f"  → {len(articles)} AI-related articles")
            all_articles.extend(articles)
        except Exception as e:
            print(f"Failed {url}: {e}")

    df = pd.DataFrame(all_articles).drop_duplicates(subset=["link"])
    return df


def save_local(df: pd.DataFrame):
    """Save dataframe locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} articles to {OUTPUT_FILE}")


def upload_to_gdrive(local_path: str, folder_id: str):
    """Upload file to Google Drive using service account JSON credentials."""
    if not folder_id or not os.path.exists(GDRIVE_CREDS):
        print("Drive upload skipped (missing folder id or credentials).")
        return

    try:
        gauth = GoogleAuth()
        gauth.LoadServiceConfigFile(GDRIVE_CREDS)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)

        file_name = os.path.basename(local_path)
        file = drive.CreateFile({"title": file_name, "parents": [{"id": folder_id}]})
        file.SetContentFile(local_path)
        file.Upload()
        print(f"Uploaded {file_name} to Google Drive folder {folder_id}")
    except Exception as e:
        print("Drive upload failed:", e)


def main():
    df = fetch_all_feeds()
    if df.empty:
        print("No AI-related articles found.")
        return
    save_local(df)
    upload_to_gdrive(OUTPUT_FILE, GDRIVE_FOLDER_ID)


if __name__ == "__main__":
    main()
