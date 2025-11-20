import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os

# Configuration
FEEDS = [
    "https://techcrunch.com/startups/feed/",
    "https://venturebeat.com/feed/",
    "https://www.theverge.com/rss/index.xml",
    "https://www.wired.com/feed/rss",
    "https://news.crunchbase.com/feed/",
]

AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning",
    "generative ai", "large language model", "llm", "computer vision",
    "nlp", "autonomous", "ai startup", "ai company", "ai tool"
]

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "articles_raw.parquet")

def parse_feed(url: str):
    """Parse one RSS feed → list of dicts."""
    try:
        parsed = feedparser.parse(url)
        articles = []
        for entry in parsed.entries:
            title = entry.get("title", "")
            summary_html = entry.get("summary", "")
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(" ", strip=True)
            
            # Basic AI filtering
            combined = (title + " " + text).lower()
            if not any(k in combined for k in AI_KEYWORDS):
                continue

            articles.append({
                "source": parsed.feed.get("title", "Unknown"),
                "title": title,
                "link": entry.get("link"),
                "summary": text,
                "published": entry.get("published", ""),
                "fetched_at": datetime.utcnow().isoformat()
            })
        return articles
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []

def fetch_all_feeds():
    """Fetch all configured feeds."""
    all_articles = []
    for url in FEEDS:
        print(f"Fetching {url}...")
        articles = parse_feed(url)
        all_articles.extend(articles)
    return pd.DataFrame(all_articles)

def save_articles(new_df: pd.DataFrame):
    """Append new articles to Parquet file, avoiding duplicates."""
    if new_df.empty:
        print("No new articles found.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_parquet(OUTPUT_FILE)
        # Deduplicate against existing data
        # We use 'link' as the unique identifier
        existing_links = set(existing_df["link"])
        new_df = new_df[~new_df["link"].isin(existing_links)]
        
        if new_df.empty:
            print("No new unique articles to append.")
            return
            
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(new_df)} new articles. Total: {len(combined_df)}")

def main():
    df = fetch_all_feeds()
    save_articles(df)

if __name__ == "__main__":
    main()
