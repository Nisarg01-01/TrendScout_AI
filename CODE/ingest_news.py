import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import os
import config

def parse_feed(url: str):
    """Parse RSS feed and filter for AI-related articles."""
    
    try:
        parsed = feedparser.parse(url)
        articles = []
        total_entries = len(parsed.entries)
        passed_filter = 0
        
        for entry in parsed.entries:
            title = entry.get("title", "")
            summary_html = entry.get("summary", "")
            soup = BeautifulSoup(summary_html, "html.parser")
            text = soup.get_text(" ", strip=True)
            
            # Trust AI-specific feeds, otherwise filter by keywords
            is_ai_feed = "artificial-intelligence" in url or "ai" in url.split('/')[-2:] or "openai" in url
            
            combined = (title + " " + text).lower()
            if not is_ai_feed and not any(k in combined for k in config.AI_KEYWORDS):
                continue
            
            passed_filter += 1
            articles.append({
                "source": parsed.feed.get("title", "Unknown"),
                "title": title,
                "link": entry.get("link"),
                "summary": text,
                "published": entry.get("published", ""),
                "fetched_at": datetime.utcnow().isoformat()
            })
        
        print(f"  [{url}] Found {total_entries} entries, {passed_filter} kept.")
        return articles
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return []

def fetch_all_feeds():
    """Fetch and combine articles from all configured news feeds."""
    
    all_articles = []
    for url in config.FEEDS:
        print(f"Fetching {url}...")
        articles = parse_feed(url)
        all_articles.extend(articles)
    return pd.DataFrame(all_articles)

def save_articles(new_df: pd.DataFrame):
    """Save new articles to storage, deduplicating against existing data."""
    
    if new_df.empty:
        print("No new articles found.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    if os.path.exists(config.ARTICLES_FILE):
        existing_df = pd.read_parquet(config.ARTICLES_FILE)
        # Deduplicate using 'link' as unique identifier
        existing_links = set(existing_df["link"])
        initial_count = len(new_df)
        new_df = new_df[~new_df["link"].isin(existing_links)]
        print(f"  Deduplication: {initial_count} candidates -> {len(new_df)} unique new articles.")
        
        if new_df.empty:
            print("No new unique articles to append.")
            return
            
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        print(f"Created new storage with {len(new_df)} articles.")

    combined_df.to_parquet(config.ARTICLES_FILE, index=False)
    print(f"Saved {len(new_df)} new articles. Total: {len(combined_df)}")

def main():
    df = fetch_all_feeds()
    save_articles(df)

if __name__ == "__main__":
    main()
