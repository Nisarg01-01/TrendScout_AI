import os
import sys
import time
import re
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Ensure project root in path so we can import config
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

BASE_URL = "https://techcrunch.com/2025/page/{page}/"

# For now this script lives under data_bootstrap/ in source control.
# We keep MAX_PAGES small by default so initial runs are cheap; you can
# increase up to ~225 when ready for a full-year import.
MAX_PAGES = 15

# Categories we consider relevant even before keyword filtering
RELEVANT_CATEGORIES = {
    "artificial-intelligence",  # AI
    "startups",
    "fundraising",
    "fintech",
    "enterprise",
    "biotech-health",
}


def is_relevant(title: str, category_slug: str) -> bool:
    """Cheap pre-filter based on category and AI/funding keywords."""
    title_l = (title or "").lower()

    # Category-based inclusion
    if category_slug in RELEVANT_CATEGORIES:
        return True

    # Keyword-based inclusion using config.AI_KEYWORDS plus funding terms
    funding_terms = ["seed round", "series a", "series b", "series c", "funding", "raises", "raised", "valuation"]
    keywords = [k.lower() for k in getattr(config, "AI_KEYWORDS", [])] + funding_terms

    return any(k in title_l for k in keywords)


def parse_archive_page(page: int) -> List[Dict]:
    url = BASE_URL.format(page=page)
    print(f"Fetching archive page {page}: {url}")
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        print(f"  ! Skipping page {page}, status {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    articles: List[Dict] = []

    # Each article headline is typically in an h3 or h2 with an <a> tag
    # We'll look for all h3/h2 under the 2025 section whose links look like article URLs.
    for h in soup.find_all(["h2", "h3"]):
        a = h.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        title = a.get_text(strip=True)

        # Filter out non-article links (e.g., video, events, etc.)
        if not href.startswith("https://techcrunch.com/2025/"):
            continue

        # Attempt to find the date and category nearby
        # Date is usually a sibling text like "- Jan 13, 2025"
        date_text = None
        category_slug = None

        # Look at following siblings for a date line
        sibling = h.find_next_sibling()
        while sibling is not None and date_text is None:
            text = sibling.get_text(" ", strip=True)
            if re.search(r"\b\d{4}\b", text):  # contains a year
                date_text = text
                break
            sibling = sibling.find_next_sibling()

        # Try to find category link (it has /category/.../ in href)
        cat_link = h.find_next("a", href=re.compile(r"/category/"))
        if cat_link and "href" in cat_link.attrs:
            m = re.search(r"/category/([^/]+)/", cat_link["href"])
            if m:
                category_slug = m.group(1)

        if not is_relevant(title, category_slug or ""):
            continue

        articles.append(
            {
                "title": title,
                "url": href,
                "date_hint": date_text,
                "category": category_slug,
                "archive_page": page,
            }
        )

    print(f"  -> kept {len(articles)} relevant articles from page {page}")
    return articles


def fetch_article_body(url: str) -> str:
    """Fetch full article HTML and extract main text body."""
    try:
        resp = requests.get(url, timeout=20)
    except Exception as e:
        print(f"    ! Error fetching {url}: {e}")
        return ""

    if resp.status_code != 200:
        print(f"    ! Non-200 for {url}: {resp.status_code}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # TechCrunch articles typically have content in <div class="article-content"> or similar
    main = soup.find("div", class_=re.compile(r"article-content|content"))
    if not main:
        # Fallback: all paragraphs under main article tag
        main = soup.find("article")
    if not main:
        return ""

    paras = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    text = "\n".join(p for p in paras if p)
    return text


def main():
    all_meta: List[Dict] = []

    for page in range(1, MAX_PAGES + 1):
        try:
            page_articles = parse_archive_page(page)
        except Exception as e:
            print(f"Error on archive page {page}: {e}")
            continue
        all_meta.extend(page_articles)
        time.sleep(1)  # be polite

    if not all_meta:
        print("No relevant articles found. Exiting.")
        return

    print(f"Total relevant article stubs: {len(all_meta)}")

    # Fetch bodies
    records = []
    for i, art in enumerate(all_meta, 1):
        print(f"[{i}/{len(all_meta)}] Fetching article body: {art['url']}")
        body = fetch_article_body(art["url"])
        if not body:
            continue
        records.append(
            {
                "id": art["url"],
                "title": art["title"],
                "source": "TechCrunch",
                "url": art["url"],
                "published": art["date_hint"],
                "category": art["category"],
                "text": body,
                "archive_page": art["archive_page"],
            }
        )
        time.sleep(0.5)

    if not records:
        print("No article bodies fetched. Exiting.")
        return

    df = pd.DataFrame(records)

    # Store bootstrap artifacts under a dedicated data_bootstrap directory
    bootstrap_dir = os.path.join(ROOT_DIR, "data_bootstrap")
    os.makedirs(bootstrap_dir, exist_ok=True)
    out_path = os.path.join(bootstrap_dir, "techcrunch_2025_bootstrap.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} articles to {out_path}")

    # Show a small sample
    print("\nSample rows:\n", df[["title", "published", "category", "url"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
