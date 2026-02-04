try:
    import feedparser  # type: ignore
except ModuleNotFoundError:
    feedparser = None
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import os
import re
import argparse
import shutil
from urllib.parse import urlsplit
from typing import Iterable
import hashlib

try:
    import requests  # type: ignore
except ModuleNotFoundError:
    requests = None
import config
from utils.id_utils import canonicalize_url, make_article_id


def parse_entry_published(entry: dict) -> str:
    """Parse published date to ISO string (UTC best-effort)."""
    published = entry.get("published", "") or ""
    try:
        parsed = entry.get("published_parsed")
        if parsed:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return str(published)


_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bToggle Mega Menu\b", re.IGNORECASE),
    re.compile(r"\bSite Search\b", re.IGNORECASE),
    re.compile(r"\bSubmit Site Search\b", re.IGNORECASE),
    re.compile(r"\bTechCrunch Desktop Logo\b", re.IGNORECASE),
    re.compile(r"\bTechCrunch Mobile Logo\b", re.IGNORECASE),
    re.compile(r"\bCrunchboard\b", re.IGNORECASE),
    re.compile(r"\bNewsletters\b", re.IGNORECASE),
    re.compile(r"\bPartner Content\b", re.IGNORECASE),
    # Common “subscribe / related links” blocks that pollute extracted HTML.
    re.compile(r"\bBy submitting your email\b", re.IGNORECASE),
    re.compile(r"\bTerms and Privacy Notice\b", re.IGNORECASE),
    re.compile(r"\bPrivacy Notice\b", re.IGNORECASE),
    re.compile(r"\bSubscribe\b", re.IGNORECASE),
    re.compile(r"\bRelated\b", re.IGNORECASE),
    # TechCrunch promo / event blocks that commonly leak into extracted HTML.
    re.compile(r"\bStartups Weekly\b", re.IGNORECASE),
    re.compile(r"\bDisrupt 20\d{2}\b", re.IGNORECASE),
    re.compile(r"\bTickets are live\b", re.IGNORECASE),
    re.compile(r"\bOne-time offer\b", re.IGNORECASE),
    re.compile(r"\bSave up to\b", re.IGNORECASE),
    re.compile(r"\bMeet investors\b", re.IGNORECASE),
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def _text_quality_score(text: str) -> float:
    """
    Heuristic quality score for extracted full text.
    Higher is better. Used to decide whether to upsert newer text.
    """
    t = _normalize_text(text)
    if not t:
        return 0.0

    # Hard fail: newsletter subscribe blocks are not article content, even if long.
    t_l = t.lower()
    if "by submitting your email" in t_l and "privacy" in t_l:
        return 0.0
    if "terms and privacy notice" in t_l and "subscribe" in t_l:
        return 0.0

    penalty = 0.0
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(t):
            # Some patterns are mild, some are strong indicators of non-article text.
            p = pat.pattern.lower()
            if "by submitting your email" in p or "terms and privacy notice" in p:
                penalty += 0.5
            elif "subscribe" in p or "related" in p:
                penalty += 0.35
            elif "disrupt" in p or "tickets are live" in p or "meet investors" in p:
                penalty += 0.35
            else:
                penalty += 0.2

    # Penalize pages that look like lists of headlines + timestamps (common in sidebars/topic pages).
    if len(re.findall(r"\b\d+\s+(?:hours?|days?|minutes?)\s+ago\b", t_l)) >= 3:
        penalty += 0.35
    if len(re.findall(r"\b\d+\s+min\b", t_l)) >= 3:
        penalty += 0.2
    # Penalize extremely long "menu-like" blobs with very low punctuation density.
    punct = sum(1 for ch in t if ch in ".?!;:")
    if len(t) > 1200 and punct / max(1, len(t)) < 0.002:
        penalty += 0.2
    # Penalize low alpha ratio (lots of symbols / nav separators).
    alpha = sum(1 for ch in t if ch.isalpha())
    if alpha / max(1, len(t)) < 0.55:
        penalty += 0.2
    return max(0.0, 1.0 - penalty)


def _should_skip_full_text_fetch(url: str) -> bool:
    """
    Decide whether to skip HTML full-text fetching for a URL.

    Some URLs (e.g., TechCrunch /video/ landing pages) consistently produce non-article boilerplate
    and should fall back to RSS summaries for cleaner snippets.
    """
    u = (url or "").strip().lower()
    if not u:
        return False

    patterns = getattr(config, "SKIP_FULL_TEXT_URL_PATTERNS", []) or []
    for p in patterns:
        ps = str(p).strip().lower()
        if ps and ps in u:
            return True

    # Safe built-in default (in case config is missing).
    parts = urlsplit(u)
    host = parts.netloc or ""
    path = parts.path or ""
    if "techcrunch.com" in host and path.startswith("/video/"):
        return True

    return False


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup(["header", "footer", "nav", "aside"]):
        try:
            tag.decompose()
        except Exception:
            pass

    def extract_from_container(container) -> str:
        if container is None:
            return ""
        # Prefer paragraph-like content over full text to avoid navigation menus.
        parts: list[str] = []
        for el in container.find_all(["p", "h2", "h3", "li"], recursive=True):
            s = el.get_text(" ", strip=True)
            s = _normalize_text(s)
            if len(s) < 20:
                continue
            s_l = s.lower()
            # Skip common "headline list" artifacts (timestamps / promos) that aren't article body text.
            if re.search(r"\b\d+\s+(?:hours?|days?|minutes?)\s+ago\b", s_l):
                continue
            if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", s_l):
                continue
            if "tickets are live" in s_l or "one-time offer" in s_l or "startups weekly" in s_l:
                continue
            parts.append(s)
        text = "\n".join(parts)
        return _normalize_text(text)

    article_tag = soup.find("article")
    text = extract_from_container(article_tag)
    if len(text) >= int(getattr(config, "MIN_ARTICLE_TEXT_CHARS", 400) or 400):
        return text

    main_tag = soup.find("main")
    text = extract_from_container(main_tag)
    if len(text) >= int(getattr(config, "MIN_ARTICLE_TEXT_CHARS", 400) or 400):
        return text

    # Fallback: best-effort whole-page text.
    raw = soup.get_text("\n", strip=True)
    raw = _normalize_text(raw)
    return raw


def fetch_full_text(url: str, timeout_s: int = 15) -> str:
    """Fetch and extract full article text (best-effort)."""
    if not url:
        return ""
    if requests is None:
        return ""
    if not getattr(config, "FETCH_FULL_TEXT", True):
        return ""
    if _should_skip_full_text_fetch(url):
        return ""

    try:
        resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "TrendScoutAI/1.0"})
        resp.raise_for_status()
        txt = extract_text_from_html(resp.text)
        # Drop known boilerplate patterns (best-effort).
        for pat in _BOILERPLATE_PATTERNS:
            txt = pat.sub(" ", txt)
        txt = _normalize_text(txt)
        # If still low quality, return empty so we fall back to RSS summary.
        if _text_quality_score(txt) < 0.55:
            return ""
        return txt
    except Exception:
        return ""


def _match_keywords(text: str, keywords: Iterable[str]) -> bool:
    t = (text or "").lower()
    for kw in keywords:
        k = (kw or "").strip().lower()
        if not k:
            continue
        # Short tokens like "ai"/"ml" must be word-boundary matched to avoid false positives ("said", "email").
        if len(k) <= 3:
            if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", t):
                return True
        else:
            if k in t:
                return True
    return False


def is_ai_relevant(feed_url: str, title: str, summary_text: str, full_text: str = "") -> bool:
    """
    Keep the dataset focused on AI-related startup/market signals.

    Rule of thumb:
    - Must mention AI (with safe keyword matching).
    - Must also look like startup/market context (funding/hiring/product/partnership/etc).
    - Exclude common low-signal consumer noise (reviews, deals, how-to).
    """
    combined = (title or "") + " " + (summary_text or "") + " " + (full_text or "")

    if _match_keywords(combined, getattr(config, "EXCLUDE_KEYWORDS", [])):
        return False

    has_ai = _match_keywords(combined, getattr(config, "AI_KEYWORDS", []))
    if not has_ai:
        return False

    # Some feeds are explicitly startup/business oriented; for those we allow a slightly looser filter.
    feed_l = (feed_url or "").lower()
    is_startup_feed = any(s in feed_l for s in ["techcrunch.com/category/startups", "news.crunchbase.com"])
    has_startup_signal = _match_keywords(combined, getattr(config, "STARTUP_SIGNAL_KEYWORDS", []))

    return has_startup_signal or is_startup_feed

def parse_feed(url: str):
    """Parse RSS feed and filter for AI-related articles."""
    
    try:
        if feedparser is None:
            raise RuntimeError("Dependency missing: feedparser. Install dependencies: pip install -r requirements.txt")
        parsed = feedparser.parse(url)
        articles = []
        total_entries = len(parsed.entries)
        passed_filter = 0
        
        max_entries = int(getattr(config, "MAX_ENTRIES_PER_FEED", 50) or 50)
        entries = list(parsed.entries)[: max_entries if max_entries > 0 else None]

        for entry in entries:
            title = entry.get("title", "")
            summary_html = entry.get("summary", "")
            soup = BeautifulSoup(summary_html, "html.parser")
            summary_text = soup.get_text(" ", strip=True)

            link = entry.get("link") or ""
            canonical_url = canonicalize_url(link)
            published = parse_entry_published(entry)

            body_text = fetch_full_text(canonical_url) if canonical_url else ""
            body_text = re.sub(r"\s+", " ", body_text).strip()

            if not is_ai_relevant(url, title, summary_text, full_text=body_text):
                continue

            passed_filter += 1

            article_id = make_article_id(canonical_url, parsed.feed.get("title", "Unknown"), title, published)

            articles.append({
                "source": parsed.feed.get("title", "Unknown"),
                "title": title,
                "link": link,
                "canonical_url": canonical_url or link,
                "article_id": article_id,
                "summary": summary_text,
                "text": body_text if len(body_text) >= int(getattr(config, "MIN_ARTICLE_TEXT_CHARS", 0) or 0) else (summary_text or body_text),
                "text_source": "fetched" if body_text else "rss",
                "published": published,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        
        print(f"  [{url}] Found {total_entries} entries, {passed_filter} kept (max_entries={max_entries}).")
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

    if "canonical_url" not in new_df.columns:
        new_df = new_df.copy()
        new_df["canonical_url"] = new_df["link"].astype(str).map(canonicalize_url)

    if os.path.exists(config.ARTICLES_FILE):
        existing_df = pd.read_parquet(config.ARTICLES_FILE)
        if "canonical_url" not in existing_df.columns and "link" in existing_df.columns:
            existing_df = existing_df.copy()
            existing_df["canonical_url"] = existing_df["link"].astype(str).map(canonicalize_url)

        # Upsert: if we re-see an existing URL but now have better text (e.g., fetched full body),
        # update the existing record rather than dropping the new one entirely.
        if "canonical_url" in existing_df.columns and "canonical_url" in new_df.columns:
            existing_df = existing_df.copy()
            new_df = new_df.copy()

            # Build index for fast lookup
            idx_by_url = {str(u): i for i, u in enumerate(existing_df["canonical_url"].astype(str).tolist()) if str(u).strip()}

            upgraded = 0
            for _, row in new_df.iterrows():
                url = str(row.get("canonical_url", "")).strip()
                if not url or url not in idx_by_url:
                    continue
                i = idx_by_url[url]
                old_text = str(existing_df.at[i, "text"]) if "text" in existing_df.columns else ""
                new_text = str(row.get("text", ""))
                old_src = str(existing_df.at[i, "text_source"]) if "text_source" in existing_df.columns else ""
                new_src = str(row.get("text_source", ""))

                # Prefer fetched over rss; prefer longer text if fetched.
                should_upgrade = False
                if new_src == "fetched" and old_src != "fetched":
                    should_upgrade = True
                if new_src == "fetched" and len(new_text) > len(old_text) * 1.2:
                    should_upgrade = True
                # Upgrade if we materially improved text quality (even if shorter).
                if new_text and (_text_quality_score(new_text) > _text_quality_score(old_text) + 0.15):
                    should_upgrade = True

                if should_upgrade and new_text:
                    existing_df.at[i, "text"] = new_text
                    if "text_source" in existing_df.columns:
                        existing_df.at[i, "text_source"] = new_src
                    else:
                        existing_df["text_source"] = existing_df.get("text_source", "")
                        existing_df.at[i, "text_source"] = new_src
                    existing_df.at[i, "fetched_at"] = row.get("fetched_at", existing_df.at[i, "fetched_at"] if "fetched_at" in existing_df.columns else "")
                    upgraded += 1

            if upgraded:
                print(f"  Upsert: upgraded {upgraded} existing articles with better text.")

        # Upsert/dedupe using canonical_url (primary) and article_id (secondary).
        existing_links = set(existing_df.get("canonical_url", pd.Series(dtype=str)).dropna().astype(str))
        existing_ids = set(existing_df.get("article_id", pd.Series(dtype=str)).dropna().astype(str))

        initial_count = len(new_df)
        new_df = new_df[~new_df["canonical_url"].astype(str).isin(existing_links)]
        if "article_id" in new_df.columns:
            new_df = new_df[~new_df["article_id"].astype(str).isin(existing_ids)]
        print(f"  Deduplication: {initial_count} candidates -> {len(new_df)} unique new articles.")
        
        if new_df.empty:
            # Still write upgraded records if any were changed.
            existing_df.to_parquet(config.ARTICLES_FILE, index=False)
            print("No new unique articles to append.")
            return
            
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        print(f"Created new storage with {len(new_df)} articles.")

    combined_df.to_parquet(config.ARTICLES_FILE, index=False)
    print(f"Saved {len(new_df)} new articles. Total: {len(combined_df)}")

def main():
    parser = argparse.ArgumentParser(
        description="Ingest RSS feeds into DATA/articles_raw.parquet (incremental by default)."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete DATA/articles_raw.parquet before ingest (writes a .bak_* backup first).",
    )
    parser.add_argument(
        "--max-entries-per-feed",
        type=int,
        default=None,
        help="Override config.MAX_ENTRIES_PER_FEED for this run (useful for quick smoke tests).",
    )
    args = parser.parse_args()

    if args.max_entries_per_feed is not None and int(args.max_entries_per_feed) > 0:
        config.MAX_ENTRIES_PER_FEED = int(args.max_entries_per_feed)

    if args.fresh and os.path.exists(config.ARTICLES_FILE):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = f"{config.ARTICLES_FILE}.bak_{stamp}"
        try:
            shutil.copy2(config.ARTICLES_FILE, bak)
            print(f"[INFO] --fresh enabled. Backup written: {bak}")
        except Exception:
            print("[WARN] Could not create backup for existing articles file.")
        try:
            os.remove(config.ARTICLES_FILE)
            print(f"[INFO] Removed existing articles file: {config.ARTICLES_FILE}")
        except Exception as e:
            raise SystemExit(f"Failed to remove existing articles file: {e}")

    df = fetch_all_feeds()
    save_articles(df)

if __name__ == "__main__":
    main()
