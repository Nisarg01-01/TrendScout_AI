import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
}


def canonicalize_url(url: str) -> str:
    """Best-effort canonicalization for article URLs (remove fragment + tracking params)."""
    if not isinstance(url, str):
        return ""

    raw = url.strip()
    if not raw:
        return ""

    parts = urlsplit(raw)

    query_items = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        key = (k or "").strip()
        if not key:
            continue
        key_lower = key.lower()
        if key_lower.startswith("utm_") or key_lower in TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, v))

    new_query = urlencode(query_items, doseq=True)
    cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, ""))  # drop fragment
    return cleaned.strip()


def article_key_from_fields(link: str, source: str, title: str, published: str) -> str:
    """Stable-ish key when canonical URL is unavailable."""
    link_val = canonicalize_url(link)
    if link_val:
        return link_val
    parts = [source, title, published]
    parts = [str(p).strip() for p in parts if p]
    return "|".join(parts)


def make_article_id(link: str, source: str, title: str, published: str) -> str:
    """Create a deterministic article ID."""
    key = article_key_from_fields(link, source, title, published)
    if not key:
        return ""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def make_snippet_id(article_id: str, chunk_index: int) -> str:
    """Create a deterministic snippet ID based on article id and chunk index."""
    if not article_id:
        return ""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{article_id}|{int(chunk_index)}"))

