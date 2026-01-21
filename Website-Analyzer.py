"""
-----------------------------
WEBSITE ANALYZER (Free Tech Detection)
Industry-grade, all-rounder version
-----------------------------
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse, urlunparse
import requests
from bs4 import BeautifulSoup

# -----------------------------
# CONFIG
# -----------------------------

DEFAULT_TIMEOUT_SEC: int = 20
MAX_CONTENT_SUMMARY_CHARS: int = 3000

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36 "
        "WebsiteAnalyzerBot/1.0"
    )
}

DEFAULT_STOPWORDS: Set[str] = {
    "that",
    "with",
    "this",
    "from",
    "your",
    "have",
    "will",
    "they",
    "their",
    "about",
    "which",
    "when",
    "where",
    "what",
    "there",
    "were",
    "been",
}


logger = logging.getLogger("website_analyzer")
if not logger.handlers:
    # Basic logging configuration; can be overridden by the caller
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


# -----------------------------
# URL UTILITIES
# -----------------------------

def normalize_url(url: str) -> str:
    """
    Normalize a URL string:

    - Add https:// if scheme is missing
    - Lower-case the scheme and hostname
    """
    if not url:
        raise ValueError("URL must not be empty")

    parsed = urlparse(url)

    if not parsed.scheme:
        # Assume https by default
        parsed = urlparse("https://" + url)

    netloc = parsed.netloc.lower()
    scheme = (parsed.scheme or "https").lower()

    normalized = parsed._replace(scheme=scheme, netloc=netloc)
    return urlunparse(normalized)


# -----------------------------
# STEP 1: FETCH HTML
# -----------------------------

def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT_SEC) -> str:
    """
    Fetch raw HTML from a URL.

    Returns an empty string on error, but logs details.
    """
    normalized_url = normalize_url(url)
    try:
        logger.info("Fetching URL: %s", normalized_url)
        resp = requests.get(normalized_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        logger.info("Fetched %d bytes from %s", len(resp.text), normalized_url)
        return resp.text
    except requests.exceptions.RequestException as exc:
        logger.error("Error fetching HTML from %s: %s", normalized_url, exc)
        return ""


# -----------------------------
# STEP 2: EXTRACT VISIBLE TEXT
# -----------------------------

def extract_visible_text(html: str) -> str:
    """
    Extract human-visible text from HTML, stripping scripts/styles/etc.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# STEP 3: SEO ANALYSIS
# -----------------------------

def _extract_meta_content(
    soup: BeautifulSoup, names: Sequence[str]
) -> Optional[str]:
    """
    Helper for safely extracting meta tags by name or property.
    """
    for name in names:
        meta_tag = soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"property": name}
        )
        if meta_tag and meta_tag.get("content"):
            return meta_tag["content"].strip()
    return None


def seo_analysis(
    html: str,
    *,
    stopwords: Optional[Set[str]] = None,
    max_keywords: int = 15,
) -> Dict[str, Any]:
    """
    Perform a lightweight SEO analysis on HTML.

    Returns:
        Dict with title, meta description, OG tags, headers, and top keywords.
    """
    if not html:
        return {
            "title": None,
            "meta_description": None,
            "og_title": None,
            "og_description": None,
            "headers": {"h1": [], "h2": [], "h3": []},
            "top_keywords": [],
        }

    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    meta_desc = _extract_meta_content(soup, ["description"])
    og_title = _extract_meta_content(soup, ["og:title"])
    og_description = _extract_meta_content(soup, ["og:description"])

    headers: Dict[str, List[str]] = {
        "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
        "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
        "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
    }

    # Keyword extraction (simple & fast)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", soup.get_text().lower())
    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    keywords = [w for w in words if w not in sw]
    keyword_freq: List[Tuple[str, int]] = Counter(keywords).most_common(
        max_keywords
    )

    return {
        "title": title,
        "meta_description": meta_desc,
        "og_title": og_title,
        "og_description": og_description,
        "headers": headers,
        "top_keywords": keyword_freq,
    }


# -----------------------------
# STEP 4: FREE TECH STACK DETECTION
# -----------------------------

def detect_tech_stack_from_html(html: str) -> List[str]:
    """
    Detect a rough tech stack using only HTML patterns (no paid services).
    """
    if not html:
        return ["Unknown / not detected"]

    html_l = html.lower()
    tech: Set[str] = set()

    # CMS
    if "wp-content" in html_l or "wordpress" in html_l:
        tech.add("WordPress")
    if "cdn.shopify.com" in html_l or "shopify" in html_l:
        tech.add("Shopify")
    if "wix.com" in html_l or "wixstatic.com" in html_l:
        tech.add("Wix")
    if "webflow" in html_l:
        tech.add("Webflow")

    # JS Frameworks & Libraries
    if "react" in html_l:
        tech.add("React")
    if "next.js" in html_l or "nextjs" in html_l:
        tech.add("Next.js")
    if "vue" in html_l:
        tech.add("Vue.js")
    if "angular" in html_l:
        tech.add("Angular")
    if "jquery" in html_l:
        tech.add("jQuery")
    if "bootstrap" in html_l:
        tech.add("Bootstrap")

    # Analytics
    if "gtag.js" in html_l or "google-analytics.com" in html_l:
        tech.add("Google Analytics")
    if "fbq(" in html_l or "facebook.net" in html_l:
        tech.add("Meta Pixel / Facebook Pixel")

    # Misc patterns (CDN detection)
    if re.search(r"cdn\.\w+\.\w+", html_l):
        tech.add("CDN detected")

    if not tech:
        tech.add("Unknown / not detected")

    return sorted(tech)


# -----------------------------
# STEP 5: MAIN WEBSITE ANALYZER
# -----------------------------

def analyze_website(
    url: str,
    *,
    max_content_chars: int = MAX_CONTENT_SUMMARY_CHARS,
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """
    High-level website analysis combining content, SEO, and tech stack.
    """
    normalized_url = normalize_url(url)
    html = fetch_html(normalized_url, timeout=timeout)

    if not html:
        logger.warning("No HTML returned from %s", normalized_url)

    text = extract_visible_text(html)
    seo = seo_analysis(html)
    tech_stack = detect_tech_stack_from_html(html)

    domain = urlparse(normalized_url).netloc

    result: Dict[str, Any] = {
        "url": normalized_url,
        "domain": domain,
        "content_summary": text[:max_content_chars],
        "seo": seo,
        "tech_stack": tech_stack,
    }

    return result


# -----------------------------
# CLI ENTRYPOINT
# -----------------------------

def _cli() -> None:
    """
    Simple command-line interface for ad-hoc usage.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Analyze a public website (free tech & SEO detection)."
    )
    parser.add_argument("url", help="Website URL to analyze")
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=MAX_CONTENT_SUMMARY_CHARS,
        help=f"Maximum characters to keep in content summary "
        f"(default: {MAX_CONTENT_SUMMARY_CHARS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    try:
        analysis = analyze_website(
            args.url,
            max_content_chars=args.max_content_chars,
            timeout=args.timeout,
        )
    except Exception as exc:  # Final safety net for CLI usage
        logger.exception("Unexpected error while analyzing website: %s", exc)
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    if args.pretty:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(analysis, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    _cli()

