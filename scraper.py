"""Step 1 - Crawl the ENSAM website and save clean text to data/pages.jsonl.

Improvements over the first notebook version:
  * URL normalisation (no more "/" vs "" duplicates) and skip-list for noisy pages
  * respects robots.txt
  * content-hash de-duplication (same text under different URLs is kept once)
  * boilerplate removal: lines that appear on most pages (menus, footers) are dropped
  * PDFs are downloaded and their text extracted (regulations, admissions, timetables...)
  * every page keeps its <title> so chunks can say where they come from

Usage:  python scraper.py
"""
import hashlib
import io
import json
import re
import time
from collections import Counter, deque
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

import config

SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".rar",
            ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".mp4", ".mp3", ".ico", ".css", ".js")
MAX_PDF_BYTES = 15 * 1024 * 1024
BOILERPLATE_RATIO = 0.3  # a line present on >30% of pages is treated as menu/footer


def normalize_url(url: str) -> str:
    """Canonical form so the same page is never fetched twice."""
    url = urldefrag(url)[0]
    p = urlparse(url)
    path = re.sub(r"/+$", "", p.path) or "/"
    query = urlencode(sorted(parse_qsl(p.query)))
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))


def should_skip(url: str) -> bool:
    return (url.lower().endswith(SKIP_EXT)
            or any(re.search(pat, url) for pat in config.SKIP_URL_PATTERNS))


def html_to_page(url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url

    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg", "iframe"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in root.get_text("\n").splitlines()]
    lines = [ln for ln in lines if ln]
    return {"url": url, "title": title, "type": "html", "lines": lines}, links


def pdf_to_page(url: str, content: bytes):
    reader = PdfReader(io.BytesIO(content))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    title = (reader.metadata.title if reader.metadata and reader.metadata.title
             else url.rsplit("/", 1)[-1])
    return {"url": url, "title": f"[PDF] {title}", "type": "pdf", "lines": [ln for ln in lines if ln]}


def crawl(start_url=config.START_URL, max_pages=config.MAX_PAGES, delay=config.CRAWL_DELAY):
    domain = urlparse(start_url).netloc.lower()
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT

    robots = RobotFileParser(urljoin(start_url, "/robots.txt"))
    try:
        robots.read()
    except Exception:
        robots = None  # no robots.txt reachable -> allow everything

    start = normalize_url(start_url)
    queue, seen, pages = deque([start]), {start}, []

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if robots and not robots.can_fetch(config.USER_AGENT, url):
            continue
        try:
            r = session.get(url, timeout=20)
        except requests.RequestException as e:
            print("  skip", url, type(e).__name__)
            continue
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200:
            continue

        if "application/pdf" in ctype or url.lower().endswith(".pdf"):
            if len(r.content) > MAX_PDF_BYTES:
                continue
            try:
                page = pdf_to_page(url, r.content)
            except Exception as e:
                print("  bad pdf", url, e)
                continue
            links = []
        elif "text/html" in ctype:
            page, links = html_to_page(url, r.text)
        else:
            continue

        for link in links:
            link = normalize_url(link)
            if (urlparse(link).netloc.lower() == domain and link not in seen
                    and not should_skip(link)):
                seen.add(link)
                queue.append(link)

        pages.append(page)
        print(f"[{len(pages)}] {url}  ({page['type']}, {sum(map(len, page['lines']))} chars)")
        time.sleep(delay)
    return pages


def clean_pages(pages, min_chars=200):
    """Drop boilerplate lines shared by many pages, then exact-duplicate pages."""
    html_pages = [p for p in pages if p["type"] == "html"]
    freq = Counter(ln for p in html_pages for ln in set(p["lines"]))
    limit = max(3, int(len(html_pages) * BOILERPLATE_RATIO))
    boilerplate = {ln for ln, n in freq.items() if n > limit}

    cleaned, hashes = [], set()
    for p in pages:
        lines = [ln for ln in p["lines"] if ln not in boilerplate]
        text = "\n".join(lines)
        digest = hashlib.sha1(text.encode()).hexdigest()
        if len(text) < min_chars or digest in hashes:
            continue
        hashes.add(digest)
        cleaned.append({"url": p["url"], "title": p["title"], "type": p["type"], "text": text})
    print(f"Removed {len(boilerplate)} boilerplate lines; kept {len(cleaned)}/{len(pages)} pages")
    return cleaned


def save_pages(pages, path=config.PAGES_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Saved", len(pages), "pages to", path)


if __name__ == "__main__":
    save_pages(clean_pages(crawl()))
