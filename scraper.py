#!/usr/bin/env python3
"""
Robust recursive scraper for documentation sites.

Usage:
    python scraper.py --base-url https://docs.frrouting.org/en/latest/ --output fr_docs --delay 0.1 --max-depth 6

Notes:
- Respects a conservative delay and max concurrency = 1 (synchronous).
- Filters non-HTML resources (images, pdfs, css, js).
- Normalizes URLs (removes anchors, ensures index.html).
- Stores files preserving site structure as Markdown (.md) with JSON metadata (.metadata.json).
- Uses simple retry logic for network errors.
- Output format optimized for RAG: Markdown content + JSON metadata.
"""

import os
import time
import json
import argparse
import requests
from datetime import datetime, timezone
from urllib.parse import urljoin, urldefrag, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import html2text

HEADERS = {
    "User-Agent": "fr-docs-scraper/1.0 (+https://example.com) - polite scraping for personal use"
}

# HTML to Markdown converter
h2t = html2text.HTML2Text()
h2t.ignore_links = False  # сохраняем ссылки
h2t.ignore_images = True  # игнорируем изображения
h2t.body_width = 0  # не переносим строки
h2t.unicode_snob = True  # лучше обрабатываем unicode

# A list of file extensions we DON'T want to fetch as pages
NON_HTML_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".tgz",
    ".css", ".js", ".woff", ".woff2", ".ico",
    ".mp4", ".avi", ".mov", ".exe", ".rpm",
    ".deb", ".rst", ".md", ".txt"  # keep .txt out because we want rendered html text
)

def is_html_like(url: str) -> bool:
    """
    Return True if we consider this URL points to a html page worth scraping.
    """
    path = urlparse(url).path.lower()
    # if path clearly ends with non-html extension — ignore
    for ext in NON_HTML_EXTENSIONS:
        if path.endswith(ext):
            return False
    # otherwise allow; we'll enforce .html or trailing slash normalization later
    return True

def normalize_url(url: str, base: str) -> str:
    """
    Normalize URL: resolve relative, remove fragment, convert directory to index.html if needed.
    """
    # Resolve relative to base
    full = urljoin(base, url)
    # Remove fragment (#anchor)
    full, _ = urldefrag(full)
    parsed = urlparse(full)

    # Ensure scheme/netloc present
    if not parsed.scheme or not parsed.netloc:
        return None

    path = parsed.path
    
    # Если уже заканчивается на index.html, возвращаем как есть
    if path.endswith("/index.html") or path == "/index.html":
        return full
    
    # Если путь заканчивается на '/', добавляем index.html
    if path.endswith("/"):
        full = full.rstrip("/") + "/index.html"
        # Перепарсим после изменения
        parsed = urlparse(full)
        path = parsed.path
    
    # Если нет расширения и не заканчивается на index.html, добавляем index.html
    if not os.path.splitext(path)[1] and not path.endswith("/index.html"):
        full = full.rstrip("/") + "/index.html"

    return full

def same_site(url: str, base_netloc: str) -> bool:
    try:
        return urlparse(url).netloc == base_netloc
    except Exception:
        return False

def safe_get(url: str, timeout: int = 10, max_retries: int = 3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            time.sleep(1 + 0.5 * attempt)
    # returning None signals failure
    return None

def extract_links_from_html(html_text: str, current_url: str, base_url: str):
    soup = BeautifulSoup(html_text, "html.parser")
    anchors = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href:
            continue
        # normalize relative and anchor-less urls later
        full = urljoin(current_url, href)
        # skip javascript: or mailto:
        if full.startswith("javascript:") or full.startswith("mailto:"):
            continue
        # skip pure anchors
        if full.startswith("#"):
            continue
        anchors.add(full)
    return anchors

def extract_markdown_and_metadata(html_text: str, url: str):
    """
    Извлекает Markdown и метаданные из HTML.
    Возвращает (markdown_text, metadata_dict).
    """
    soup = BeautifulSoup(html_text, "html.parser")
    
    # Извлекаем заголовок страницы
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    
    # Извлекаем основной контент
    main = soup.find("div", attrs={"role": "main"})
    if not main:
        main = soup.find("article")
    if not main:
        main = soup.find("body")
    if not main:
        main = soup
    
    # Конвертируем в Markdown
    markdown_text = h2t.handle(str(main))
    
    # Извлекаем метаданные
    metadata = {
        "url": url,
        "title": title,
        "scraped_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "fr-docs-scraper"
    }
    
    # Опционально: извлекаем описание из meta tags
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        metadata["description"] = meta_desc["content"]
    
    # Опционально: breadcrumbs (если есть)
    breadcrumbs = []
    nav = soup.find("nav", attrs={"aria-label": "breadcrumb"}) or soup.find("ol", class_="breadcrumb")
    if nav:
        for li in nav.find_all("li"):
            text = li.get_text(strip=True)
            if text:
                breadcrumbs.append(text)
    if breadcrumbs:
        metadata["breadcrumbs"] = breadcrumbs
    
    return markdown_text.strip(), metadata

def save_markdown_for_rag(output_dir: str, base_url: str, url: str, markdown: str, metadata: dict):
    """
    Сохраняет Markdown файл и JSON с метаданными для RAG.
    Структура: 
    - page.md
    - page.metadata.json
    """
    # Создаем относительный путь используя urlparse
    base_parsed = urlparse(base_url)
    url_parsed = urlparse(url)
    
    # Если это тот же домен, извлекаем путь относительно base_url
    if url_parsed.netloc == base_parsed.netloc:
        base_path = base_parsed.path.rstrip("/")
        url_path = url_parsed.path
        if url_path.startswith(base_path):
            rel = url_path[len(base_path):].lstrip("/")
        else:
            rel = url_path.lstrip("/")
    else:
        # Разные домены - используем полный путь с доменом
        rel = url_parsed.netloc + url_parsed.path.lstrip("/")
    
    # Убираем .html если есть
    if rel.endswith(".html"):
        rel = rel[:-5]
    if rel.endswith("/"):
        rel = rel.rstrip("/")
    if not rel:
        rel = "index"
    
    # Заменяем недопустимые символы для файловой системы
    rel = rel.replace("://", "_").replace("/", os.sep)
    
    # Путь для .md файла
    md_filepath = os.path.join(output_dir, rel + ".md")
    os.makedirs(os.path.dirname(md_filepath), exist_ok=True)
    
    # Путь для .json метаданных
    json_filepath = os.path.join(output_dir, rel + ".metadata.json")
    
    # Сохраняем Markdown
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    # Сохраняем метаданные
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return md_filepath, json_filepath

def crawl(base_url: str, output_dir: str, delay: float = 0.1, max_depth: int = 6, max_pages: int = None):
    """
    Recursive crawl, synchronous, polite.
    - base_url should be like https://docs.frrouting.org/en/latest/
    """
    base_url_norm = normalize_url(base_url, base_url)
    if not base_url_norm:
        raise ValueError("Invalid base_url")

    base_netloc = urlparse(base_url_norm).netloc
    visited = set()
    to_visit = [(base_url_norm, 0)]
    pages_saved = 0

    pbar = tqdm(total=max_pages or 0, unit="page", desc="scraped", leave=True) if max_pages else None

    while to_visit:
        url, depth = to_visit.pop()
        if url in visited:
            continue
        if depth > max_depth:
            # skip too deep links
            continue

        visited.add(url)

        # skip non-html obvious links
        if not is_html_like(url):
            # print skipped ext (optional)
            # print("[skip ext]", url)
            continue

        # ensure site
        if not same_site(url, base_netloc):
            continue

        # polite delay
        time.sleep(delay)

        resp = safe_get(url)
        if resp is None:
            print(f"[!] Failed to fetch: {url}")
            continue

        # content-type check - prefer text/html
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            # skip binary resources
            # print("[skip ctype]", url, ctype)
            continue

        # Извлекаем Markdown и метаданные
        markdown, metadata = extract_markdown_and_metadata(resp.text, url)
        
        try:
            md_path, json_path = save_markdown_for_rag(
                output_dir, base_url_norm, url, markdown, metadata
            )
            pages_saved += 1
            if pbar:
                pbar.update(1)
            print(f"[OK] {md_path}")
        except Exception as e:
            print(f"[!] Error saving {url}: {e}")

        # extract links and push into to_visit
        found = extract_links_from_html(resp.text, url, base_url_norm)
        for f in found:
            # normalize, ensure same site, skip non-html
            norm = normalize_url(f, base_url_norm)
            if not norm:
                continue
            if not same_site(norm, base_netloc):
                continue
            if norm in visited:
                continue
            if not is_html_like(norm):
                continue
            # avoid huge crawl loops
            if norm not in [u for u, _d in to_visit]:
                to_visit.append((norm, depth + 1))

        # optional: stop after max_pages
        if max_pages and pages_saved >= max_pages:
            break

    if pbar:
        pbar.close()
    print(f"\nDone. pages_saved={pages_saved} visited_total={len(visited)}")

def main():
    parser = argparse.ArgumentParser(description="Simple polite doc scraper")
    parser.add_argument("--base-url", required=True, help="Base documentation URL (e.g. https://docs.frrouting.org/en/latest/)")
    parser.add_argument("--output", default="frr_docs", help="Output directory")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests in seconds")
    parser.add_argument("--max-depth", type=int, default=6, help="Max recursion depth")
    parser.add_argument("--max-pages", type=int, default=None, help="Max number of pages to save (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    crawl(args.base_url, args.output, args.delay, args.max_depth, args.max_pages)

if __name__ == "__main__":
    main()

