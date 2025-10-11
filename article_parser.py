import asyncio, logging
import aiohttp
from bs4 import BeautifulSoup
from functools import partial

ssl_ctx = None  # задаётся снаружи, если нужно

try:
    from .utils import clean_text
except ImportError:
    clean_text = lambda x: " ".join(x.split())  # fallback


async def extract_article_text(url: str, ssl_context=None) -> str:
    """Извлекает основной текст статьи (3 попытки + fallback) с логированием."""
    ctx = ssl_context or ssl_ctx
    html = ""

    # ---------- 1. Скачиваем HTML ----------
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(url, ssl=ctx or False) as r:
                if r.status != 200:
                    logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                    return ""
                html = await r.text(errors="ignore")
    except Exception as e:
        logging.warning(f"⚠️ Ошибка загрузки {url}: {e}")
        return ""

    if not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside"]):
        tag.decompose()

    loop = asyncio.get_running_loop()

    # ---------- 2. Простой метод <p> ----------
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = clean_text(" ".join(paragraphs))
    logging.debug(f"[simple <p>] len={len(text.split())}")
    if isinstance(text, str) and len(text.split()) >= 20:  # порог снижен
        return text[:10000].rsplit(" ", 1)[0]

    # ---------- 3. Trafilatura ----------
    try:
        import trafilatura
        def trafilatura_extract(html_inner):
            return trafilatura.extract(html_inner, include_comments=False, favor_recall=True)
        extracted = await loop.run_in_executor(None, partial(trafilatura_extract, html))
        if extracted:
            extracted = clean_text(extracted)
            logging.debug(f"[trafilatura] len={len(extracted.split())}")
            if len(extracted.split()) >= 15:  # порог снижен
                return extracted[:10000].rsplit(" ", 1)[0]
    except Exception as e:
        logging.debug(f"trafilatura fail: {e}")

    # ---------- 4. Readability ----------
    try:
        from readability import Document
        def readability_extract(html_inner):
            doc = Document(html_inner)
            summary_html = doc.summary()
            return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
        extracted = await loop.run_in_executor(None, partial(readability_extract, html))
        if extracted:
            extracted = clean_text(extracted)
            logging.debug(f"[readability] len={len(extracted.split())}")
            if len(extracted.split()) >= 15:  # порог снижен
                return extracted[:10000].rsplit(" ", 1)[0]
    except Exception as e:
        logging.debug(f"readability fail: {e}")

    # ---------- 5. Meta fallback ----------
    meta = (soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", property="og:description") or
            soup.find("meta", property="twitter:description"))
    if meta and meta.get("content"):
        content = clean_text(meta.get("content", ""))
        logging.debug(f"[meta fallback] len={len(content.split())}")
        return content[:1000].rsplit(" ", 1)[0]

    logging.debug("[extract_article_text] ничего не извлечено")
    return ""
