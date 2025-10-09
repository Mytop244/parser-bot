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
    """Извлекает основной текст статьи (3 попытки + fallback)."""
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

    # ---------- 2. Простой метод ----------
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = clean_text(" ".join(paragraphs))
    if isinstance(text, str) and len(text.split()) >= 50:
        return text[:5000].rsplit(" ", 1)[0]

    loop = asyncio.get_running_loop()

    # ---------- 3. Trafilatura ----------
    try:
        import trafilatura
        def trafilatura_extract(html_inner):
            return trafilatura.extract(html_inner, include_comments=False, favor_recall=True)
        extracted = await loop.run_in_executor(None, partial(trafilatura_extract, html))
        if extracted and len(extracted.split()) >= 30:
            return clean_text(extracted)[:5000].rsplit(" ", 1)[0]
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
        if extracted and len(extracted.split()) >= 30:
            return clean_text(extracted)[:5000].rsplit(" ", 1)[0]
    except Exception as e:
        logging.debug(f"readability fail: {e}")

    # ---------- 5. Meta fallback ----------
    meta = (soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", property="og:description") or
            soup.find("meta", property="twitter:description"))
    if meta and meta.get("content"):
        return clean_text(meta.get("content", ""))[:1000].rsplit(" ", 1)[0]

    return ""
