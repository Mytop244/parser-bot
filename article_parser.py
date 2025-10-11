import asyncio, logging
import aiohttp
from bs4 import BeautifulSoup
from functools import partial

ssl_ctx = None  # задаётся снаружи, если нужно

try:
    from .utils import clean_text
except ImportError:
    clean_text = lambda x: " ".join(x.split())  # fallback


async def extract_article_text(
    url: str,
    ssl_context=None,
    max_length: int = 5000,
    session: aiohttp.ClientSession | None = None
) -> str:
    """Извлекает текст статьи. Ретрай загрузки, потоковое чтение с лимитом, несколько экстракторов."""
    ctx = ssl_context if ssl_context is not None else ssl_ctx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsBot/1.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    # лимит скачиваемого HTML (байт)
    MAX_DOWNLOAD = max(200_000, min(1_000_000, max_length * 200))

    async def _read_limited(resp, max_bytes):
        chunks = []
        size = 0
        async for chunk in resp.content.iter_chunked(8192):
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes:
                break
        try:
            return b"".join(chunks).decode(errors="ignore")
        except Exception:
            return b"".join(chunks).decode("utf-8", errors="ignore")

    # ---------- 1. Скачиваем HTML с ретраями ----------
    html = ""
    import asyncio, logging, aiohttp
    from logging.handlers import RotatingFileHandler
    from bs4 import BeautifulSoup
    from functools import partial

    # ------------------ LOGGING (ротация 5 МБ × 3) ------------------
    handler = RotatingFileHandler("extract.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[handler, logging.StreamHandler()]
    )
    # ----------------------------------------------------------------

    ssl_ctx = None

    try:
        from .utils import clean_text
    except ImportError:
        clean_text = lambda x: " ".join(x.split())


    async def extract_article_text(
        url: str,
        ssl_context=None,
        max_length: int = 5000,
        session: aiohttp.ClientSession | None = None
    ) -> str:
        ctx = ssl_context if ssl_context is not None else ssl_ctx

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsBot/1.0",
            "Accept-Language": "en-US,en;q=0.9"
        }

        MAX_DOWNLOAD = max(200_000, min(1_000_000, max_length * 200))

        async def _read_limited(resp, max_bytes):
            chunks, size = [], 0
            async for chunk in resp.content.iter_chunked(8192):
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    logging.debug(f"⚠️ HTML truncated at {size} bytes for {url}")
                    break
            try:
                return b"".join(chunks).decode(errors="ignore")
            except Exception:
                return b"".join(chunks).decode("utf-8", errors="ignore")

        html = ""
        backoff = 1
        for attempt in range(1, 4):
            try:
                if session is None:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=headers) as s:
                        async with s.get(url, ssl=ctx) as r:
                            if r.status != 200:
                                logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                                return ""
                            html = await _read_limited(r, MAX_DOWNLOAD)
                else:
                    async with session.get(url, ssl=ctx, headers=headers) as r:
                        if r.status != 200:
                            logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                            return ""
                        html = await _read_limited(r, MAX_DOWNLOAD)
                break
            except Exception as e:
                logging.debug(f"load attempt {attempt} failed for {url}: {e}")
                if attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    logging.warning(f"⚠️ Ошибка загрузки {url}: {e}")
                    return ""

        if not html.strip():
            logging.warning(f"⚠️ Пустой HTML для {url}")
            return ""

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "form"]):
            tag.decompose()

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = clean_text(" ".join(paragraphs))
        logging.debug(f"📝 <p> text length: {len(text)} for {url}")

        if len(text.split()) >= 50:
            out = text[:max_length].rsplit(" ", 1)[0]
            logging.info(f"✅ Returned <p> text ({len(out)} chars) for {url}")
            return out

        loop = asyncio.get_running_loop()

        try:
            import trafilatura
            extracted = await loop.run_in_executor(
                None,
                partial(trafilatura.extract, html, include_comments=False, favor_recall=True)
            )
            logging.debug(f"🧩 trafilatura length: {len(extracted) if extracted else 0} for {url}")
            if extracted and len(extracted.split()) >= 30:
                out = clean_text(extracted)[:max_length].rsplit(" ", 1)[0]
                logging.info(f"✅ Returned trafilatura text ({len(out)} chars) for {url}")
                return out
        except Exception as e:
            logging.debug(f"trafilatura fail: {e}")

        try:
            from readability import Document
            def readability_extract(html_inner):
                doc = Document(html_inner)
                summary_html = doc.summary()
                return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
            extracted = await loop.run_in_executor(None, partial(readability_extract, html))
            logging.debug(f"📖 readability length: {len(extracted) if extracted else 0} for {url}")
            if extracted and len(extracted.split()) >= 30:
                out = clean_text(extracted)[:max_length].rsplit(" ", 1)[0]
                logging.info(f"✅ Returned readability text ({len(out)} chars) for {url}")
                return out
        except Exception as e:
            logging.debug(f"readability fail: {e}")

        meta = (soup.find("meta", attrs={"name": "description"}) or
                soup.find("meta", property="og:description") or
                soup.find("meta", property="twitter:description"))
        if meta and meta.get("content"):
            meta_text = clean_text(meta.get("content", ""))
            logging.debug(f"🪶 meta fallback length: {len(meta_text)} for {url}")
            out = meta_text[:min(max_length, 1000)].rsplit(" ", 1)[0]
            logging.info(f"✅ Returned meta fallback ({len(out)} chars) for {url}")
            return out

        logging.warning(f"⚠️ Не удалось извлечь текст для {url}")
        return ""
