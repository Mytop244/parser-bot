import os
import asyncio
import logging
import aiohttp
import feedparser
import time
import tempfile

RSS_FILE = "rss.txt"
MAX_CONCURRENCY = 10
RETRIES = 3
BASE_TIMEOUT = 10  # seconds

logger = logging.getLogger(__name__)

async def fetch_text(session, url, timeout):
    for attempt in range(1, RETRIES + 1):
        try:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.debug("Non-200 %s -> %s", resp.status, url)
                    return None
                return await resp.text()
        except asyncio.TimeoutError:
            logger.debug("Timeout (%d) for %s", attempt, url)
        except Exception as e:
            logger.debug("Error (%d) for %s: %s", attempt, url, e)
        # exponential backoff
        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
    return None

async def check_feed(session, url, sem):
    url = url.strip()
    if not url:
        return url, None
    async with sem:
        text = await fetch_text(session, url, timeout=aiohttp.ClientTimeout(total=BASE_TIMEOUT))
        if not text:
            return url, False
        try:
            feed = feedparser.parse(text)
            ok = bool(getattr(feed, "entries", []))
            return url, ok
        except Exception as e:
            logger.debug("Feed parse error for %s: %s", url, e)
            return url, False

def mark_line(line, ok):
    if not line or line.strip().startswith("#"):
        return line
    if ok:
        return line
    return "# " + line

async def main():
    if not os.path.exists(RSS_FILE):
        logger.error("Файл %s не найден.", RSS_FILE)
        return

    with open(RSS_FILE, "r", encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f.readlines()]

    # --- Удаление дубликатов, сохраняя порядок ---
    seen_urls = set()
    unique_lines = []
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            unique_lines.append(line)
            continue
        if s not in seen_urls:
            seen_urls.add(s)
            unique_lines.append(line)
        else:
            logging.info("🧹 Удалён дубликат: %s", s)
    raw_lines = unique_lines


    # build list of URLs preserving blank/comment lines and positions
    urls = []
    index_map = []
    for idx, l in enumerate(raw_lines):
        s = l.strip()
        if not s or s.startswith("#"):
            index_map.append(None)  # keep as-is
        else:
            urls.append(s)
            index_map.append(s)

    headers = {"User-Agent": "rss-checker/1.0 (+https://example.org)"}
    timeout = aiohttp.ClientTimeout(total=BASE_TIMEOUT)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        tasks = [asyncio.create_task(check_feed(session, u, sem)) for u in urls]
        results = await asyncio.gather(*tasks)

    # map results url->ok
    results_map = {u: ok for u, ok in results}

    new_lines = []
    url_iter = iter(urls)
    for orig in raw_lines:
        s = orig.strip()
        if not s or s.startswith("#"):
            new_lines.append(orig)
            continue

        try:
            u = next(url_iter)
        except StopIteration:
            new_lines.append(orig)
            continue

        ok = results_map.get(u, False)
        mark = "✅" if ok else "❌"
        logger.info("%s %s", mark, u)

        # если строка уже закомментирована и снова ок — убираем #
        if ok and orig.lstrip().startswith("# "):
            new_lines.append(orig.lstrip("# ").strip())
        elif not ok and not orig.lstrip().startswith("# "):
            new_lines.append("# " + orig)
        else:
            new_lines.append(orig)


    # atomic write
    dirn = os.path.dirname(os.path.abspath(RSS_FILE))
    fd, tmp_path = tempfile.mkstemp(dir=dirn, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            tf.write("\n".join(new_lines) + "\n")
        os.replace(tmp_path, RSS_FILE)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
