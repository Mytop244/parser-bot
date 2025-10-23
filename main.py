import os, sys, json, time, asyncio, ssl, logging, tempfile, re, html, calendar
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from functools import partial
from dotenv import load_dotenv
import aiohttp, feedparser
import random
import atexit
from bs4 import BeautifulSoup
from telegram import Bot
from telegram.error import RetryAfter, TimedOut, NetworkError

# Windows event loop policy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---- CONFIG / ENV ----
load_dotenv()
STATE_FILE = "state.json"
LEGACY_SEEN = "seen.json"
LEGACY_SENT = "sent_links.json"

SMART_PAUSE = os.getenv("SMART_PAUSE", "0") == "1"
SMART_PAUSE_MIN = int(os.getenv("SMART_PAUSE_MIN", "30"))
SMART_PAUSE_MAX = int(os.getenv("SMART_PAUSE_MAX", "60"))

STATE_DAYS_LIMIT = int(os.getenv("STATE_DAYS_LIMIT", "3"))
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
raw_chat = os.environ.get("CHAT_ID")
CHAT_ID = int(raw_chat) if raw_chat not in (None, "") else None
_env_rss = [u.strip() for u in os.environ.get("RSS_URLS", "").split(",") if u.strip()]
RSS_FILE = os.path.join(os.path.dirname(__file__) or '.', 'rss.txt')
RSS_URLS = _env_rss
if os.path.exists(RSS_FILE):
    try:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            RSS_URLS = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
    except Exception:
        pass

NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))
DAYS_LIMIT = int(os.environ.get("DAYS_LIMIT", 1))
ROUND_ROBIN_MODE = int(os.environ.get("ROUND_ROBIN_MODE", 1))
AI_STUDIO_KEY = os.environ.get("AI_STUDIO_KEY")
GEMINI_MODEL = os.environ.get("AI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_MODEL_FALLBACK = os.environ.get("OLLAMA_MODEL_FALLBACK", "gpt-oss:120b")
PARSER_MAX_TEXT_LENGTH = int(os.environ.get("PARSER_MAX_TEXT_LENGTH",
                                           os.environ.get("MAX_TEXT_LENGTH", "10000")))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 180))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 1200))
GEMINI_PROMPT = os.getenv("GEMINI_PROMPT",
    "Сделай профессиональное краткое резюме новости на русском языке, без вступления, дели на абзацы:\n{content}")
OLLAMA_PROMPT = os.getenv("OLLAMA_PROMPT",
    "Не делай вступлений. Сделай резюме новости на русском языке:\n{content}")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 500))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", 500))
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", GEMINI_MODEL)
BATCH_SIZE_SMALL = int(os.environ.get("BATCH_SIZE_SMALL", 5))
PAUSE_SMALL = int(os.environ.get("PAUSE_SMALL", 3))
BATCH_SIZE_MEDIUM = int(os.environ.get("BATCH_SIZE_MEDIUM", 15))
PAUSE_MEDIUM = int(os.environ.get("PAUSE_MEDIUM", 5))
BATCH_SIZE_LARGE = int(os.environ.get("BATCH_SIZE_LARGE", 25))
PAUSE_LARGE = int(os.environ.get("PAUSE_LARGE", 10))
SINGLE_MESSAGE_PAUSE = int(os.environ.get("SINGLE_MESSAGE_PAUSE", 1))
PARSE_MODE = os.getenv("PARSE_MODE", "HTML")
HEADER_TEMPLATE = os.getenv("HEADER_TEMPLATE",
    "<b>{title}</b>\n📡 <i>{source}</i> | 🗓 {date}\n━━━━━━━━━━━━━━━\n")
FOOTER_TEMPLATE = os.getenv("FOOTER_TEMPLATE",
    "\n━━━━━━━━━━━━━━━\n🤖 <i>Модель: {model}</i>\n🔗 <a href=\"{link}\">Читать статью</a>")
BODY_PREFIX = os.getenv("BODY_PREFIX", "💬 ")
HTML_SAFE_LIMIT = 4096

# Глобальная переменная для адаптивной паузы при ошибках модели
last_error = ""

def set_last_error(val: str):
    """Safely set module-level last_error from inside async functions without using 'global' repeatedly."""
    try:
        globals()['last_error'] = val
    except Exception:
        pass

if hasattr(time, "tzset"):
    os.environ["TZ"] = os.environ.get("TIMEZONE", "UTC")
    time.tzset()

if not TELEGRAM_TOKEN or not CHAT_ID:
    sys.exit("❌ TELEGRAM_TOKEN или CHAT_ID не заданы")
if not RSS_URLS:
    sys.exit("❌ RSS_URLS не заданы")

# --- logging (kept similar) ---
LOG_FILE = "parser.log"
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)
model_logger = logging.getLogger("model")
model_logger.setLevel(logging.INFO)
model_logger.addHandler(console_handler); model_logger.addHandler(file_handler); model_logger.propagate=False

# --- SSL ---
SSL_VERIFY = os.getenv("SSL_VERIFY", "1") not in ("0", "false", "False")
ssl_ctx = ssl.create_default_context()
if not SSL_VERIFY:
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

# --- Пул соединений и глобальная сессия ---
_global_session = None

async def get_session():
    """Создаёт или возвращает общую сессию после запуска event loop"""
    global _global_session
    if _global_session is None or _global_session.closed:
        timeout = aiohttp.ClientTimeout(total=20, connect=5)
        connector = aiohttp.TCPConnector(limit=50, ssl=ssl_ctx, ttl_dns_cache=300)
        _global_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _global_session

# Telegram bot (single instance)
from telegram.request import HTTPXRequest as Request
req = Request(connect_timeout=15, read_timeout=60, write_timeout=60)
bot = Bot(token=TELEGRAM_TOKEN, request=req)

# ---- small cache and helpers ----
_cache = {}

def split_text_safe(text: str, limit: int = HTML_SAFE_LIMIT) -> list[str]:
    parts = []
    while len(text) > limit:
        pos = text.rfind("\n", 0, limit)
        if pos == -1:
            pos = text.rfind(" ", 0, limit)
        if pos == -1:
            pos = limit
        parts.append(text[:pos].strip())
        text = text[pos:].strip()
    if text:
        parts.append(text)
    return parts

def clean_text(text: str) -> str:
    try:
        if "<" in text and ">" in text:
            text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        pass
    return " ".join(text.split())

def parse_iso_utc(s):
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc)
    if not s:
        raise ValueError("empty date")
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = None
        for fmt in ("%d.%m.%Y, %H:%M", "%Y-%m-%d %H:%M",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError(f"Неверный формат даты: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

# ---- state management ----
state = {"seen": {}, "sent": {}}
try:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            state["seen"] = data.get("seen", {}) or {}
            state["sent"] = data.get("sent", {}) or {}
except Exception:
    state = {"seen": {}, "sent": {}}

def save_state_atomic(data, path=STATE_FILE):
    fd, tmp = tempfile.mkstemp(prefix="tmp_state_", dir=os.path.dirname(path) or ".")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

_state_lock = asyncio.Lock()
async def save_state_async():
    import aiofiles
    async with _state_lock:
        now = time.time()
        cutoff = now - STATE_DAYS_LIMIT * 86400
        for k in ("seen","sent"):
            state[k] = {u: ts for u, ts in state.get(k, {}).items() if (isinstance(ts,(int,float)) and ts>=cutoff) or (not isinstance(ts,(int,float)) and (parse_iso_utc(ts).timestamp()>=cutoff if ts else False))}
        fd, tmp_path = tempfile.mkstemp(prefix="state_", suffix=".json", dir=".")
        os.close(fd)
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(state, ensure_ascii=False, indent=2))
            os.replace(tmp_path, STATE_FILE)
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass

def mark_state(kind: str, key: str, value):
    if kind not in state:
        state[kind] = {}
    # normalize value to int timestamp
    ts = None
    if isinstance(value, (int, float)):
        ts = int(value)
    elif isinstance(value, datetime):
        ts = int(value.astimezone(timezone.utc).timestamp())
    elif isinstance(value, str):
        try:
            dt = parse_iso_utc(value)
            ts = int(dt.timestamp())
        except Exception:
            try:
                ts = int(float(value))
            except Exception:
                ts = int(time.time())
    else:
        ts = int(time.time())

    state[kind][key] = ts
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        asyncio.create_task(save_state_async())
    else:
        try:
            save_state_atomic(state, STATE_FILE)
        except Exception:
            try:
                asyncio.run(save_state_async())
            except Exception:
                logging.exception("Не удалось сохранить state при mark_state")
    logging.debug(f"Updated state[{kind}][{key}] = {ts!r}")

# migrate legacy files
def migrate_legacy_files():
    migrated = False
    state_local = {"seen": {}, "sent": {}}
    try:
        if os.path.exists(LEGACY_SEEN):
            with open(LEGACY_SEEN, "r", encoding="utf-8") as f:
                state_local["seen"] = json.load(f)
            migrated = True
        if os.path.exists(LEGACY_SENT):
            with open(LEGACY_SENT, "r", encoding="utf-8") as f:
                state_local["sent"] = json.load(f)
            migrated = True
        if migrated:
            save_state_atomic(state_local, STATE_FILE)
            if os.path.exists(LEGACY_SEEN): os.remove(LEGACY_SEEN)
            if os.path.exists(LEGACY_SENT): os.remove(LEGACY_SENT)
            logging.info("✅ Миграция выполнена")
    except Exception:
        logging.exception("⚠️ Ошибка миграции")

migrate_legacy_files()

# ---- HTTP helpers ----
async def fetch_text_limited(response, max_bytes: int, ctx_url: str = ""):
    """
    Accepts an aiohttp response (streaming) and returns up to max_bytes of decoded text.
    ctx_url used only for logging.
    """
    chunks, size = [], 0
    async for chunk in response.content.iter_chunked(8192):
        chunks.append(chunk)
        size += len(chunk)
        if size >= max_bytes:
            logging.debug(f"⚠️ HTML truncated at {size} bytes for {ctx_url}")
            break
    try:
        return b"".join(chunks).decode(errors="ignore")
    except Exception:
        return b"".join(chunks).decode("utf-8", errors="ignore")

async def fetch_url(session: aiohttp.ClientSession, url: str, head_only=False):
    try:
        headers = {"User-Agent": "NewsBot/1.0"}
        if head_only:
            async with session.head(url, ssl=ssl_ctx, headers=headers) as r:
                if r.status == 405:
                    async with session.get(url, ssl=ssl_ctx, headers=headers) as r2:
                        return url, "✅ OK" if r2.status == 200 else f"⚠️ HTTP {r2.status}"
                return url, "✅ OK" if r.status == 200 else f"⚠️ HTTP {r.status}"
        async with session.get(url, ssl=ssl_ctx, headers=headers) as r:
            if r.status != 200:
                raise Exception(f"HTTP {r.status}")
            body = await r.text()
            return body
    except Exception as e:
        return (url, f"❌ {e.__class__.__name__}") if head_only else None

async def fetch_and_check(session, url, head_only=False):
    res = await fetch_url(session, url, head_only=head_only)
    if head_only:
        return res
    if not res:
        return []
    body = res
    feed = feedparser.parse(body)
    news = []
    for e in feed.entries:
        pub = None
        if getattr(e, "published_parsed", None):
            pub = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=timezone.utc)
        summary = e.get("summary", "") or e.get("description", "") or ""
        news.append((
            e.get("title", "Без заголовка").strip(),
            e.get("link", "").strip(),
            feed.feed.get("title", "Неизвестный источник").strip(),
            summary,
            pub
        ))
    return news

# ---- article extraction (kept behavior, but reuses passed session when available) ----
async def extract_article_text(url: str, ssl_context=None, max_length: int = 5000, session: aiohttp.ClientSession | None = None):
    ctx = ssl_context or ssl.create_default_context()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsBot/1.0", "Accept-Language": "en-US,en;q=0.9"}
    MAX_DOWNLOAD = max(200_000, min(1_000_000, max_length * 200))
    html_text = ""
    backoff = 1
    for attempt in range(1, 4):
        try:
            if session is None:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=headers) as s:
                    async with s.get(url, ssl=ctx) as r:
                        if r.status != 200:
                            logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                            raise Exception(f"HTTP {r.status}")
                        html_text = await fetch_text_limited(r, MAX_DOWNLOAD, url)
            else:
                async with session.get(url, ssl=ctx, headers=headers) as r:
                    if r.status != 200:
                        logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                        raise Exception(f"HTTP {r.status}")
                    html_text = await fetch_text_limited(r, MAX_DOWNLOAD, url)
            break
        except Exception as e:
            logging.debug(f"load attempt {attempt} failed for {url}: {e}")
            if attempt < 3:
                await asyncio.sleep(backoff); backoff *= 2
            else:
                logging.warning(f"⚠️ Ошибка загрузки {url}: {e}")
                return ""
    if not html_text.strip():
        logging.warning(f"⚠️ Пустой HTML для {url}")
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "form"]):
        tag.decompose()
    article = soup.find("article")
    paragraphs = [p.get_text(" ", strip=True) for p in (article.find_all("p") if article else soup.find_all("p"))]
    joined = " ".join(paragraphs).strip()
    text = clean_text(joined)
    logging.debug(f"📝 <p> text length: {len(text)} for {url}")
    if len(text.split()) >= 50:
        out = text[:max_length].rsplit(" ", 1)[0]
        logging.info(f"✅ Returned <p> text ({len(out)} chars) for {url}")
        return out
    loop = asyncio.get_running_loop()
    # trafilatura -> readability fallbacks (same order)
    try:
        import trafilatura
        def trafilatura_extract(html_inner):
            return trafilatura.extract(html_inner, include_comments=False, favor_recall=True)
        extracted = await loop.run_in_executor(None, partial(trafilatura_extract, html_text))
        if extracted and len(extracted.split()) >= 30:
            out = clean_text(extracted)[:max_length]
            logging.info(f"✅ Returned trafilatura text ({len(out)} chars) for {url}")
            return out.rsplit(" ", 1)[0] if " " in out else out
    except Exception as e:
        logging.debug(f"trafilatura fail: {e}")
    try:
        from readability import Document
        def readability_extract(html_inner):
            doc = Document(html_inner)
            summary_html = doc.summary()
            return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
        extracted = await loop.run_in_executor(None, partial(readability_extract, html_text))
        if extracted and len(extracted.split()) >= 30:
            out = clean_text(extracted)[:max_length]
            logging.info(f"✅ Returned readability text ({len(out)} chars) for {url}")
            return out.rsplit(" ", 1)[0] if " " in out else out
    except Exception as e:
        logging.debug(f"readability fail: {e}")
    meta = (soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description") or soup.find("meta", property="twitter:description"))
    if meta and meta.get("content"):
        meta_text = clean_text(meta.get("content", ""))
        out = meta_text[:min(max_length, 1000)].rsplit(" ", 1)[0]
        logging.info(f"✅ Returned meta fallback ({len(out)} chars) for {url}")
        return out
    logging.warning(f"⚠️ Не удалось извлечь текст для {url}")
    return ""


async def parse_html(url, html_text):
    try:
        # Используем lxml-парсер если доступен (быстрее)
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.string.strip() if soup.title else ""
    paragraphs = " ".join(p.get_text(" ", strip=True) for p in soup.select("p"))
    return f"{title}\n\n{paragraphs[:3000]}"


async def feed_to_items(feed_url):
    try:
        loop = asyncio.get_running_loop()
        feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
        return [(getattr(e, 'title', ''), getattr(e, 'link', ''), getattr(e, 'published', None)) for e in feed.entries]
    except Exception as e:
        logging.error(f"Feed parse error {feed_url}: {e}")
        return []


# --- Быстрое закрытие сессии при выходе ---
@atexit.register
def _close_session():
    try:
        global _global_session
        if _global_session and not _global_session.closed:
            asyncio.run(_global_session.close())
    except Exception:
        pass

# ---- Model wrappers (Gemini + Ollama) ----
async def summarize_ollama(text: str):
    global last_error
    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]
    prompt = OLLAMA_PROMPT.format(content=prompt_text)
    logging.info(f"🧠 [OLLAMA INPUT] >>> {prompt_text[:500]}")
    async def run_model(model_name: str):
        global last_error
        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "options": {"num_predict": MODEL_MAX_TOKENS}}
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as resp:
                    if resp.status != 200:
                        logging.error(f"⚠️ Ollama {model_name} HTTP {resp.status}")
                        return None, model_name
                    text_acc = ""
                    try:
                        async for chunk in resp.content:
                            if not chunk: continue
                            try:
                                s = chunk.decode("utf-8")
                            except Exception:
                                continue
                            for line in s.splitlines():
                                line = line.strip()
                                if not line: continue
                                try:
                                    data = json.loads(line)
                                except Exception:
                                    continue
                                text_acc += data.get("response", "")
                    except Exception as e:
                        logging.error(f"❌ Ollama ({model_name}) stream error: {e}")
                        set_last_error(f"Ollama stream error: {e}")
                        return None, model_name
                    output = text_acc.strip()
                    if not output:
                        logging.warning(f"⚠️ Ollama ({model_name}) пустой ответ")
                        return None, model_name
                    elapsed = round(time.time() - start_time, 2)
                    logging.info(f"✅ Ollama ({model_name}) за {elapsed} сек")
                    logging.info(f"🧠 [OLLAMA OUTPUT] <<< {output[:800]}")
                    return output, model_name
        except asyncio.TimeoutError as e:
            logging.error(f"⏰ Ollama ({model_name}) таймаут")
            set_last_error(f"Ollama timeout: {e}")
        except Exception as e:
            logging.error(f"❌ Ollama ({model_name}): {e}")
            set_last_error(f"Ollama error: {e}")
        return None, model_name

    result, used_model = await run_model(OLLAMA_MODEL)
    if not result:
        logging.warning(f"⚠️ Переключаюсь на резервную модель {OLLAMA_MODEL_FALLBACK}")
        result, used_model = await run_model(OLLAMA_MODEL_FALLBACK)
    if not result:
        set_last_error("Ollama no result")
        return prompt_text[:2000] + "...", "local-fallback"
    set_last_error("")
    return result, used_model

async def summarize_gemini(text, max_tokens=200, retries=3):
    global last_error
    text = clean_text(text)
    prompt_text = GEMINI_PROMPT.format(content=text[:PARSER_MAX_TEXT_LENGTH])
    if not AI_STUDIO_KEY:
        logging.debug("⚠️ AI_STUDIO_KEY не задан, fallback на Ollama")
        return await summarize_ollama(text)
    payload = {"contents":[{"parts":[{"text":prompt_text}]}], "generationConfig":{"maxOutputTokens": max_tokens or MODEL_MAX_TOKENS}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": AI_STUDIO_KEY, "Content-Type": "application/json"}
    backoff = 1
    for attempt in range(1, retries+1):
        try:
            logging.info(f"🧠 [GEMINI INPUT] >>> {prompt_text[:500]}")
            session = await get_session()
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status == 429:
                    logging.warning("⚠️ Gemini quota exceeded — fallback to Ollama")
                    return await summarize_ollama(text)
                if resp.status >= 400:
                    logging.warning(f"⚠️ Gemini HTTP {resp.status}: {body}")
                    await asyncio.sleep(backoff); backoff *= 2
                    continue
                result = json.loads(body)
        except Exception as e:
            logging.warning(f"⚠️ Gemini error: {e}")
            set_last_error(f"Gemini error: {e}")
            await asyncio.sleep(backoff); backoff *= 2
            continue
        try:
            candidates = result.get("candidates")
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    text_out = parts[0]["text"]
                    logging.info(f"✅ Gemini OK ({GEMINI_MODEL})")
                    set_last_error("")
                    return text_out.strip(), GEMINI_MODEL
        except Exception as e:
            logging.warning(f"⚠️ Ошибка парсинга Gemini: {e}")
            set_last_error(f"Gemini parse error: {e}")
    logging.error("❌ Gemini не ответил, fallback на Ollama")
    set_last_error("Gemini no response")
    return await summarize_ollama(text)

# ---- sanitization & splitting helpers (centralized) ----
def sanitize_summary(summary_raw: str):
    s = summary_raw or ""
    s = re.sub(r'(?m)^[\s]*[\*\-\u2013]\s+', '• ', s)
    s = re.sub(r'(?m)^[\s]*[\*\-]{2,}\s*$', '', s)
    s = re.sub(r'(?m)^[\s]*\*\s*', '• ', s)
    s = re.sub(r'\*\*([^\n*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*([^\n*]+?)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'__([^_\n]+?)__', r'<u>\1</u>', s)
    s = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', s)
    s = re.sub(r'\[([^\]]+?)\]\((https?://[^\s)]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'[ \t]{2,}', ' ', s).strip()
    try:
        s = html.unescape(s)
    except Exception:
        pass
    # attempt to use bleach if available
    try:
        import importlib
        bleach = importlib.import_module('bleach')
        allowed_tags = ['b','i','code','a','pre','u']
        allowed_attrs = {'a': ['href','title']}
        sanitized = bleach.clean(s, tags=allowed_tags, attributes=allowed_attrs, strip=True)
    except Exception:
        tmp = s.replace("&nbsp;", " ")
        tmp = re.sub(r'&lt;a\s+href=&quot;(https?://[^&quot;]+)&quot;&gt;', r'<a href="\1">', tmp)
        tmp = re.sub(r'&lt;/a&gt;', r'</a>', tmp)
        allowed = ('b','i','u','code','pre','a')
        tmp = re.sub(r'&lt;/?([a-zA-Z0-9]+)[^&]*&gt;', lambda m: f"<{m.group(1)}>" if m.group(1).lower() in allowed else "", tmp)
        sanitized = tmp
    return sanitized

def split_html_preserve(text: str, limit: int = HTML_SAFE_LIMIT - 200):
    parts = []
    i = 0
    L = len(text)
    while i < L:
        j = min(i + limit, L)
        lt = text.rfind('<', i, j)
        gt = text.rfind('>', i, j)
        if lt > gt:
            next_gt = text.find('>', j)
            if next_gt != -1:
                j = next_gt + 1
        parts.append(text[i:j])
        i = j
    return parts

# ---- send helpers ----
async def send_with_retry(chat_id: int, part_msg: str, attempts: int = 3):
    for attempt in range(attempts):
        try:
            await bot.send_message(chat_id=chat_id, text=part_msg, parse_mode="HTML")
            return
        except RetryAfter as e:
            # Bot is rate-limited, sleep for the suggested period then retry
            logging.warning(f"⏳ Rate limited, retry after {getattr(e, 'retry_after', 'unknown')}s: {e}")
            wait = getattr(e, 'retry_after', 5)
            await asyncio.sleep(wait + 1)
        except (TimedOut, NetworkError) as e:
            logging.warning(f"Сетевая ошибка/таймаут: {e}, попытка {attempt+1}")
            await asyncio.sleep(3 * (attempt + 1))
        except Exception as e:
            logging.warning(f"Send attempt {attempt+1} failed: {e}")
            await asyncio.sleep(3 * (attempt + 1))
    logging.error("❌ Не удалось отправить сообщение после всех попыток")

async def send_long_message(bot_instance, chat_id: int, text: str, parse_mode="HTML", delay: int = 1):
    if text in _cache:
        parts = _cache[text]
    else:
        paragraphs = text.split("\n")
        parts, current = [], ""
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            if len(current) + len(para) + 1 < HTML_SAFE_LIMIT:
                current += ("" if not current else "\n") + para
            else:
                if current: parts.append(current)
                if len(para) >= HTML_SAFE_LIMIT:
                    parts.extend(split_text_safe(para, HTML_SAFE_LIMIT))
                    current = ""
                else:
                    current = para
        if current: parts.append(current)
        _cache[text] = parts
    for part in parts:
        logging.info((part[:120] + "…") if len(part) > 120 else part)
        try:
            await bot_instance.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
        except Exception as e:
            logging.error(f"Ошибка при отправке сообщения: {e}")
        await asyncio.sleep(delay)

# ---- main news logic ----
async def send_news(session: aiohttp.ClientSession):
    # load queued news (one-time)
    all_news = []
    if os.path.exists("news_queue.json"):
        try:
            with open("news_queue.json", "r", encoding="utf-8") as f:
                queued = json.load(f)
            for item in queued:
                if len(item) == 4:
                    t, l, s, p = item
                    p_dt = None
                    if isinstance(p, str):
                        try: p_dt = parse_iso_utc(p)
                        except Exception: p_dt = None
                    elif isinstance(p, datetime):
                        p_dt = p.astimezone(timezone.utc) if p.tzinfo else p.replace(tzinfo=timezone.utc)
                    all_news.append((t, l, s, "", p_dt))
                elif len(item) == 5:
                    t, l, s, summary, p = item
                    p_dt = None
                    if isinstance(p, str):
                        try: p_dt = parse_iso_utc(p)
                        except Exception: p_dt = None
                    elif isinstance(p, datetime):
                        p_dt = p.astimezone(timezone.utc) if p.tzinfo else p.replace(tzinfo=timezone.utc)
                    all_news.append((t, l, s, summary, p_dt))
            try: os.remove("news_queue.json")
            except Exception: pass
        except Exception:
            pass

    # fetch feeds in parallel
    results = await asyncio.gather(*[fetch_and_check(session, url) for url in RSS_URLS])
    for r in results:
        if not r: continue
        for it in r:
            if len(it) == 4:
                t, l, s, p = it
                all_news.append((t, l, s, "", p))
            elif len(it) == 5:
                all_news.append(it)

    if not all_news:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_LIMIT)
    sent_links = state.get("sent", {})
    last_index = state.get("meta", {}).get("last_source_index", 0)

    # clean sent_links (keep ones within cutoff)
    clean_sent = {}
    cutoff_ts = cutoff.timestamp()
    for k, v in sent_links.items():
        try:
            if isinstance(v, (int, float)):
                if v >= cutoff_ts: clean_sent[k] = v
            else:
                if parse_iso_utc(v) >= cutoff: clean_sent[k] = v
        except Exception:
            continue
    sent_links = clean_sent

    # prepare ordering (round-robin or by date)
    MIN_DT = datetime.fromtimestamp(0, tz=timezone.utc)
    if ROUND_ROBIN_MODE:
        sources = defaultdict(deque)
        for t, l, s, summary, p in sorted(all_news, key=lambda x: x[4] or MIN_DT, reverse=True):
            sources[s].append((t, l, s, summary, p))
        src_list = list(sources.keys())
        queue = []
        i = last_index
        if src_list:
            while any(sources.values()):
                s = src_list[i % len(src_list)]
                if sources[s]:
                    queue.append(sources[s].popleft())
                i += 1
        new_items = [n for n in queue if n[1] not in sent_links]
    else:
        new_items = [n for n in sorted(all_news, key=lambda x: x[4] or MIN_DT, reverse=True) if n[1] not in sent_links]

    total = len(new_items)
    pause = PAUSE_SMALL if total <= BATCH_SIZE_SMALL else PAUSE_MEDIUM if total <= BATCH_SIZE_MEDIUM else PAUSE_LARGE
    current_batch = new_items[:NEWS_LIMIT or total]
    queue_rest = new_items[NEWS_LIMIT or total:]

    sent_count = 0
    for item in current_batch:
        t, l, s, summary, p = (item if len(item) == 5 else (item[0], item[1], item[2], "", item[3]))
        if l in state.get("seen", {}):
            logging.debug(f"🔁 Пропускаю уже отправленную ссылку: {l}")
            continue
        local_time = (p or datetime.now(timezone.utc)).astimezone(timezone.utc)
        local_time_str = local_time.strftime("%d.%m.%Y, %H:%M")
        try:
            article_text = await extract_article_text(l, ssl_ctx, max_length=PARSER_MAX_TEXT_LENGTH, session=session)
        except Exception as e:
            logging.warning(f"extract_article_text error for {l}: {e}")
            article_text = None

        def is_text_relevant(title: str, text: str, min_words: int = 3) -> bool:
            title_words = [w.lower() for w in re.findall(r'\w+', title)]
            text_lower = (text or "").lower()
            count = sum(1 for w in title_words if w in text_lower)
            return count >= min_words

        content = article_text if article_text and len(article_text) >= 300 and is_text_relevant(t, article_text) else t
        content = content[:PARSER_MAX_TEXT_LENGTH]
        logging.debug(f"📝 Контент для модели ({len(content)} символов)")

        active = (ACTIVE_MODEL or "").lower()
        if "gemini" in active:
            logging.info(f"🧩 Используем GEMINI лимит {GEMINI_MAX_TOKENS} токенов для {ACTIVE_MODEL}")
            summary_text, used_model = await summarize_gemini(content, max_tokens=GEMINI_MAX_TOKENS)
        else:
            logging.info(f"🧩 Используем OLLAMA лимит {OLLAMA_MAX_TOKENS} токенов для {ACTIVE_MODEL}")
            summary_text, used_model = await summarize_ollama(content[:PARSER_MAX_TEXT_LENGTH])

        MAX_TITLE_LEN = 120
        title_clean = t.strip()
        if len(title_clean) > MAX_TITLE_LEN:
            title_clean = title_clean[:MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"
        summary_clean = sanitize_summary((summary_text or "").strip())

        title_safe = html.escape(title_clean)
        summary_safe = summary_clean
        link_safe = html.escape(l, quote=True)

        header = HEADER_TEMPLATE.format(title=title_safe, source=s, date=local_time_str)
        body = f"{BODY_PREFIX}{summary_safe}"
        footer = FOOTER_TEMPLATE.format(model=used_model, link=link_safe)

        body_parts = split_html_preserve(body)
        assembled_parts = []
        for idx, part in enumerate(body_parts):
            if len(body_parts) == 1:
                msg = header + part + footer
            elif idx == 0:
                msg = header + part
            elif idx == len(body_parts) - 1:
                msg = part + footer
            else:
                msg = part
            assembled_parts.append(msg)

        # send
        for attempt in range(3):
            try:
                for part_msg in assembled_parts:
                    await send_with_retry(CHAT_ID, part_msg)
                    await asyncio.sleep(SINGLE_MESSAGE_PAUSE)
                ts_now = int(time.time())
                mark_state("sent", l, ts_now)
                mark_state("seen", l, ts_now)
                sent_count += 1
                logging.info(f"📤 Отправлено: {title_clean[:50]}...")
                break
            except Exception as e:
                logging.error(f"❌ Ошибка отправки: {e}")
                if "429" in str(e):
                    await asyncio.sleep(30)
                else:
                    await asyncio.sleep(5)

        # requeue rest
        if queue_rest:
            safe_queue = []
            for it in queue_rest:
                if len(it) == 5:
                    tt, ll, ss, summ, pp = it
                elif len(it) == 4:
                    tt, ll, ss, pp = it
                    summ = ""
                else:
                    continue
                if isinstance(pp, str):
                    try: pp = parse_iso_utc(pp)
                    except Exception: pass
                iso_p = pp.isoformat() if hasattr(pp, "isoformat") else (None if pp is None else str(pp))
                safe_queue.append((tt, ll, ss, summ, iso_p))
            try:
                with open("news_queue.json", "w", encoding="utf-8") as f:
                    json.dump(safe_queue, f, ensure_ascii=False, indent=2)
            except Exception:
                logging.exception("Не удалось записать news_queue.json")

    # finalize state
    state.setdefault("sent", {})
    # state уже обновлялся через mark_state; удаляем устаревшие записи
    cutoff_ts = cutoff.timestamp() if isinstance(cutoff, datetime) else cutoff
    state["sent"] = {k: v for k, v in state.get("sent", {}).items() if (isinstance(v, (int, float)) and v > cutoff_ts)}
    # добавляем новые отправленные ссылки
    state["sent"].update(sent_links)
    if ROUND_ROBIN_MODE and 'src_list' in locals() and src_list:
        state.setdefault("meta", {})
        state["meta"]["last_source_index"] = (last_index + sent_count) % len(src_list)
    try:
        await save_state_async()
    except Exception:
        logging.warning("⚠️ Не удалось сохранить state.json")

    logging.info(f"✅ Отправлено {sent_count}/{len(current_batch)} новостей. Пауза {pause} сек")
    # ⚡ Умная пауза через .env
    if SMART_PAUSE and sent_count == 0:
        # ensure sensible bounds
        min_p = max(1, SMART_PAUSE_MIN)
        max_p = max(min_p, SMART_PAUSE_MAX)
        base = pause // 4 or min_p
        base = max(min_p, min(base, max_p))
        # add small jitter to avoid thundering herd
        jitter = int(random.uniform(-0.15, 0.15) * base)
        fast_retry = max(min_p, min(max_p, base + jitter))
        logging.info(f"⏩ SMART_PAUSE активна: новых новостей нет, следующая проверка через {fast_retry} сек (base={base}, jitter={jitter})")
        await asyncio.sleep(fast_retry)
    else:
        await asyncio.sleep(pause)

# ---- main loop ----
async def check_sources(urls=None):
    if urls is None:
        urls = RSS_URLS
    session = await get_session()
    tasks = [fetch_url(session, u, head_only=True) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("🔍 Проверка источников:")
    for item in results:
        if isinstance(item, Exception):
            logging.warning(f"Ошибка при проверке источника: {item}")
            continue
        if not item or not isinstance(item, tuple) or len(item) != 2:
            logging.warning(f"Непредвиденный результат при проверке источника: {item}")
            continue
        u, s = item
        logging.info(f"  {s} — {u}")

async def main():
    last_check = datetime.now(timezone.utc)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        try:
            while True:
                now = datetime.now(timezone.utc)
                if (now - last_check) > timedelta(days=1):
                    await check_sources()
                    last_check = now
                logging.info("🔄 Проверка новостей...")
                try:
                    await asyncio.wait_for(send_news(session), timeout=INTERVAL)
                except asyncio.TimeoutError:
                    logging.warning("⏰ send_news превысил лимит времени и был прерван")
                except Exception as e:
                    logging.exception(f"❌ Ошибка в send_news: {e}")
                logging.info(f"⏰ Следующая проверка через {INTERVAL // 60} мин\n")
                print("💤 цикл завершён, жду следующий", flush=True)
                # 💤 адаптивная пауза при ошибках модели
                delay = INTERVAL
                try:
                    if isinstance(last_error, str) and ("Gemini 503" in last_error or "Service Unavailable" in last_error):
                        delay = min(INTERVAL * 3, 300)  # максимум 5 минут
                        logging.warning(f"⚠️ Увеличена пауза до {delay} сек из-за ошибки модели: {last_error}")
                except Exception:
                    pass
                await asyncio.sleep(delay)
        except KeyboardInterrupt:
            logging.info("🛑 Завершение по Ctrl+C, сохраняем state…")
        finally:
            try:
                await save_state_async()
            except Exception as e:
                logging.warning(f"⚠️ Не удалось сохранить state.json при выходе: {e}")

if __name__ == "__main__":
    logging.info(f"💬 MODEL_MAX_TOKENS = {MODEL_MAX_TOKENS}")
    logging.info(f"📰 PARSER_MAX_TEXT_LENGTH = {PARSER_MAX_TEXT_LENGTH}")
    asyncio.run(main())
