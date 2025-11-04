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


# --- Blocked words helper (модульная функция) ---
def is_blocked_article(title: str, text: str, blocked_words: list | None = None) -> bool:
    """Возвращает True, если статья содержит любое запрещённое слово или фразу."""
    bw = blocked_words if blocked_words is not None else BLOCKED_WORDS
    if not bw:
        return False
    combined = f"{title or ''} {text or ''}".casefold()
    for bad in bw:
        b = (bad or "").strip()
        if not b:
            continue
        try:
            # ищем по границам слов
            if re.search(r'\b' + re.escape(b) + r'\b', combined, flags=re.IGNORECASE):
                logging.info(f"🚫 Блокировка статьи по слову: '{b}'")
                return True
        except re.error:
            # fallback — простая подстрока
            if b in combined:
                return True
    return False

# Windows event loop policy
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---- CONFIG / ENV ----
load_dotenv()

# --- concurrency limiter for network calls ---
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))
_network_semaphore = asyncio.Semaphore(CONCURRENCY)

# 🧩 Загружаем список запрещённых слов из .env (case-insensitive, разделитель - запятая)
BLOCKED_WORDS = [w.strip().lower() for w in os.getenv("BLOCKED_WORDS", "").split(",") if w.strip()]

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

# Gemini configuration and keys rotation
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_MODEL_FALLBACK = os.environ.get("OLLAMA_MODEL_FALLBACK", "gpt-oss:120b")
PARSER_MAX_TEXT_LENGTH = int(os.environ.get("PARSER_MAX_TEXT_LENGTH",
                                           os.environ.get("MAX_TEXT_LENGTH", "10000")))
MIN_ARTICLE_WORDS = int(os.environ.get("MIN_ARTICLE_WORDS", "50"))
MIN_TITLE_WORDS = int(os.environ.get("MIN_TITLE_WORDS", "5"))
MIN_TITLE_MATCHES = int(os.environ.get("MIN_TITLE_MATCHES", "3"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 180))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 1200))
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "120"))
GEMINI_PROMPT = os.getenv("GEMINI_PROMPT",
    "Сделай профессиональное краткое резюме новости на русском языке, без вступления, дели на абзацы:\n{content}")
OLLAMA_PROMPT = os.getenv("OLLAMA_PROMPT",
    "Не делай вступлений. Сделай резюме новости на русском языке:\n{content}")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 500))

# --- Gemini key rotation & temporary block support ---
_gemini_key_lock = asyncio.Lock()
GEMINI_BLOCK_MINUTES = int(os.getenv("GEMINI_BLOCK_MINUTES", "10"))
_blocked_keys = {}

def _get_active_keys():
    now = time.time()
    return [k for k in GEMINI_KEYS if k not in _blocked_keys or _blocked_keys.get(k, 0) < now]

def _block_key_temporarily(key: str):
    _blocked_keys[key] = time.time() + GEMINI_BLOCK_MINUTES * 60
    logging.warning(f"🚫 Ключ временно заблокирован на {GEMINI_BLOCK_MINUTES} мин: {key[:8]}…")
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


# --- helper to limit concurrency for network-bound coroutines ---
async def _limited(coro):
    async with _network_semaphore:
        return await coro

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


def is_recent(entry):
    """Возвращает True, если новость опубликована в пределах DAYS_LIMIT дней."""
    try:
        if not getattr(entry, "published_parsed", None):
            return True
        pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        limit_date = datetime.now(timezone.utc) - timedelta(days=DAYS_LIMIT)
        return pub_date >= limit_date
    except Exception:
        return True

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
    try:
        import aiofiles
        have_aiofiles = True
    except Exception:
        have_aiofiles = False

    async with _state_lock:
        now = time.time()
        cutoff = now - STATE_DAYS_LIMIT * 86400
        for k in ("seen","sent"):
            state[k] = {u: ts for u, ts in state.get(k, {}).items() if (isinstance(ts,(int,float)) and ts>=cutoff) or (not isinstance(ts,(int,float)) and (parse_iso_utc(ts).timestamp()>=cutoff if ts else False))}
        fd, tmp_path = tempfile.mkstemp(prefix="state_", suffix=".json", dir=".")
        os.close(fd)
        try:
            if have_aiofiles:
                async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(state, ensure_ascii=False, indent=2))
                os.replace(tmp_path, STATE_FILE)
            else:
                def _sync_write():
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(state, ensure_ascii=False, indent=2))
                    os.replace(tmp_path, STATE_FILE)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _sync_write)
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
        # save no more often than once per 10 сек (debounce)
        last = getattr(mark_state, "_last_save_ts", 0)
        now_ts = time.time()
        if now_ts - last >= 10:
            asyncio.create_task(save_state_async())
            mark_state._last_save_ts = now_ts
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
    state_local = {"seen": {}, "sent": {}}
    migrated = False

    try:
        if os.path.exists(LEGACY_SEEN):
            with open(LEGACY_SEEN, "r", encoding="utf-8") as f:
                try:
                    state_local["seen"] = json.load(f)
                    migrated = True
                except json.JSONDecodeError:
                    logging.exception("Invalid JSON in LEGACY_SEEN")
                    state_local["seen"] = {}
                    migrated = True

        if os.path.exists(LEGACY_SENT):
            with open(LEGACY_SENT, "r", encoding="utf-8") as f:
                state_local["sent"] = json.load(f)
                migrated = True

        if migrated:
            save_state_atomic(state_local, STATE_FILE)
            # обновляем глобальный state, чтобы текущий процесс сразу использовал новые данные
            state.clear()
            state.update(state_local)
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
        return (url, f"❌ {e.__class__.__name__}") if head_only else (url, None)

async def fetch_and_check(session, url, head_only=False):
    logging.info(f"🔍 Проверяю источник: {url}")
    res = await _limited(fetch_url(session, url, head_only=head_only))

    if head_only:
        if res:
            logging.debug(f"✅ Источник доступен: {url}")
        else:
            logging.warning(f"⚠️ Источник не отвечает: {url}")
        return res

    # защита: всегда иметь news, и корректно обрабатывать ошибки загрузки
    news = []

    if not res:
        logging.warning(f"⚠️ Источник не отвечает: {url}")
        return None

    # если fetch_url вернул tuple => это индикатор ошибки/метаданных, не содержимого
    if isinstance(res, tuple):
        # res пример: (url, None) или (url, "❌ Error")
        logging.warning(f"⚠️ Некорректный ответ при загрузке {url}: {res[1] if len(res) > 1 else res[0]}")
        return None

    logging.debug(f"✅ Источник доступен: {url}")
    body = res

    # парсинг feedparser в executor (не блокирует event loop)
    loop = asyncio.get_running_loop()
    try:
        feed = await loop.run_in_executor(None, feedparser.parse, body)
        entries = list(feed.entries)
    except Exception as e:
        logging.error(f"⚠️ Ошибка парсинга RSS {url}: {e}")
        return None

    if not entries:
        logging.debug(f"⚠️ Пустой фид: {url}")
        return None

    # уступаем CPU
    await asyncio.sleep(0)

    old_len = len(entries)
    entries = [e for e in entries if is_recent(e)]
    if old_len != len(entries):
        logging.debug(f"🕓 Отфильтровано {old_len - len(entries)} старых новостей (>{DAYS_LIMIT} дн.)")

    for e in entries:
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

    new_count = sum(1 for _, link, _, _, _ in news if link not in state.get("seen", {}))
    logging.info(f"🆕 Найдено {new_count} новых новостей из {len(news)} в источнике: {url}")
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
            sess = session or await get_session()
            async with sess.get(url, ssl=ctx, headers=headers) as r:
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
        if not _global_session:
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                loop.call_soon_threadsafe(lambda: asyncio.create_task(_global_session.close()))
            except Exception:
                pass
        else:
            try:
                asyncio.run(_global_session.close())
            except Exception:
                try:
                    _global_session.close()
                except Exception:
                    pass
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
            # reuse shared session to avoid recreating many short-lived sessions
            sess = await get_session()
            async with sess.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as resp:
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
        logging.error("🚫 Отклонено: модель local-fallback не используется для публикаций")
        return None, None
    set_last_error("")
    return result, used_model

async def summarize_gemini(text: str, max_tokens: int | None = None):
    text = clean_text(text)
    prompt_text = GEMINI_PROMPT.format(content=text[:PARSER_MAX_TEXT_LENGTH])

    if not GEMINI_KEYS:
        logging.debug("⚠️ GEMINI_KEYS не заданы, fallback на Ollama")
        fallback_text, fallback_model = await summarize_ollama(text)
        if fallback_model is None:
            logging.error("❌ Нет доступных моделей: Gemini отсутствует, Ollama не ответил — сигнал паузы")
            set_last_error("No model available; pause 1h")
            return None, "pause_1h"
        return fallback_text, fallback_model

    # Добавлен лимит токенов (используется значение max_tokens или глобальная константа)
    effective_max = max_tokens if (max_tokens is not None) else GEMINI_MAX_TOKENS
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": int(effective_max)},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    session = await get_session()

    # rotate keys under lock
    async with _gemini_key_lock:
        active = _get_active_keys()
        if not active:
            logging.error("❌ Нет доступных ключей Gemini (все временно заблокированы)")
            fallback_text, fallback_model = await summarize_ollama(text)
            if fallback_model is None:
                logging.error("❌ Нет доступных моделей: Gemini отсутствует, Ollama не ответил — сигнал паузы")
                set_last_error("No model available; pause 1h")
                return None, "pause_1h"
            return fallback_text, fallback_model
        meta = state.setdefault("meta", {})
        idx = int(meta.get("gemini_key_index", 0)) % len(active)
        key_to_use = active[idx]
        meta["gemini_key_index"] = (idx + 1) % len(active)
        asyncio.create_task(save_state_async())

    headers = {"Content-Type": "application/json", "x-goog-api-key": key_to_use}

    total = len(GEMINI_KEYS)
    blocked = len(_blocked_keys)
    logging.info(f"🔑 Активных ключей: {len(active)}/{total}, заблокировано: {blocked}")

    for attempt in range(3):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=MODEL_TIMEOUT)) as resp:
                if resp.status in (403, 429):
                    _block_key_temporarily(key_to_use)
                    raise RuntimeError(f"ключ временно заблокирован ({resp.status})")
                resp.raise_for_status()
                data = await resp.json()
                candidates = data.get("candidates")
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        text_out = parts[0]["text"]
                        logging.info(f"✅ Gemini OK ({GEMINI_MODEL})")
                        set_last_error("")
                        return text_out.strip(), GEMINI_MODEL
                logging.warning("⚠️ Gemini: 200, но нет валидного candidates — retrying")
        except Exception as e:
            logging.warning(f"⚠️ Ошибка Gemini [{key_to_use[:8]}…] попытка {attempt+1}: {e}")
            await asyncio.sleep(3)

    logging.error("❌ Gemini не ответил после 3 попыток")
    fallback_text, fallback_model = await summarize_ollama(text)
    if fallback_model is None:
        logging.error("❌ Gemini не ответил и Ollama тоже — сигнал паузы 1ч")
        set_last_error("No model available; pause 1h")
        return None, "pause_1h"
    return fallback_text, fallback_model

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
            if next_gt != -1 and next_gt - i <= limit * 2:
                j = next_gt + 1
            else:
                j = i + limit
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

    # Фильтрация по дате: не обрабатываем новости старше DAYS_LIMIT дней
    limit_date = datetime.now(timezone.utc) - timedelta(days=DAYS_LIMIT)
    before_count = len(all_news)
    def _is_recent_item(item):
        # item = (title, link, source, summary, pub)
        pub = item[4]
        try:
            if pub is None:
                return True
            if isinstance(pub, datetime):
                return pub >= limit_date
            # strings (ISO) or other -> try parse
            return parse_iso_utc(pub) >= limit_date
        except Exception:
            return True
    all_news = [it for it in all_news if _is_recent_item(it)]
    filtered = before_count - len(all_news)
    if filtered:
        logging.debug(f"🕓 Отфильтровано {filtered} старых новостей (>{DAYS_LIMIT} дн.) из общей очереди")

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
        if l in sent_links or l in state.get("seen", {}):
            logging.debug(f"🔁 Пропускаю уже отправленную ссылку: {l}")
            continue
        local_time = (p or datetime.now(timezone.utc)).astimezone(timezone.utc)
        local_time_str = local_time.strftime("%d.%m.%Y, %H:%M")
        try:
            # limit concurrent article downloads
            article_text = await _limited(extract_article_text(l, ssl_ctx, max_length=PARSER_MAX_TEXT_LENGTH, session=session))
        except Exception as e:
            logging.warning(f"extract_article_text error for {l}: {e}")
            article_text = None

        # 🚫 Пропускаем статью, если она содержит запрещённые слова/фразы
        if is_blocked_article(t, article_text or ""):
            logging.info(f"🚫 Пропущено из-за запрещённых слов: {t}")
            ts_now = int(time.time())
            mark_state("seen", l, ts_now)
            mark_state("sent", l, ts_now)
            sent_links[l] = ts_now
            state.setdefault("seen", {})[l] = ts_now
            continue

        def is_text_relevant(title: str, text: str, min_words: int = 3) -> bool:
            title_words = [w.lower() for w in re.findall(r'\w+', title)]
            text_lower = (text or "").lower()
            count = sum(1 for w in title_words if w in text_lower)
            return count >= min_words


        # ---------- Проверка информативности статьи ----------
        def is_informative(title: str, text: str,
                           min_words: int = MIN_ARTICLE_WORDS,
                           min_title_matches: int = MIN_TITLE_MATCHES) -> bool:
            """
            Эвристика: статья считается информативной, если:
            - содержит >= min_words слов
            - или заголовок встречается в тексте не менее min_title_matches раз
            Отбрасывает технические/пустые записи.
            """
            if not text:
                return False
            words = re.findall(r'\w+', text)
            if len(words) >= min_words:
                return True
            title_words = [w.lower() for w in re.findall(r'\w+', title) if len(w) > 2]
            if not title_words:
                return False
            text_lower = text.lower()
            matches = sum(1 for w in title_words if w in text_lower)
            return matches >= min_title_matches

        # ---------- Выбор контента для модели ----------
        # 1) используем полный текст, если он информативен по эвристике
        # 2) иначе — используем только заголовок, если он не слишком короткий
        # 3) иначе — пропускаем новость и помечаем как просмотренную/отправленную
        use_article = article_text and is_informative(t, article_text)
        if use_article:
            content = article_text[:PARSER_MAX_TEXT_LENGTH]
        else:
            title_word_count = len(re.findall(r'\w+', t))
            if title_word_count >= MIN_TITLE_WORDS:
                content = t
            else:
                logging.info(f"⏭️ Пропускаю неинформативную/очень короткую новость: {l}")
                ts_now = int(time.time())
                mark_state("seen", l, ts_now)
                mark_state("sent", l, ts_now)
                sent_links[l] = ts_now
                state.setdefault("seen", {})[l] = ts_now
                continue

        content = content[:PARSER_MAX_TEXT_LENGTH]
        logging.debug(f"📝 Контент для модели ({len(content)} символов)")

        active = (ACTIVE_MODEL or "").lower()
        if "gemini" in active:
            logging.info(f"🧩 Используем GEMINI лимит {GEMINI_MAX_TOKENS} токенов для {ACTIVE_MODEL}")
            summary_text, used_model = await summarize_gemini(content, max_tokens=GEMINI_MAX_TOKENS)
        else:
            logging.info(f"🧩 Используем OLLAMA лимит {OLLAMA_MAX_TOKENS} токенов для {ACTIVE_MODEL}")
            summary_text, used_model = await summarize_ollama(content[:PARSER_MAX_TEXT_LENGTH])

        # 🕒 если моделей нет — отмечаем паузу 1 час и выходим из send_news
        if used_model == "pause_1h" or (summary_text is None and used_model in (None, "pause_1h")):
            logging.warning("⏸️ Нет доступных моделей (Gemini заблокирован и Ollama не отвечает). Устанавливаю паузу 1 час и выхожу.")
            state.setdefault("meta", {})["pause_until"] = int(time.time() + 3600)
            try:
                await save_state_async()
            except Exception:
                logging.debug("Не удалось сразу сохранить state при установке pause_until")
            return

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
                # сразу фиксируем в локальной памяти, чтобы не было дублей
                sent_links[l] = ts_now
                state.setdefault("seen", {})[l] = ts_now
                try:
                    await save_state_async()
                except Exception:
                    logging.warning("Не удалось сразу сохранить state после отправки")
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
    # Очистка state["seen"] по cutoff, чтобы не рос бесконечно
    clean_seen = {}
    for k, v in state.get("seen", {}).items():
        try:
            if isinstance(v, (int, float)):
                if v >= cutoff_ts:
                    clean_seen[k] = v
            else:
                if parse_iso_utc(v).timestamp() >= cutoff_ts:
                    clean_seen[k] = v
        except Exception:
            continue
    state["seen"] = clean_seen
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
                logging.info(f"⏰ Следующая проверка через {INTERVAL // 60} мин (или пауза по state.meta)")
                print("💤 цикл завершён, жду следующий", flush=True)
                # Сначала проверяем, не установлена ли глобальная пауза от send_news
                pause_until = state.get("meta", {}).get("pause_until")
                if pause_until:
                    try:
                        remaining = int(pause_until - time.time())
                    except Exception:
                        remaining = 0
                    if remaining > 0:
                        dt = datetime.fromtimestamp(pause_until, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
                        logging.info(f"⏸️ Глобальная пауза активна до {dt} (осталось {remaining} сек).")
                        await asyncio.sleep(remaining)
                    # очистим метку паузы и сохраним state
                    try:
                        state.setdefault("meta", {}).pop("pause_until", None)
                        await save_state_async()
                    except Exception:
                        logging.debug("Не удалось сохранить state после снятия pause_until")
                    # после ожидания продолжаем следующий цикл сразу
                    continue
                # 💤 адаптивная пауза при ошибках модели (обычный путь)
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
