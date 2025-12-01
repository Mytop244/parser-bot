import os, sys, json, time, asyncio, ssl, logging, tempfile, re, html, calendar, shutil
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict, deque
from functools import partial
from dotenv import load_dotenv
import aiohttp, feedparser
import random
import atexit
from bs4 import BeautifulSoup
from telegram import Bot
from telegram.error import RetryAfter, TimedOut, NetworkError
from telegram.request import HTTPXRequest as Request
import aiosqlite
from logging.handlers import RotatingFileHandler

# --- Windows event loop policy ---
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BASE_DIR = os.path.dirname(
    sys.executable if getattr(sys, 'frozen', False) else __file__
)

def fix_path(name: str) -> str:
    return os.path.join(BASE_DIR, name)

# ---- CONFIG / ENV ----
load_dotenv()

# --- Logging Setup (Rotating) ---
LOG_FILE = fix_path("parser.log")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# Ротация логов: макс 5 МБ, хранить 3 файла
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# --- Config Variables ---
CONCURRENCY = int(os.getenv("CONCURRENCY", "10"))
_network_semaphore = asyncio.Semaphore(CONCURRENCY)

BLOCKED_WORDS = [w.strip().lower() for w in os.getenv("BLOCKED_WORDS", "").split(",") if w.strip()]

DB_PATH = fix_path("bot_history.db")
META_FILE = fix_path("bot_meta.json")
STATE_JSON_PATH = fix_path("state.json") # Для миграции

SMART_PAUSE = os.getenv("SMART_PAUSE", "0") == "1"
SMART_PAUSE_MIN = int(os.getenv("SMART_PAUSE_MIN", "30"))
SMART_PAUSE_MAX = int(os.getenv("SMART_PAUSE_MAX", "60"))

STATE_DAYS_LIMIT = int(os.getenv("STATE_DAYS_LIMIT", "7"))
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
raw_chat = os.environ.get("CHAT_ID")
CHAT_ID = int(raw_chat) if raw_chat not in (None, "") else None

_env_rss = [u.strip() for u in os.environ.get("RSS_URLS", "").split(",") if u.strip()]
RSS_FILE = fix_path("rss.txt")  # Используем fix_path, как для логов и БД
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

GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_MODEL_FALLBACK = os.environ.get("OLLAMA_MODEL_FALLBACK", "gpt-oss:120b")
PARSER_MAX_TEXT_LENGTH = int(os.environ.get("PARSER_MAX_TEXT_LENGTH", "10000"))
MIN_ARTICLE_WORDS = int(os.environ.get("MIN_ARTICLE_WORDS", "50"))
MIN_TITLE_WORDS = int(os.environ.get("MIN_TITLE_WORDS", "5"))
MIN_TITLE_MATCHES = int(os.environ.get("MIN_TITLE_MATCHES", "3"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 180))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 1200))
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "120"))
GEMINI_PROMPT = os.getenv("GEMINI_PROMPT", "Сделай профессиональное краткое резюме новости на русском языке, без вступления, дели на абзацы:\n{content}")
OLLAMA_PROMPT = os.getenv("OLLAMA_PROMPT", "Не делай вступлений. Сделай резюме новости на русском языке:\n{content}")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 500))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 3))

GEMINI_BLOCK_MINUTES = int(os.getenv("GEMINI_BLOCK_MINUTES", "10"))
_gemini_key_lock = asyncio.Lock()
_blocked_keys = {}

OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", 500))
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", GEMINI_MODEL)

BATCH_SIZE_SMALL = int(os.environ.get("BATCH_SIZE_SMALL", 5))
PAUSE_SMALL = int(os.environ.get("PAUSE_SMALL", 3))
BATCH_SIZE_MEDIUM = int(os.environ.get("BATCH_SIZE_MEDIUM", 15))
PAUSE_MEDIUM = int(os.environ.get("PAUSE_MEDIUM", 5))
BATCH_SIZE_LARGE = int(os.environ.get("BATCH_SIZE_LARGE", 25))
PAUSE_LARGE = int(os.environ.get("PAUSE_LARGE", 10))
SINGLE_MESSAGE_PAUSE = int(os.environ.get("SINGLE_MESSAGE_PAUSE", 1))

HEADER_TEMPLATE = os.getenv("HEADER_TEMPLATE", "<b>{title}</b>\n📡 <i>{source}</i> | 🗓 {date}\n━━━━━━━━━━━━━━━\n")
FOOTER_TEMPLATE = os.getenv("FOOTER_TEMPLATE", "\n━━━━━━━━━━━━━━━\n🤖 <i>Модель: {model}</i>\n🔗 <a href=\"{link}\">Читать статью</a>")
BODY_PREFIX = os.getenv("BODY_PREFIX", "💬 ")
HTML_SAFE_LIMIT = 4096

APP_TZ_NAME = os.getenv("TIMEZONE", "UTC")
try:
    APP_TZ = ZoneInfo(APP_TZ_NAME)
except Exception:
    APP_TZ = timezone.utc

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise RuntimeError("❌ TELEGRAM_TOKEN или CHAT_ID не заданы.")
if not RSS_URLS:
    raise RuntimeError("❌ RSS_URLS не заданы.")

# --- SSL ---
SSL_VERIFY = os.getenv("SSL_VERIFY", "1") not in ("0", "false", "False")
ssl_ctx = ssl.create_default_context()
if not SSL_VERIFY:
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

# --- GLOBAL CLASSES (DB & Meta) ---

class Database:
    def __init__(self, path):
        self.path = path
        self.conn = None

    async def connect(self):
        """Подключение к SQLite и создание таблиц"""
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                url TEXT NOT NULL,
                kind TEXT NOT NULL,
                timestamp INTEGER,
                PRIMARY KEY (url, kind)
            )
        """)
        await self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON history (timestamp);")
        await self.conn.commit()
        await self._migrate_legacy()

    async def close(self):
        if self.conn:
            await self.conn.close()

    async def _migrate_legacy(self):
        """Миграция из старого state.json"""
        if not os.path.exists(STATE_JSON_PATH): return
        logging.info("🔄 Миграция старого state.json в SQLite...")
        try:
            with open(STATE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            async with self.conn.cursor() as cur:
                for kind in ["seen", "sent"]:
                    items = data.get(kind, {})
                    if isinstance(items, dict):
                        for url, ts in items.items():
                            try:
                                ts_val = int(ts) if isinstance(ts, (int, float)) else int(time.time())
                            except: ts_val = int(time.time())
                            await cur.execute(
                                "INSERT OR IGNORE INTO history (url, kind, timestamp) VALUES (?, ?, ?)", 
                                (url, kind, ts_val)
                            )
            await self.conn.commit()
            backup_name = STATE_JSON_PATH + ".bak"
            shutil.move(STATE_JSON_PATH, backup_name)
            logging.info(f"✅ Миграция завершена. state.json -> {backup_name}")
        except Exception as e:
            logging.error(f"❌ Ошибка миграции: {e}")

    async def exists(self, kind: str, url: str) -> bool:
        """Проверка наличия записи"""
        if not self.conn: return False
        async with self.conn.execute("SELECT 1 FROM history WHERE url=? AND kind=?", (url, kind)) as cur:
            return await cur.fetchone() is not None

    async def add(self, kind: str, url: str, ts: int = None):
        """Добавление записи"""
        if not self.conn: return
        if ts is None: ts = int(time.time())
        await self.conn.execute(
            "INSERT OR REPLACE INTO history (url, kind, timestamp) VALUES (?, ?, ?)", 
            (url, kind, int(ts))
        )
        await self.conn.commit()

    async def cleanup(self, days: int):
        """Очистка старых записей"""
        if not self.conn: return
        cutoff = int(time.time() - (days * 86400))
        await self.conn.execute("DELETE FROM history WHERE timestamp < ?", (cutoff,))
        await self.conn.commit()
        logging.info(f"🧹 База данных очищена (записи старше {days} дней)")

class MetaManager:
    """Управление настройками (ключи, паузы) через JSON"""
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f: 
                    self.data = json.load(f)
            except: self.data = {}
        else: self.data = {}

    def save(self):
        try:
            fd, tmp = tempfile.mkstemp(prefix="meta_", dir=os.path.dirname(self.path))
            os.close(fd)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            logging.error(f"Meta save error: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

# --- INSTANCES ---
db = Database(DB_PATH)
meta_mgr = MetaManager(META_FILE)
_global_session = None
last_error = ""

# --- HELPERS ---

def set_last_error(val: str):
    global last_error
    last_error = val

async def get_session():
    global _global_session
    if _global_session is None or _global_session.closed:
        timeout = aiohttp.ClientTimeout(total=20, connect=5)
        connector = aiohttp.TCPConnector(limit=50, ssl=ssl_ctx, ttl_dns_cache=300)
        _global_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _global_session

async def _limited(coro):
    async with _network_semaphore:
        return await coro

req = Request(connect_timeout=15, read_timeout=60, write_timeout=60)
bot = Bot(token=TELEGRAM_TOKEN, request=req)
_cache = {}

def split_text_safe(text: str, limit: int = HTML_SAFE_LIMIT) -> list[str]:
    parts = []
    while len(text) > limit:
        pos = text.rfind("\n", 0, limit)
        if pos == -1: pos = text.rfind(" ", 0, limit)
        if pos == -1: pos = limit
        parts.append(text[:pos].strip())
        text = text[pos:].strip()
    if text: parts.append(text)
    return parts

def clean_text(text: str) -> str:
    try:
        if "<" in text and ">" in text:
            text = BeautifulSoup(text, "html.parser").get_text()
    except Exception: pass
    return " ".join(text.split())

def parse_iso_utc(s):
    if isinstance(s, datetime): return s.astimezone(APP_TZ)
    if not s: raise ValueError("empty date")
    s = s.strip()
    if s.endswith("Z"): s = s[:-1] + "+00:00"
    try: dt = datetime.fromisoformat(s)
    except:
        dt = None
        for fmt in ("%d.%m.%Y, %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S%z"):
            try: dt = datetime.strptime(s, fmt); break
            except ValueError: continue
        if dt is None: raise ValueError(f"Invalid date: {s}")
    return dt.astimezone(APP_TZ) if dt.tzinfo else dt.replace(tzinfo=APP_TZ)

def is_blocked_article(title: str, text: str, blocked_words: list | None = None) -> bool:
    bw = blocked_words if blocked_words is not None else BLOCKED_WORDS
    if not bw: return False
    combined = f"{title or ''} {text or ''}".casefold()
    for bad in bw:
        if not bad: continue
        try:
            if re.search(r'\b' + re.escape(bad) + r'\b', combined, flags=re.IGNORECASE):
                logging.info(f"🚫 Blocked by word: '{bad}'")
                return True
        except re.error:
            if bad in combined: return True
    return False

def validate_content_relevance(title: str, text: str) -> bool:
    """
    Проверяет, соответствует ли текст заголовку.
    Возвращает True, если найдено пересечение ключевых слов.
    """
    if not text or len(text) < 50:
        return False

    # Разбиваем на слова, убираем короткие (предлоги и т.д.)
    def get_words(s):
        return set(w.lower() for w in re.findall(r'\w{4,}', s))

    title_words = get_words(title)
    text_words = get_words(text)

    if not title_words:
        return True # Если заголовок слишком короткий, пропускаем проверку

    # Ищем общие слова. Если нет ни одного общего слова >3 букв, это левый текст
    return bool(title_words.intersection(text_words))

# --- NETWORK / PARSING ---

async def fetch_text_limited(response, max_bytes: int, ctx_url: str = ""):
    chunks, size = [], 0
    async for chunk in response.content.iter_chunked(8192):
        chunks.append(chunk)
        size += len(chunk)
        if size >= max_bytes: break
    try: return b"".join(chunks).decode(errors="ignore")
    except: return b"".join(chunks).decode("utf-8", errors="ignore")

async def fetch_url(session, url, head_only=False):
    try:
        headers = {"User-Agent": "NewsBot/1.0"}
        if head_only:
            async with session.head(url, ssl=ssl_ctx, headers=headers) as r:
                return url, "✅ OK" if r.status == 200 else f"⚠️ HTTP {r.status}"
        async with session.get(url, ssl=ssl_ctx, headers=headers) as r:
            if r.status != 200: raise Exception(f"HTTP {r.status}")
            return await r.text()
    except Exception as e:
        return (url, f"❌ {e.__class__.__name__}") if head_only else (url, None)

async def fetch_and_check(session, url):
    logging.info(f"🔍 Checking: {url}")
    res = await _limited(fetch_url(session, url))
    if isinstance(res, tuple) or not res:
        logging.warning(f"⚠️ Source failed: {url}")
        return None

    loop = asyncio.get_running_loop()
    try:
        feed = await loop.run_in_executor(None, feedparser.parse, res)
        entries = list(feed.entries)
    except Exception: return None

    if not entries: return None
    
    # Фильтрация по дате (DAYS_LIMIT)
    limit_date = datetime.now(APP_TZ) - timedelta(days=DAYS_LIMIT)
    news = []
    
    for e in entries:
        try:
            if getattr(e, "published_parsed", None):
                pub = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=APP_TZ)
                if pub < limit_date: continue
            else:
                pub = None # Если даты нет, считаем свежей (или игнорим, по желанию)
                
            summary = e.get("summary", "") or e.get("description", "") or ""
            news.append((
                e.get("title", "Без заголовка").strip(),
                e.get("link", "").strip(),
                feed.feed.get("title", "Неизвестный источник").strip(),
                summary,
                pub
            ))
        except Exception: continue
        
    return news

async def extract_article_text(url: str, ssl_context=None, max_length: int = 5000, session: aiohttp.ClientSession | None = None):
    ctx = ssl_context or ssl.create_default_context()
    headers = {"User-Agent": "Mozilla/5.0 NewsBot/1.0"}
    try:
        sess = session or await get_session()
        async with sess.get(url, ssl=ctx, headers=headers) as r:
            if r.status != 200: return ""
            html_text = await fetch_text_limited(r, 500_000, url)
    except: return ""

    if not html_text.strip(): return ""
    
    # Simple extraction logic
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "form"]):
        tag.decompose()
        
    article = soup.find("article") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    text = " ".join(paragraphs).strip()
    
    if len(text.split()) >= 30:
        return text[:max_length].rsplit(" ", 1)[0]
    
    # Fallback: meta description
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
    if meta and meta.get("content"):
        return meta["content"][:max_length]
        
    return ""

# --- MODEL WRAPPERS ---

def _get_active_keys():
    now = time.time()
    return [k for k in GEMINI_KEYS if k not in _blocked_keys or _blocked_keys.get(k, 0) < now]

def _block_key_temporarily(key: str):
    _blocked_keys[key] = time.time() + GEMINI_BLOCK_MINUTES * 60

async def summarize_ollama(text: str):
    prompt = OLLAMA_PROMPT.format(content=text[:PARSER_MAX_TEXT_LENGTH])
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": MODEL_MAX_TOKENS}}
    
    try:
        sess = await get_session()
        async with sess.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=OLLAMA_TIMEOUT) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("response", "").strip(), OLLAMA_MODEL
    except Exception as e:
        logging.error(f"Ollama error: {e}")
    return None, None

async def summarize_gemini(text: str, max_tokens: int | None = None):
    text = clean_text(text)
    prompt_text = GEMINI_PROMPT.format(content=text[:PARSER_MAX_TEXT_LENGTH])
    
    if not GEMINI_KEYS:
        return await summarize_ollama(text)

    eff_max = max_tokens or GEMINI_MAX_TOKENS
        
    # Системная инструкция для подавления "болтливости" модели
    sys_instr = "Ты — API для генерации JSON/HTML. Твоя единственная задача — вернуть текст резюме. ЗАПРЕЩЕНО: вступления, приветствия, фразы 'Вот резюме', 'В статье'. Начинай сразу с первого предложения новости."

    payload = {
        "system_instruction": {"parts": [{"text": sys_instr}]},
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": int(eff_max)},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    session = await get_session()

    attempts = 0
    while attempts < MAX_ATTEMPTS:
        async with _gemini_key_lock:
            active = _get_active_keys()
            if not active:
                logging.warning("Все ключи Gemini заблокированы, пробую Ollama...")
                return await summarize_ollama(text)
            
            # Ротация через MetaManager
            idx = int(meta_mgr.get("gemini_key_index", 0)) % len(active)
            key_to_use = active[idx]
            meta_mgr.set("gemini_key_index", (idx + 1) % len(active))

        try:
            async with session.post(url, headers={"Content-Type": "application/json", "x-goog-api-key": key_to_use}, json=payload, timeout=MODEL_TIMEOUT) as resp:
                if resp.status in (403, 429):
                    _block_key_temporarily(key_to_use)
                    attempts += 1
                    continue
                resp.raise_for_status()
                data = await resp.json()
                try:
                    res = data["candidates"][0]["content"]["parts"][0]["text"]
                    return res.strip(), GEMINI_MODEL
                except: return None, GEMINI_MODEL
        except Exception as e:
            logging.warning(f"Gemini error: {e}")
            attempts += 1
            await asyncio.sleep(2)
            
    return await summarize_ollama(text)

# --- UTILS ---

def sanitize_summary(s: str):
    # Убираем мусорные вступления
    garbage = [
        r'^(?:конечно[,:]?|вот|держи|итак[,:]?)\s*',
        r'^(?:вот|ниже)?\s*(?:краткое)?\s*резюме[:\.]?\s*',
        r'^в\s+статье\s+(?:говорится|рассказывается|речь идет)\s+(?:о|том, что)?\s*',
    ]
    for g in garbage:
        s = re.sub(g, '', s, flags=re.IGNORECASE | re.MULTILINE).strip()
    if not s: return ""
    s = re.sub(r'(?m)^[\s]*[\*\-\u2013]\s+', '• ', s)
    s = re.sub(r'\*\*([^\n*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*([^\n*]+?)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', s)
    s = re.sub(r'\[([^\]]+?)\]\((https?://[^\s)]+)\)', r'<a href="\2">\1</a>', s)
    return s.strip()

def split_html_preserve(text: str, limit: int = HTML_SAFE_LIMIT - 200):
    parts, i, L = [], 0, len(text)
    while i < L:
        j = min(i + limit, L)
        lt, gt = text.rfind('<', i, j), text.rfind('>', i, j)
        if lt > gt: 
            next_gt = text.find('>', j)
            j = next_gt + 1 if next_gt != -1 and next_gt - i <= limit * 2 else i + limit
        parts.append(text[i:j])
        i = j
    return parts

async def send_with_retry(chat_id, text):
    for attempt in range(3):
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            return
        except RetryAfter as e:
            await asyncio.sleep(getattr(e, 'retry_after', 5) + 1)
        except Exception as e:
            logging.warning(f"Send failed ({attempt}): {e}")
            await asyncio.sleep(5)

# --- MAIN LOGIC ---

async def send_news(session: aiohttp.ClientSession):
    # 1. Загрузка очереди (из файла, для совместимости/повторов)
    all_news = []
    if os.path.exists("news_queue.json"):
        try:
            with open("news_queue.json", "r", encoding="utf-8") as f:
                queued = json.load(f)
            # Упрощенная загрузка, считаем что формат соблюден
            for item in queued:
                if len(item) >= 4:
                    # Восстанавливаем datetime
                    item_list = list(item)
                    if isinstance(item_list[-1], str):
                         try: item_list[-1] = parse_iso_utc(item_list[-1])
                         except: pass
                    all_news.append(tuple(item_list))
        except: pass

    # 2. Скачивание RSS
    tasks = [fetch_and_check(session, url) for url in RSS_URLS]
    results = await asyncio.gather(*tasks)
    for r in results:
        if r: all_news.extend(r)

    if not all_news: return

    # 3. Фильтрация и дедупликация через DB
    unique_news = []
    for item in all_news:
        link = item[1]
        # Проверяем в БД: была ли отправлена или просмотрена
        if await db.exists("sent", link) or await db.exists("seen", link):
            continue
        unique_news.append(item)
    
    # Сортировка по дате (новые сверху)
    unique_news.sort(key=lambda x: x[4] or datetime.min.replace(tzinfo=APP_TZ), reverse=True)
    
    current_batch = unique_news[:NEWS_LIMIT]
    queue_rest = unique_news[NEWS_LIMIT:]
    sent_count = 0

    # 4. Обработка
    for item in current_batch:
        # Нормализация кортежа
        if len(item) == 5: t, l, s, summary_raw, p = item
        else: t, l, s, p = item; summary_raw = ""

        # Финальная проверка перед отправкой (на случай гонки)
        if await db.exists("sent", l): continue

        logging.info(f"⚙️ Обработка: {t}")
        
        # Скачивание
        article_text = await _limited(extract_article_text(l, ssl_ctx, max_length=PARSER_MAX_TEXT_LENGTH, session=session))

        # --- ВАЛИДАЦИЯ И ВЫБОР КОНТЕНТА ---
        clean_rss_summary = clean_text(summary_raw)
        is_relevant = validate_content_relevance(t, article_text)
        
        if is_relevant and len(re.findall(r'\w+', article_text)) >= MIN_ARTICLE_WORDS:
            # Текст прошел проверку и он достаточно длинный
            content = article_text
        elif clean_rss_summary and len(clean_rss_summary) > 20:
            # Текст плохой/левый, используем описание из RSS
            logging.info(f"⚠️ Текст не прошел проверку. Fallback to RSS summary: {l}")
            content = clean_rss_summary
        else:
            # Совсем ничего нет, берем заголовок
            content = t



        # Проверка на запрещенные слова
        if is_blocked_article(t, content):
            await db.add("seen", l)
            continue

        # LLM Summary
        content = content[:PARSER_MAX_TEXT_LENGTH]
        active_lower = (ACTIVE_MODEL or "").lower()
        
        if "gemini" in active_lower:
            summ_text, used_model = await summarize_gemini(content)
        else:
            summ_text, used_model = await summarize_ollama(content)

        # Обработка паузы/ошибки модели
        if used_model == "pause_1h" or (not summ_text and used_model is None):
            logging.warning("⏸️ Нет модели, пауза 1 час.")
            meta_mgr.set("pause_until", int(time.time() + 3600))
            break # Прерываем батч

        # Формирование сообщения
        local_time_str = (p or datetime.now(APP_TZ)).astimezone(APP_TZ).strftime("%d.%m.%Y, %H:%M")
        msg = (HEADER_TEMPLATE.format(title=html.escape(t.strip()), source=s, date=local_time_str) +
               BODY_PREFIX + sanitize_summary(summ_text or "") +
               FOOTER_TEMPLATE.format(model=used_model, link=html.escape(l, quote=True)))

        # Отправка
        parts = split_html_preserve(msg)
        try:
            for part in parts:
                await send_with_retry(CHAT_ID, part)
                await asyncio.sleep(SINGLE_MESSAGE_PAUSE)
            
            # Успех -> в базу
            ts = int(time.time())
            await db.add("sent", l, ts)
            await db.add("seen", l, ts)
            sent_count += 1
            logging.info(f"📤 Отправлено: {t[:40]}...")
            
        except Exception as e:
            logging.error(f"❌ Ошибка отправки: {e}")

    # 5. Сохранение остатка очереди (queue_rest)
    if queue_rest:
        safe_queue = []
        for item in queue_rest:
            lst = list(item)
            # Date -> ISO string для json
            if isinstance(lst[-1], datetime): lst[-1] = lst[-1].isoformat()
            safe_queue.append(lst)
        try:
            with open("news_queue.json", "w", encoding="utf-8") as f:
                json.dump(safe_queue, f, ensure_ascii=False)
        except: pass

    # Паузы
    if sent_count > 0:
        await asyncio.sleep(PAUSE_MEDIUM)
    elif SMART_PAUSE:
        # Умная пауза с джиттером
        base = max(SMART_PAUSE_MIN, min(SMART_PAUSE_MAX, PAUSE_SMALL))
        wait = base + random.uniform(-2, 2)
        logging.info(f"💤 Smart Pause: {wait:.1f}s")
        await asyncio.sleep(wait)

async def check_sources():
    session = await get_session()
    for u in RSS_URLS:
        await fetch_url(session, u, head_only=True)

async def main():
    # 1. Инициализация БД
    await db.connect()
    # Чистим старую историю (например, >7 дней)
    await db.cleanup(STATE_DAYS_LIMIT)
    
    last_check = datetime.now(APP_TZ)
    
    # Настройка таймаута сессии
    if "ollama" in (ACTIVE_MODEL or "").lower():
        t_out = aiohttp.ClientTimeout(total=None)
        base_timeout = None
    else:
        t_out = aiohttp.ClientTimeout(total=INTERVAL)
        base_timeout = INTERVAL

    async with aiohttp.ClientSession(timeout=t_out) as session:
        logging.info("🚀 Bot started. Waiting for tasks...")
        try:
            while True:
                # Проверка глобальной паузы
                pause_until = meta_mgr.get("pause_until")
                if pause_until:
                    rem = pause_until - time.time()
                    if rem > 0:
                        logging.info(f"⏸️ Pause active for {int(rem)}s")
                        await asyncio.sleep(rem)
                        meta_mgr.set("pause_until", None)
                        continue
                    else:
                        meta_mgr.set("pause_until", None)

                # Периодическая проверка доступности источников
                now = datetime.now(APP_TZ)
                if (now - last_check) > timedelta(days=1):
                    await check_sources()
                    last_check = now

                # Запуск цикла новостей
                logging.info("🔄 Checking RSS...")
                try:
                    if base_timeout:
                        await asyncio.wait_for(send_news(session), timeout=base_timeout)
                    else:
                        await send_news(session)
                except asyncio.TimeoutError:
                    logging.warning("⏰ Timeout in main loop")
                except Exception as e:
                    logging.exception(f"❌ Main loop exception: {e}")

                # Задержка между циклами
                await asyncio.sleep(INTERVAL)

        except KeyboardInterrupt:
            logging.info("🛑 Bot stopping...")
        finally:
            await db.close()

@atexit.register
def _cleanup():
    # Синхронная очистка, если нужна
    pass

if __name__ == "__main__":
    asyncio.run(main())