import os
import sys
import json
import time
import asyncio
import ssl
import logging
import tempfile
import re
import html
import calendar
import shutil
import random
import atexit
import gc
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from logging.handlers import RotatingFileHandler

# Сторонние библиотеки
from dotenv import load_dotenv
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from telegram import Bot
from telegram.error import RetryAfter, TimedOut, NetworkError
from telegram.request import HTTPXRequest as Request
import aiosqlite

# ---- НАСТРОЙКИ ПУТЕЙ ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fix_path(name: str) -> str:
    return os.path.join(BASE_DIR, name)

# Загрузка переменных окружения
load_dotenv(fix_path(".env"))

# ---- ЛОГИРОВАНИЕ ----
LOG_FILE = fix_path("bot.log")
# Формат: Время | Уровень | Сообщение (коротко для экрана телефона)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

# Ротация: макс 2 МБ, хранить 2 файла (экономим место на телефоне)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=2, encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Снижаем шум от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ---- КОНФИГУРАЦИЯ ----
# Уменьшаем конкурентность для экономии батареи
CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))
_network_semaphore = asyncio.Semaphore(CONCURRENCY)

BLOCKED_WORDS = [w.strip().lower() for w in os.getenv("BLOCKED_WORDS", "").split(",") if w.strip()]

DB_PATH = fix_path("bot_history.db")
META_FILE = fix_path("bot_meta.json")

# Настройки RSS
RSS_FILE = fix_path("rss.txt")
RSS_URLS = [u.strip() for u in os.environ.get("RSS_URLS", "").split(",") if u.strip()]
if os.path.exists(RSS_FILE):
    try:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            file_urls = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            RSS_URLS.extend(file_urls)
    except Exception as e:
        logging.error(f"Ошибка чтения rss.txt: {e}")

# Настройки Telegram
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
try:
    CHAT_ID = int(os.environ.get("CHAT_ID", "0"))
except:
    CHAT_ID = None

if not TELEGRAM_TOKEN or not CHAT_ID:
    logging.critical("❌ TELEGRAM_TOKEN или CHAT_ID не заданы в .env!")
    sys.exit(1)

# Настройки AI
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b") # Легкая модель для телефона

# Прочие настройки
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))
DAYS_LIMIT = int(os.environ.get("DAYS_LIMIT", 2))
PARSER_MAX_TEXT_LENGTH = int(os.environ.get("PARSER_MAX_TEXT_LENGTH", "8000"))
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))

# Timezone
try:
    APP_TZ = ZoneInfo(os.getenv("TIMEZONE", "UTC"))
except:
    APP_TZ = timezone.utc

# SSL (иногда в Termux проблемы с сертификатами)
SSL_VERIFY = os.getenv("SSL_VERIFY", "1") == "1"
ssl_ctx = ssl.create_default_context()
if not SSL_VERIFY:
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

# ---- МЕНЕДЖЕР СОСТОЯНИЯ (DB) ----
class Database:
    def __init__(self, path):
        self.path = path
        self.conn = None

    async def connect(self):
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
        await self.conn.commit()

    async def close(self):
        if self.conn:
            await self.conn.close()

    async def exists(self, kind: str, url: str) -> bool:
        if not self.conn: return False
        async with self.conn.execute("SELECT 1 FROM history WHERE url=? AND kind=?", (url, kind)) as cur:
            return await cur.fetchone() is not None

    async def add(self, kind: str, url: str):
        if not self.conn: return
        await self.conn.execute(
            "INSERT OR REPLACE INTO history (url, kind, timestamp) VALUES (?, ?, ?)", 
            (url, kind, int(time.time()))
        )
        await self.conn.commit()

    async def cleanup(self, days: int):
        cutoff = int(time.time() - (days * 86400))
        await self.conn.execute("DELETE FROM history WHERE timestamp < ?", (cutoff,))
        await self.conn.commit()

# ---- МЕНЕДЖЕР META (JSON) ----
class MetaManager:
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

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
        except: pass

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

# ---- ГЛОБАЛЬНЫЕ ОБЪЕКТЫ ----
db = Database(DB_PATH)
meta_mgr = MetaManager(META_FILE)
_session = None

async def get_session():
    global _session
    if _session is None or _session.closed:
        # Оптимизация для мобильной сети: короткий таймаут подключения, длинный чтения
        timeout = aiohttp.ClientTimeout(total=45, connect=10)
        # keepalive_timeout меньше, чтобы не держать мертвые сокеты
        connector = aiohttp.TCPConnector(limit=50, ssl=ssl_ctx, ttl_dns_cache=300, keepalive_timeout=30)
        _session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _session

bot = Bot(token=TELEGRAM_TOKEN, request=Request(connect_timeout=15, read_timeout=30))

# ---- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----
def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def fetch_url(url):
    session = await get_session()
    try:
        async with _network_semaphore:
            async with session.get(url, headers={"User-Agent": "TermuxBot/1.0"}, ssl=ssl_ctx) as response:
                if response.status == 200:
                    return await response.text()
    except Exception as e:
        logging.warning(f"Ошибка загрузки {url}: {e}")
    return None

async def extract_content(url):
    """Простая экстракция текста для экономии ресурсов Termux"""
    html_text = await fetch_url(url)
    if not html_text: return ""
    
    soup = BeautifulSoup(html_text, "html.parser")
    # Удаляем мусор
    for tag in soup(["script", "style", "nav", "footer", "iframe", "header"]):
        tag.decompose()
        
    # Ищем основной текст
    text = ""
    article = soup.find('article')
    if article:
        text = article.get_text(" ", strip=True)
    else:
        # Fallback: ищем параграфы
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])
        
    return text[:PARSER_MAX_TEXT_LENGTH]

# ---- AI SUMMARIZATION ----
async def summarize_gemini(text: str):
    if not GEMINI_KEYS: return None
    
    # Ротация ключей
    idx = int(meta_mgr.get("gemini_idx", 0)) % len(GEMINI_KEYS)
    key = GEMINI_KEYS[idx]
    meta_mgr.set("gemini_idx", idx + 1)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": f"Сделай краткое резюме новости на русском языке (3-4 предложения, без вступлений):\n{text}"}]}]
    }
    
    session = await get_session()
    try:
        async with session.post(url, json=payload, timeout=MODEL_TIMEOUT) as resp:
            if resp.status != 200:
                logging.error(f"Gemini API Error: {resp.status}")
                return None
            data = await resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logging.error(f"Gemini Exception: {e}")
        return None

async def summarize_ollama(text: str):
    # Локальный Ollama на Termux
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Резюмируй на русском языке:\n{text}",
        "stream": False,
        "options": {"num_ctx": 2048}
    }
    session = await get_session()
    try:
        async with session.post(url, json=payload, timeout=120) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("response", "")
    except Exception:
        pass
    return None

# ---- ОСНОВНОЙ ЦИКЛ ----
async def process_feed():
    logging.info("📡 Проверка RSS лент...")
    session = await get_session()
    
    feeds_data = []
    # Параллельная загрузка RSS
    tasks = [fetch_url(url) for url in RSS_URLS]
    results = await asyncio.gather(*tasks)
    
    for i, xml in enumerate(results):
        if not xml: continue
        try:
            feed = feedparser.parse(xml)
            source_title = feed.feed.get("title", RSS_URLS[i])
            for entry in feed.entries:
                # Фильтр по дате (если есть)
                pub_ts = time.time()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_ts = calendar.timegm(entry.published_parsed)
                
                # Пропускаем старое
                if time.time() - pub_ts > DAYS_LIMIT * 86400: continue
                
                feeds_data.append({
                    "title": entry.get("title", "Без заголовка"),
                    "link": entry.get("link", ""),
                    "source": source_title,
                    "ts": pub_ts
                })
        except Exception as e:
            logging.error(f"Ошибка парсинга {RSS_URLS[i]}: {e}")

    # Сортировка: старые сначала, чтобы соблюсти хронологию отправки, или новые
    feeds_data.sort(key=lambda x: x["ts"])
    
    count = 0
    for item in feeds_data:
        if count >= NEWS_LIMIT: break
        link = item["link"]
        
        if await db.exists("sent", link) or await db.exists("seen", link):
            continue

        logging.info(f"🆕 Новая статья: {item['title']}")
        
        # Загрузка контента
        content = await extract_content(link)
        
        # Проверка блокировки слов
        combined_text = (item["title"] + " " + content).lower()
        if any(w in combined_text for w in BLOCKED_WORDS):
            logging.info(f"🚫 Заблокировано (фильтр): {item['title']}")
            await db.add("seen", link)
            continue
            
        if len(content.split()) < 20:
             logging.info("⏭️ Слишком короткая статья, пропуск")
             await db.add("seen", link)
             continue

        # Генерация саммари
        summary = None
        if GEMINI_KEYS:
            summary = await summarize_gemini(content)
        
        # Fallback to Ollama if configured
        if not summary and OLLAMA_MODEL:
            logging.info("Gemini недоступен, пробую Ollama...")
            summary = await summarize_ollama(content)
            
        if not summary:
            summary = "Не удалось сгенерировать краткое содержание."

        # Отправка
        msg = (
            f"<b>{html.escape(item['title'])}</b>\n"
            f"📡 <i>{html.escape(item['source'])}</i>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{html.escape(summary)}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🔗 <a href=\"{link}\">Читать полностью</a>"
        )
        
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
            await db.add("sent", link)
            await db.add("seen", link)
            count += 1
            logging.info("✅ Отправлено")
            await asyncio.sleep(3) # Пауза между сообщениями
        except Exception as e:
            logging.error(f"Ошибка отправки: {e}")
            await asyncio.sleep(5)

    # Очистка памяти после цикла
    gc.collect()

async def main():
    await db.connect()
    # Чистка старой истории раз в сутки
    await db.cleanup(7)
    
    logging.info("🚀 Бот запущен в Termux")
    
    try:
        while True:
            try:
                await process_feed()
            except Exception as e:
                logging.error(f"Сбой цикла: {e}")
            
            logging.info(f"💤 Сон {INTERVAL} сек...")
            await asyncio.sleep(INTERVAL)
    except KeyboardInterrupt:
        logging.info("🛑 Остановка...")
    finally:
        await db.close()
        if _session:
            await _session.close()

if __name__ == "__main__":
    asyncio.run(main())