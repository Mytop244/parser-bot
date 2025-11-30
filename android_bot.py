import os
import sys
import asyncio
import logging
import time
import html
import random
import shutil
import ssl
import calendar
from datetime import datetime, timedelta, timezone

# Сторонние библиотеки
import aiohttp
import feedparser
import aiosqlite
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import RetryAfter, NetworkError
from logging.handlers import RotatingFileHandler

# --- НАСТРОЙКИ ОКРУЖЕНИЯ ANDROID ---

# Определяем папку, где лежит скрипт (важно для Pydroid 3 / Termux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fix_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)

# Загрузка конфигов
load_dotenv(fix_path(".env"))

# --- ЛОГИРОВАНИЕ (LITE) ---
# Ограничиваем размер лога 1 МБ, чтобы не забивать память телефона
LOG_FILE = fix_path("bot_lite.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1024*1024, backupCount=1, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# RSS: Сначала ищем в .env, затем в файле
RSS_URLS = [u.strip() for u in os.getenv("RSS_URLS", "").split(",") if u.strip()]
RSS_FILE = fix_path("rss.txt")
if os.path.exists(RSS_FILE):
    try:
        with open(RSS_FILE, "r", encoding="utf-8") as f:
            file_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            if file_urls:
                RSS_URLS = file_urls
    except Exception as e:
        logger.error(f"Ошибка чтения rss.txt: {e}")

# Настройки API и производительности
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Flash модель быстрее и дешевле
GEMINI_PROMPT = os.getenv("GEMINI_PROMPT", "Кратко (3-5 предложений) перескажи суть новости на русском. Без вступлений.")

# Экономия батареи и трафика
CONCURRENCY = int(os.getenv("CONCURRENCY", "3"))  # Мало потоков для телефона
INTERVAL = int(os.getenv("INTERVAL", "1800"))     # 30 минут паузы
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "5"))
DAYS_LIMIT = int(os.getenv("DAYS_LIMIT", "2"))

if not TELEGRAM_TOKEN or not CHAT_ID:
    logger.critical("❌ Не задан TELEGRAM_TOKEN или CHAT_ID в .env")
    sys.exit(1)

if not RSS_URLS:
    logger.critical("❌ Список RSS пуст. Проверьте .env или rss.txt")
    sys.exit(1)

# --- TERMUX WAKE LOCK ---
# Не дает Android убить процесс в фоне
def acquire_wakelock():
    if shutil.which("termux-wake-lock"):
        os.system("termux-wake-lock")
        logger.info("🔋 Termux WakeLock включен")

def release_wakelock():
    if shutil.which("termux-wake-unlock"):
        os.system("termux-wake-unlock")
        logger.info("🪫 Termux WakeLock выключен")

# --- БАЗА ДАННЫХ ---
class Database:
    def __init__(self):
        self.path = fix_path("bot_history.db")
        self.conn = None

    async def connect(self):
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sent_news (
                link TEXT PRIMARY KEY,
                timestamp INTEGER
            )
        """)
        await self.conn.commit()

    async def is_sent(self, link):
        async with self.conn.execute("SELECT 1 FROM sent_news WHERE link=?", (link,)) as cursor:
            return await cursor.fetchone() is not None

    async def add_sent(self, link):
        await self.conn.execute(
            "INSERT OR REPLACE INTO sent_news (link, timestamp) VALUES (?, ?)", 
            (link, int(time.time()))
        )
        await self.conn.commit()

    async def cleanup(self, days=7):
        cutoff = int(time.time()) - (days * 86400)
        await self.conn.execute("DELETE FROM sent_news WHERE timestamp < ?", (cutoff,))
        await self.conn.commit()

# --- СЕТЬ И AI ---

async def get_session():
    # Настройки таймаутов под мобильную сеть (медленнее, но надежнее)
    timeout = aiohttp.ClientTimeout(total=45, connect=15)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False, ttl_dns_cache=300)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)

async def fetch_rss(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                return feedparser.parse(content)
    except Exception as e:
        logger.warning(f"Ошибка RSS {url}: {e}")
    return None

async def summarize_gemini(text, session):
    if not GEMINI_KEYS:
        return text[:600] + "..."
    
    # Ротация ключей
    api_key = random.choice(GEMINI_KEYS)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": f"{GEMINI_PROMPT}\n\nТекст:\n{text[:8000]}"}]}]
    }
    
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"Gemini Error {resp.status}")
    except Exception as e:
        logger.error(f"Gemini Exception: {e}")
    
    return text[:600] + "..." # Fallback

# --- ОСНОВНАЯ ЛОГИКА ---

async def main():
    acquire_wakelock()
    db = Database()
    await db.connect()
    await db.cleanup(DAYS_LIMIT)
    
    bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest(connection_pool_size=CONCURRENCY))
    
    logger.info("🚀 Бот запущен на Android")

    try:
        while True:
            logger.info("🔄 Проверка лент...")
            
            # Создаем сессию заново каждый цикл (стабильнее при смене сетей Wi-Fi <-> 4G)
            async with await get_session() as session:
                
                tasks = [fetch_rss(session, url) for url in RSS_URLS]
                feeds_results = await asyncio.gather(*tasks)
                
                news_queue = []
                
                for feed in feeds_results:
                    if not feed or not feed.entries: continue
                    
                    # Берем свежие записи
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=DAYS_LIMIT)
                    
                    for entry in feed.entries[:NEWS_LIMIT]:
                        link = entry.get("link", "")
                        if await db.is_sent(link):
                            continue
                            
                        # Проверка даты (если есть)
                        pub_struct = entry.get("published_parsed")
                        if pub_struct:
                            pub_date = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                            if pub_date < cutoff_date:
                                continue
                        
                        # Собираем данные
                        title = entry.get("title", "Без заголовка")
                        source = feed.feed.get("title", "RSS")
                        
                        # Пытаемся взять summary или description. 
                        # Если нет - не парсим страницу (экономим трафик), отправляем заголовок.
                        raw_text = entry.get("summary") or entry.get("description") or title
                        clean_text = BeautifulSoup(raw_text, "html.parser").get_text()
                        
                        news_queue.append({
                            "title": title,
                            "link": link,
                            "source": source,
                            "text": clean_text
                        })

                # Обработка очереди (отправляем не всё сразу, чтобы не словить бан)
                count = 0
                for item in news_queue:
                    if count >= NEWS_LIMIT: break
                    
                    logger.info(f"⚡ Обработка: {item['title'][:30]}")
                    
                    summary = await summarize_gemini(item['text'], session)
                    
                    msg = (
                        f"<b>{html.escape(item['title'])}</b>\n"
                        f"📡 {html.escape(item['source'])}\n\n"
                        f"{html.escape(summary)}\n\n"
                        f"🔗 <a href='{item['link']}'>Читать далее</a>"
                    )
                    
                    try:
                        await bot.send_message(
                            chat_id=CHAT_ID, 
                            text=msg, 
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )
                        await db.add_sent(item['link'])
                        count += 1
                        await asyncio.sleep(3) # Пауза между сообщениями
                    except RetryAfter as e:
                        logger.warning(f"Flood limit. Ждем {e.retry_after} сек")
                        await asyncio.sleep(e.retry_after)
                    except Exception as e:
                        logger.error(f"Ошибка отправки: {e}")

            logger.info(f"💤 Сон {INTERVAL} сек...")
            await asyncio.sleep(INTERVAL)

    except KeyboardInterrupt:
        logger.info("🛑 Остановка...")
    finally:
        await db.conn.close()
        release_wakelock()

if __name__ == "__main__":
    asyncio.run(main())