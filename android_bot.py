import os
import sys
import time
import asyncio
import ssl
import logging
import re
import html
import random
import atexit
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Tuple

# Сторонние библиотеки
from dotenv import load_dotenv
import aiohttp
import feedparser
from bs4 import BeautifulSoup
import aiosqlite
from telegram import Bot
from telegram.request import HTTPXRequest as Request
from telegram.error import RetryAfter, NetworkError, TimedOut

# ==========================================
# ⚙️ КОНФИГУРАЦИЯ
# ==========================================

# Определяем пути (работает и для скрипта, и для скомпилированного exe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def fix_path(name: str) -> str:
    return os.path.join(BASE_DIR, name)

load_dotenv(fix_path(".env"))

# Логирование
LOG_FILE = fix_path("bot.log")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=2, encoding="utf-8")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger("NewsBot")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Настройки Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RSS_FILE = fix_path("rss.txt")
RSS_URLS = [u.strip() for u in os.getenv("RSS_URLS", "").split(",") if u.strip()]

if os.path.exists(RSS_FILE):
    try:
        with open(RSS_FILE, 'r', encoding='utf-8') as f:
            RSS_URLS.extend([l.strip() for l in f if l.strip() and not l.strip().startswith('#')])
    except Exception as e:
        logger.error(f"Ошибка rss.txt: {e}")

if not TELEGRAM_TOKEN or not CHAT_ID:
    logger.critical("❌ TELEGRAM_TOKEN или CHAT_ID не заданы")
    sys.exit(1)

# AI Настройки
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# Параметры
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", 5))
INTERVAL = int(os.getenv("INTERVAL", 600))
DAYS_LIMIT = int(os.getenv("DAYS_LIMIT", 2))
BLOCKED_WORDS = [w.strip().lower() for w in os.getenv("BLOCKED_WORDS", "").split(",") if w.strip()]

# SSL Context
ssl_ctx = ssl.create_default_context()
if os.getenv("SSL_VERIFY", "1") == "0":
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

# ==========================================
# 🗄️ DATABASE CLASS
# ==========================================
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
        if self.conn: await self.conn.close()

    async def is_processed(self, url: str) -> bool:
        if not self.conn: return False
        async with self.conn.execute("SELECT 1 FROM history WHERE url=? AND kind IN ('sent', 'seen')", (url,)) as cur:
            return await cur.fetchone() is not None

    async def mark(self, kind: str, url: str):
        if not self.conn: return
        await self.conn.execute("INSERT OR REPLACE INTO history (url, kind, timestamp) VALUES (?, ?, ?)", 
                                (url, kind, int(time.time())))
        await self.conn.commit()

    async def cleanup(self, days: int):
        cutoff = int(time.time() - (days * 86400))
        await self.conn.execute("DELETE FROM history WHERE timestamp < ?", (cutoff,))
        await self.conn.commit()

# ==========================================
# 🧠 AI ENGINE
# ==========================================
class AIEngine:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.key_idx = 0
        self.blocked_keys = {} # key: timestamp

    def _get_gemini_key(self) -> Optional[str]:
        now = time.time()
        valid_keys = [k for k in GEMINI_KEYS if self.blocked_keys.get(k, 0) < now]
        if not valid_keys: return None
        
        self.key_idx = self.key_idx % len(valid_keys)
        key = valid_keys[self.key_idx]
        self.key_idx += 1
        return key

    async def summarize(self, text: str) -> Tuple[str, str]:
        text = text[:8000] # Лимит символов для AI
        
        # 1. Попытка через Gemini
        if GEMINI_KEYS:
            for _ in range(3): # 3 попытки с разными ключами
                key = self._get_gemini_key()
                if not key: break
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
                payload = {"contents": [{"parts": [{"text": f"Сделай информативное саммари новости на русском языке. Выдели главное жирным. Текст:\n{text}"}]}]}
                
                try:
                    async with self.session.post(url, json=payload, timeout=40) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data["candidates"][0]["content"]["parts"][0]["text"].strip(), "Gemini"
                        elif resp.status in (429, 503):
                            logger.warning(f"Gemini 429/503. Блокирую ключ на 5 мин.")
                            self.blocked_keys[key] = time.time() + 300
                        else:
                            logger.error(f"Gemini Error {resp.status}")
                except Exception as e:
                    logger.error(f"Gemini connection error: {e}")

        # 2. Фолбэк на Ollama
        if OLLAMA_MODEL:
            try:
                url = "http://127.0.0.1:11434/api/generate"
                payload = {"model": OLLAMA_MODEL, "prompt": f"Резюмируй на русском:\n{text}", "stream": False}
                async with self.session.post(url, json=payload, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "").strip(), "Ollama"
            except Exception: pass
            
        return "", "None"

# ==========================================
# 🕸️ CONTENT PARSER
# ==========================================
class ContentParser:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def parse(self, url: str) -> Tuple[str, Optional[str]]:
        """Возвращает (текст, url_картинки)"""
        headers = {"User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36"}
        try:
            async with self.session.get(url, headers=headers, ssl=ssl_ctx, timeout=15) as resp:
                if resp.status != 200: return "", None
                raw_html = await resp.text()
        except Exception as e:
            logger.warning(f"Ошибка загрузки {url}: {e}")
            return "", None

        soup = BeautifulSoup(raw_html, "html.parser")
        
        # Картинка
        img_url = None
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            img_url = og_img["content"]

        # Чистка
        for tag in soup(["script", "style", "nav", "footer", "iframe", "header", "form", "aside", "button"]):
            tag.decompose()

        # Поиск текста
        article = soup.find("article")
        if not article:
            # Ищем самый большой блок текста
            nodes = soup.find_all("div")
            article = max(nodes, key=lambda n: len(n.get_text()), default=soup)

        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all(['p', 'h2', 'li']) if len(p.get_text(strip=True)) > 30]
        return "\n\n".join(paragraphs), img_url

# ==========================================
# ✈️ TELEGRAM SENDER
# ==========================================
class TelegramSender:
    def __init__(self, token):
        self.bot = Bot(token=token, request=Request(connect_timeout=10, read_timeout=30))

    def _split_text(self, text: str, limit=4000) -> List[str]:
        if len(text) <= limit: return [text]
        parts = []
        while len(text) > limit:
            split_at = text.rfind('\n', 0, limit)
            if split_at == -1: split_at = limit
            parts.append(text[:split_at])
            text = text[split_at:]
        parts.append(text)
        return parts

    async def send(self, chat_id, title, source, summary, link, img, model):
        # Очистка Markdown
        summary = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', summary)
        summary = re.sub(r'\*(.*?)\*', r'<i>\1</i>', summary)
        
        base_text = f"<b>{html.escape(title)}</b>\n📡 <i>{html.escape(source)}</i>\n\n{summary}\n\n🤖 {model} | 🔗 <a href=\"{link}\">Читать</a>"
        chunks = self._split_text(base_text, 1024 if img else 4096)

        try:
            # 1. Если есть картинка, отправляем первый кусок как caption
            if img:
                try:
                    await self.bot.send_photo(chat_id=chat_id, photo=img, caption=chunks[0], parse_mode="HTML")
                    chunks = chunks[1:] # Остальное шлем текстом
                except Exception as e:
                    logger.warning(f"Ошибка отправки фото: {e}. Шлю текст.")
            
            # 2. Отправка текстовых частей
            for chunk in chunks:
                if not chunk.strip(): continue
                await self.bot.send_message(chat_id=chat_id, text=chunk, parse_mode="HTML", disable_web_page_preview=True)
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Ошибка TG: {e}")

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
async def main():
    db = Database(fix_path("bot_history.db"))
    await db.connect()
    
    conn = aiohttp.TCPConnector(limit=5, ssl=ssl_ctx)
    session = aiohttp.ClientSession(connector=conn)
    
    ai = AIEngine(session)
    parser = ContentParser(session)
    tg = TelegramSender(TELEGRAM_TOKEN)
    
    logger.info("🚀 Бот запущен...")

    try:
        while True:
            await db.cleanup(DAYS_LIMIT + 1)
            
            # Асинхронный сбор RSS
            tasks = []
            for url in RSS_URLS:
                tasks.append(session.get(url, timeout=10))
            
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Парсинг RSS
                all_entries = []
                for i, resp in enumerate(responses):
                    if isinstance(resp, Exception) or resp.status != 200: continue
                    xml = await resp.text()
                    feed = feedparser.parse(xml)
                    src = feed.feed.get("title", RSS_URLS[i])
                    
                    for entry in feed.entries:
                        all_entries.append((entry, src))
                
                # Обработка записей
                count = 0
                for entry, src in all_entries:
                    if count >= NEWS_LIMIT: break
                    link = entry.get("link", "")
                    title = entry.get("title", "")
                    
                    if not link or await db.is_processed(link): continue
                    
                    # Проверка блокировок
                    if any(w in title.lower() for w in BLOCKED_WORDS):
                        await db.mark("seen", link)
                        continue

                    logger.info(f"🆕 {title}")
                    
                    # Парсинг тела
                    text, img = await parser.parse(link)
                    if len(text) < 100:
                        logger.info("Пропуск (короткий текст)")
                        await db.mark("seen", link)
                        continue
                        
                    # AI Саммари
                    summary, model = await ai.summarize(text)
                    if not summary: summary = text[:400] + "..."

                    # Отправка
                    await tg.send(CHAT_ID, title, src, summary, link, img, model)
                    
                    await db.mark("sent", link)
                    await db.mark("seen", link)
                    count += 1
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")

            logger.info(f"💤 Сон {INTERVAL} сек...")
            await asyncio.sleep(INTERVAL)

    except KeyboardInterrupt:
        logger.info("🛑 Стоп.")
    finally:
        await session.close()
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())