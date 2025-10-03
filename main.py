import aiohttp
from bs4 import BeautifulSoup
from telegram import Bot
import os
import asyncio
import sys
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from dotenv import load_dotenv

# -------------------------------
# 🔧 Логирование с ежедневной ротацией
# -------------------------------
os.makedirs("log", exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
)
log_filename = datetime.now().strftime("log/parser-%Y-%m-%d.log")

file_handler = TimedRotatingFileHandler(
    log_filename, when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
file_handler.suffix = "%Y-%m-%d.log"
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

# -------------------------------
# 🔧 Загружаем настройки из .env
# -------------------------------
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
RSS_URLS = os.environ.get("RSS_URLS", "https://www.wired.com/feed/rss,https://www.reuters.com/rssFeed/worldNews").split(",")
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))
SENT_LINKS_FILE = os.environ.get("SENT_LINKS_FILE", "sent_links.json")

if not TELEGRAM_TOKEN or not CHAT_ID:
    sys.exit("❌ Ошибка: TELEGRAM_TOKEN или CHAT_ID не заданы")

try:
    CHAT_ID = int(CHAT_ID)
except Exception:
    pass

bot = Bot(token=TELEGRAM_TOKEN)

# -------------------------------
# 📂 Локальное хранилище отправленных ссылок
# -------------------------------
if os.path.exists(SENT_LINKS_FILE):
    try:
        with open(SENT_LINKS_FILE, "r", encoding="utf-8") as f:
            sent_links = set(json.load(f))
    except Exception:
        sent_links = set()
else:
    sent_links = set()

def save_links():
    with open(SENT_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(sent_links), f, ensure_ascii=False, indent=2)

# -------------------------------
# 🌐 Получение новостей
# -------------------------------
async def fetch_news(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logging.error(f"Ошибка загрузки {url}: {resp.status}")
                    return []
                text = await resp.text()
    except Exception as e:
        logging.error(f"Сетевая ошибка {url}: {e}")
        return []

    soup = BeautifulSoup(text, "lxml-xml")
    return [(i.title.text, i.link.text, url) for i in soup.find_all("item")]

# -------------------------------
# 📩 Отправка новостей
# -------------------------------
async def send_news():
    all_news = []
    for url in RSS_URLS:
        news = await fetch_news(url.strip())
        all_news.extend(news)

    if not all_news:
        logging.warning("Нет новостей")
        return

    for title, link, source in all_news[:NEWS_LIMIT]:
        if link in sent_links:
            continue
        try:
            await bot.send_message(chat_id=CHAT_ID, text=f"📌 {title}\n{link}\n🌍 {source}")
            sent_links.add(link)
            save_links()
            logging.info(f"Отправлено: {title}")
        except Exception as e:
            logging.error(f"Ошибка отправки: {e}")
        await asyncio.sleep(1)

# -------------------------------
# 🔄 Цикл воркера
# -------------------------------
async def main():
    while True:
        logging.info("Начало проверки новостей")
        await send_news()
        logging.info(f"Следующая проверка через {INTERVAL // 60} мин")
        await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
