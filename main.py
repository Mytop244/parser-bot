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
log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
)

# текущий файл будет называться parser-YYYY-MM-DD.log
log_filename = datetime.now().strftime("parser-%Y-%m-%d.log")

file_handler = TimedRotatingFileHandler(
    log_filename,
    when="midnight",     # новая ротация каждый день
    interval=1,
    backupCount=7,       # храним 7 дней
    encoding="utf-8",
    utc=False
)
file_handler.suffix = "%Y-%m-%d.log"
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# -------------------------------
# 🔧 Загружаем настройки из .env
# -------------------------------
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
RSS_URL = os.environ.get("RSS_URL", "https://www.wired.com/feed/rss")
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
async def fetch_news():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(RSS_URL) as resp:
                if resp.status != 200:
                    logging.error(f"Ошибка загрузки RSS: {resp.status}")
                    return []
                text = await resp.text()
    except Exception as e:
        logging.error(f"Сетевая ошибка: {e}")
        return []

    soup = BeautifulSoup(text, "lxml-xml")
    return [(i.title.text, i.link.text) for i in soup.find_all("item")]

# -------------------------------
# 📩 Отправка новостей
# -------------------------------
async def send_news():
    news = await fetch_news()
    if not news:
        logging.warning("Нет новостей")
        return

    for title, link in news[:NEWS_LIMIT]:
        if link in sent_links:
            continue
        try:
            await bot.send_message(chat_id=CHAT_ID, text=f"{title}\n{link}")
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
