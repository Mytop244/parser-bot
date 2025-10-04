import aiohttp
from bs4 import BeautifulSoup
from telegram import Bot
import os
import asyncio
import sys
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from email.utils import parsedate_to_datetime
import time

# -------------------------------
# 🔧 Загружаем настройки из .env
# -------------------------------
load_dotenv()

TIMEZONE = os.environ.get("TIMEZONE", "UTC")
os.environ['TZ'] = TIMEZONE
time.tzset()  # применяем TZ

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
RSS_URLS = os.environ.get("RSS_URLS", "").split(",")
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))
SENT_LINKS_FILE = os.environ.get("SENT_LINKS_FILE", "sent_links.json")
DAYS_LIMIT = int(os.environ.get("DAYS_LIMIT", 1))  # по умолчанию 1 день

if not TELEGRAM_TOKEN or not CHAT_ID:
    sys.exit("❌ Ошибка: TELEGRAM_TOKEN или CHAT_ID не заданы")

try:
    CHAT_ID = int(CHAT_ID)
except Exception:
    pass

bot = Bot(token=TELEGRAM_TOKEN)

# -------------------------------
# 🔧 Логирование с ежедневной ротацией
# -------------------------------
os.makedirs("log", exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
)
log_formatter.converter = time.localtime  # локальное время

log_filename = datetime.now().strftime("log/parser-%Y-%m-%d.log")

file_handler = TimedRotatingFileHandler(
    log_filename,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
    utc=False
)
file_handler.suffix = "%Y-%m-%d.log"
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

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

    news_list = []
    for i in soup.find_all("item"):
        title = i.title.text if i.title else "Без заголовка"
        link = i.link.text if i.link else ""
        pub_date = None
        if i.pubDate:
            try:
                dt = parsedate_to_datetime(i.pubDate.text)
                if dt is not None:
                    # Приводим дату к UTC и делаем её offset-naive (без tzinfo)
                    if dt.tzinfo is None:
                        # Если нет tz, считаем дату в UTC
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    pub_date = dt
            except Exception:
                pass
        news_list.append((title, link, url, pub_date))

    return news_list

# -------------------------------
# ✅ Проверка источников
# -------------------------------
async def check_sources():
    logging.info("🔍 Ежедневная проверка источников...")
    for url in RSS_URLS:
        if not url.strip():
            continue
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url.strip()) as resp:
                    if resp.status == 200:
                        logging.info(f"✅ Источник доступен: {url}")
                    else:
                        logging.warning(f"⚠️ Источник {url} вернул статус {resp.status}")
        except Exception as e:
            logging.error(f"❌ Источник {url} недоступен: {e}")

# -------------------------------
# 📩 Отправка новостей
# -------------------------------
async def send_news():
    all_news = []
    for url in RSS_URLS:
        if url.strip():
            news = await fetch_news(url.strip())
            all_news.extend(news)

    if not all_news:
        logging.warning("Нет новостей")
        return

    # 🔹 фильтр по дате публикации
    cutoff_date = datetime.utcnow() - timedelta(days=DAYS_LIMIT)
    all_news = [n for n in all_news if n[3] and n[3] >= cutoff_date]

    if not all_news:
        logging.info(f"Нет новостей за последние {DAYS_LIMIT} дн.")
        return

    logging.info(
        f"Найдено {len(all_news)} свежих новостей | Источники: {', '.join(set(url for _, _, url, _ in all_news))}"
    )

    # 🔹 сортировка по дате (новые первыми)
    all_news.sort(key=lambda x: x[3] or datetime.min, reverse=True)

    # 🔹 очищаем старые ссылки
    try:
        # читаем старые ссылки с датами
        with open(SENT_LINKS_FILE, "r", encoding="utf-8") as f:
            sent_data = json.load(f)
        if isinstance(sent_data, dict):
            # старый формат {link: "YYYY-MM-DD HH:MM"}
            sent_data = {
                link: date_str for link, date_str in sent_data.items()
                if datetime.strptime(date_str, "%Y-%m-%d %H:%M") >= cutoff_date
            }
        else:
            # старый формат — просто список ссылок, оставляем пустым
            sent_data = {}
    except Exception:
        sent_data = {}

    # 🔹 только новые (не в sent_data)
    new_items = []
    for title, link, source, pub_date in all_news:
        if link not in sent_data:
            new_items.append((title, link, source, pub_date))
            sent_data[link] = pub_date.strftime("%Y-%m-%d %H:%M") if pub_date else "без даты"

    sent_count = 0
    for title, link, source, pub_date in new_items[:NEWS_LIMIT]:
        try:
            date_str = pub_date.strftime("%Y-%m-%d %H:%M") if pub_date else "без даты"
            await bot.send_message(
                chat_id=CHAT_ID,
                text=f"📰 {title}\n{link}\n📅 {date_str}\n🌍 {source}"
            )
            sent_count += 1
            logging.info(f"Отправлено: {title} | Источник: {source} | Дата: {date_str}")
        except Exception as e:
            logging.error(f"Ошибка отправки: {e}")
        await asyncio.sleep(1)

    # 🔹 сохраняем очищенные и обновлённые ссылки
    with open(SENT_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(sent_data, f, ensure_ascii=False, indent=2)

    logging.info(f"Всего отправлено новостей: {sent_count} из {len(new_items)} (ограничение {NEWS_LIMIT})")

# -------------------------------
# 🔄 Цикл воркера
# -------------------------------
async def main():
    last_check = datetime.min
    while True:
        now = datetime.now()
        # раз в сутки проверка источников
        if (now - last_check) > timedelta(days=1):
            await check_sources()
            last_check = now

        logging.info("Начало проверки новостей")
        await send_news()
        logging.info(f"Следующая проверка через {INTERVAL // 60} мин")
        await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
