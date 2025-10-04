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
from collections import defaultdict, deque
from playwright.async_api import async_playwright   # 🔥 Playwright

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
ROUND_ROBIN_MODE = int(os.environ.get("ROUND_ROBIN_MODE", 1))  # 0 = обычный режим, 1 = по кругу
BROWSER = os.environ.get("BROWSER", "chromium").lower()  # ⚡ выбор браузера

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
log_formatter.converter = time.localtime

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
# 🌐 Получение новостей (RSS)
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
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    pub_date = dt
            except Exception:
                pass
        news_list.append((title, link, url, pub_date))

    return news_list

# -------------------------------
# 📰 Загрузка статьи через Playwright
# -------------------------------
async def fetch_article(link: str) -> str:
    try:
        async with async_playwright() as p:
            if BROWSER == "firefox":
                browser = await p.firefox.launch(headless=True)
            elif BROWSER == "webkit":
                browser = await p.webkit.launch(headless=True)
            else:
                browser = await p.chromium.launch(headless=True)

            page = await browser.new_page()
            await page.goto(link, timeout=60000)

            try:
                content = await page.inner_text("article")     # основной вариант
            except:
                content = await page.inner_text("body")        # fallback

            await browser.close()
            return content.strip()
    except Exception as e:
        logging.error(f"Playwright ошибка: {e}")
        return ""

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

    cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=DAYS_LIMIT)
    all_news = [n for n in all_news if n[3] and n[3] >= cutoff_date]

    if not all_news:
        logging.info(f"Нет новостей за последние {DAYS_LIMIT} дн.")
        return

    # читаем историю
    try:
        with open(SENT_LINKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        sent_data = data.get("links", {}) if isinstance(data, dict) else {}
        last_index = data.get("last_source_index", 0) if isinstance(data, dict) else 0
        sent_data = {link: date_str for link, date_str in sent_data.items()
                     if datetime.strptime(date_str, "%Y-%m-%d %H:%M") >= cutoff_date}
    except Exception:
        sent_data = {}
        last_index = 0

    if ROUND_ROBIN_MODE == 1:
        sources = defaultdict(deque)
        for title, link, source, pub_date in sorted(all_news, key=lambda x: x[3] or datetime.min, reverse=True):
            sources[source].append((title, link, source, pub_date))
        source_list = list(sources.keys())
        rr_queue = []
        i = last_index
        while any(sources.values()):
            src = source_list[i % len(source_list)]
            if sources[src]:
                rr_queue.append(sources[src].popleft())
            i += 1
        new_items = [item for item in rr_queue if item[1] not in sent_data]
    else:
        all_news.sort(key=lambda x: x[3] or datetime.min, reverse=True)
        new_items = [item for item in all_news if item[1] not in sent_data]

    # отправка
    sent_count = 0
    limit = len(new_items) if NEWS_LIMIT == 0 else min(len(new_items), NEWS_LIMIT)

    for j, (title, link, source, pub_date) in enumerate(new_items[:limit]):
        try:
            date_str = pub_date.strftime("%Y-%m-%d %H:%M") if pub_date else "без даты"
            article_text = await fetch_article(link)

            if article_text:
                msg = f"{title}\n\n{article_text[:3500]}..."
                await bot.send_message(chat_id=CHAT_ID, text=msg)
            else:
                await bot.send_message(chat_id=CHAT_ID, text=link)

            sent_data[link] = date_str
            sent_count += 1
            logging.info(f"Отправлено: {title} | Источник: {source} | Дата: {date_str}")
        except Exception as e:
            logging.error(f"Ошибка отправки: {e}")
        await asyncio.sleep(1)

    save_data = {"links": sent_data}
    if ROUND_ROBIN_MODE == 1 and 'source_list' in locals() and source_list:
        new_last_index = (last_index + sent_count) % len(source_list)
        save_data["last_source_index"] = new_last_index

    with open(SENT_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    logging.info(f"Всего отправлено новостей: {sent_count} из {len(new_items)} (лимит {NEWS_LIMIT})")

# -------------------------------
# 🔄 Цикл воркера
# -------------------------------
async def main():
    last_check = datetime.min
    while True:
        now = datetime.now()
        if (now - last_check) > timedelta(days=1):
            await check_sources()
            last_check = now

        logging.info("Начало проверки новостей")
        await send_news()
        logging.info(f"Следующая проверка через {INTERVAL // 60} мин")
        await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
