import os
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import xml.etree.ElementTree as ET
from telebot import TeleBot

# -------------------------------
# Создание папки pars и поддиректорий
# -------------------------------
BASE_DIR = "pars"
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "news_debug.log")
DATA_FILE = os.path.join(DATA_DIR, "sent_news.txt")
ENV_FILE = os.path.join(BASE_DIR, ".env")

# -------------------------------
# Если .env нет — создаём шаблон
# -------------------------------
if not os.path.exists(ENV_FILE):
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(
            "TELEGRAM_TOKEN=\n"
            "TELEGRAM_CHAT_ID=\n"
            "NEWS_URL=\n"
            "MAX_AGE_DAYS=1\n"
            "CHECK_INTERVAL=15\n"
            "KEYWORDS=нейросеть,ИИ,AI\n"
        )
    print(f"📝 Создан шаблон .env в {ENV_FILE}. Заполните его и запустите скрипт снова.")
    exit(0)

# -------------------------------
# Загрузка настроек из .env
# -------------------------------
load_dotenv(dotenv_path=ENV_FILE)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_URL = os.getenv("NEWS_URL")
MAX_AGE_DAYS = int(os.getenv("MAX_AGE_DAYS", 1))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 15))  # минуты
KEYWORDS = os.getenv("KEYWORDS", "").split(",")  # ключевые слова через запятую

# -------------------------------
# Настройка логирования
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# -------------------------------
# Инициализация Telegram
# -------------------------------
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or not NEWS_URL:
    logging.error("❌ TELEGRAM_TOKEN, TELEGRAM_CHAT_ID или NEWS_URL не заданы в .env")
    exit(1)

bot = TeleBot(TELEGRAM_TOKEN)

# -------------------------------
# Загрузка уже отправленных новостей
# -------------------------------
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        sent_news_ids = set(line.strip() for line in f)
else:
    sent_news_ids = set()

# -------------------------------
# Получение новостей
# -------------------------------
def fetch_news():
    try:
        response = requests.get(NEWS_URL, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        text = response.text

        if "xml" in content_type or text.strip().startswith("<"):
            return parse_rss(text)
        else:
            return parse_json(text)
    except Exception as e:
        logging.error(f"❌ Ошибка при получении новостей: {e}")
        return []

# -------------------------------
# Парсинг RSS
# -------------------------------
def parse_rss(xml_text):
    news_list = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.findall(".//item"):
            news_list.append({
                "id": item.findtext("guid") or item.findtext("link"),
                "title": item.findtext("title") or "<без заголовка>",
                "link": item.findtext("link") or "",
                "pubDate": item.findtext("pubDate") or ""
            })
    except Exception as e:
        logging.error(f"❌ Ошибка парсинга RSS: {e}")
    return news_list

# -------------------------------
# Парсинг JSON
# -------------------------------
def parse_json(json_text):
    news_list = []
    try:
        data = requests.utils.json.loads(json_text)
        articles = data.get("articles") or data.get("items") or data
        for item in articles:
            news_list.append({
                "id": item.get("id") or item.get("link") or item.get("url"),
                "title": item.get("title") or "<без заголовка>",
                "link": item.get("link") or item.get("url") or "",
                "pubDate": item.get("publishedAt") or item.get("pubDate") or ""
            })
    except Exception as e:
        logging.error(f"❌ Ошибка парсинга JSON: {e}")
    return news_list

# -------------------------------
# Фильтр и отправка
# -------------------------------
def parse_and_send_news(news_items):
    new_sent = []

    for item in news_items:
        news_id = str(item.get("id"))
        title = item.get("title")
        link = item.get("link")
        pubDate = item.get("pubDate")

        # Проверка дубликатов
        if news_id in sent_news_ids:
            continue

        # Проверка даты
        if pubDate:
            try:
                news_dt = datetime.strptime(pubDate[:25], "%a, %d %b %Y %H:%M:%S")
                if datetime.now() - news_dt > timedelta(days=MAX_AGE_DAYS):
                    continue
            except Exception:
                pass

        # Фильтр по ключевым словам
        if KEYWORDS and not any(word.lower() in title.lower() for word in KEYWORDS):
            continue

        # Отправка
        send_to_telegram(item)
        sent_news_ids.add(news_id)
        new_sent.append(news_id)

    if new_sent:
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            for nid in new_sent:
                f.write(nid + "\n")

def send_to_telegram(news_item):
    title = news_item.get("title")
    link = news_item.get("link")
    pubDate = news_item.get("pubDate")
    message = f"{title}\n{link}\nДата: {pubDate}"
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info(f"💬 Отправлено: {title}")
    except Exception as e:
        logging.error(f"❌ Ошибка отправки в Telegram: {e}")

# -------------------------------
# Основной цикл
# -------------------------------
if __name__ == "__main__":
    logging.info("🚀 Старт новостного бота в папке pars")
    while True:
        news = fetch_news()
        if news:
            parse_and_send_news(news)
        else:
            logging.info("ℹ Нет новых новостей")
        time.sleep(CHECK_INTERVAL * 60)
