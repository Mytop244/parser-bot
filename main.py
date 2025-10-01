import aiohttp
from bs4 import BeautifulSoup
from telegram import Bot
import os
import asyncio
import sys
import json
from pathlib import Path
from dotenv import load_dotenv  # NEW

# Загружаем переменные из .env
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    sys.exit("❌ Ошибка: TELEGRAM_TOKEN или CHAT_ID не заданы")

bot = Bot(token=TELEGRAM_TOKEN)

DATA_FILE = Path("sent_links.json")
if DATA_FILE.exists():
    try:
        sent_links = set(json.loads(DATA_FILE.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        sent_links = set()
else:
    sent_links = set()

def save_links():
    DATA_FILE.write_text(json.dumps(list(sent_links), ensure_ascii=False, indent=2), encoding="utf-8")

async def fetch_news():
    url = "https://www.wired.com/feed/rss"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"❌ Ошибка загрузки RSS: {resp.status}")
                    return []
                text = await resp.text()
    except Exception as e:
        print(f"❌ Сетевая ошибка: {e}")
        return []

    soup = BeautifulSoup(text, "lxml-xml")
    return [(i.title.text, i.link.text) for i in soup.find_all("item")]

async def send_news():
    news = await fetch_news()
    if not news:
        print("⚠️ Нет новостей")
        return

    for title, link in news[:5]:
        if link in sent_links:
            continue
        try:
            await bot.send_message(chat_id=CHAT_ID, text=f"{title}\n{link}")
            sent_links.add(link)
            save_links()
            print(f"✅ Отправлено: {title}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        await asyncio.sleep(1)

async def main():
    while True:
        await send_news()
        print("⏳ Жду 10 минут...")
        await asyncio.sleep(600)

if __name__ == "__main__":
    asyncio.run(main())
