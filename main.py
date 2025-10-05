import os, sys, json, time, asyncio, ssl, logging, subprocess
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from dotenv import load_dotenv
import aiohttp, feedparser
from telegram import Bot
from bs4 import BeautifulSoup

# ---------------- ENV ----------------
load_dotenv()
if hasattr(time, "tzset"):
    os.environ["TZ"] = os.environ.get("TIMEZONE", "UTC")
    time.tzset()
else:
    logging.info("⏰ Windows: пропускаем установку TZ (tzset недоступен)")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID", 0))
RSS_URLS = [u.strip() for u in os.environ.get("RSS_URLS", "").split(",") if u.strip()]
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))
SENT_LINKS_FILE = os.environ.get("SENT_LINKS_FILE", "sent_links.json")
DAYS_LIMIT = int(os.environ.get("DAYS_LIMIT", 1))
ROUND_ROBIN_MODE = int(os.environ.get("ROUND_ROBIN_MODE", 1))
AI_STUDIO_KEY = os.environ.get("AI_STUDIO_KEY")
AI_PROJECT_ID = os.environ.get("AI_PROJECT_ID")
GEMINI_MODEL = os.environ.get("AI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")

# Батчи
BATCH_SIZE_SMALL = int(os.environ.get("BATCH_SIZE_SMALL", 5))
PAUSE_SMALL = int(os.environ.get("PAUSE_SMALL", 3))
BATCH_SIZE_MEDIUM = int(os.environ.get("BATCH_SIZE_MEDIUM", 15))
PAUSE_MEDIUM = int(os.environ.get("PAUSE_MEDIUM", 5))
BATCH_SIZE_LARGE = int(os.environ.get("BATCH_SIZE_LARGE", 25))
PAUSE_LARGE = int(os.environ.get("PAUSE_LARGE", 10))
SINGLE_MESSAGE_PAUSE = int(os.environ.get("SINGLE_MESSAGE_PAUSE", 1))


if not TELEGRAM_TOKEN or not CHAT_ID: 
    sys.exit("❌ TELEGRAM_TOKEN или CHAT_ID не заданы")
bot = Bot(token=TELEGRAM_TOKEN)

# ---------------- LOG ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("parser.log", encoding="utf-8")]
)

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# ---------------- HELPERS ----------------
async def fetch_and_check(url, head_only=False):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
        try:
            if head_only:
                async with s.head(url, ssl=ssl_ctx) as r:
                    return url, "✅ OK" if r.status == 200 else f"⚠️ HTTP {r.status}"
            async with s.get(url, ssl=ssl_ctx) as r:
                if r.status != 200: raise Exception(f"HTTP {r.status}")
                feed = feedparser.parse(await r.read())
                news=[]
                for e in feed.entries:
                    pub=None
                    if hasattr(e,'published_parsed') and e.published_parsed:
                        pub=datetime.fromtimestamp(datetime(*e.published_parsed[:6]).timestamp(), tz=timezone.utc)
                    news.append((e.get("title","Без заголовка").strip(), e.get("link","").strip(),
                                 feed.feed.get("title","Неизвестный источник").strip(), pub))
                return news
        except Exception as e:
            return (url, f"❌ {e.__class__.__name__}") if head_only else []

def clean_text(text: str) -> str:
    try:
        if "<" in text and ">" in text:
            text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        pass
    return " ".join(text.split())

# ---------------- Ollama local fallback ----------------
async def summarize_ollama(text):
    short_text = ". ".join(text.split(".")[:3])
    cmd = ['ollama', 'generate', 'ggml-model', f'Сделай краткое резюме новости:\n{short_text}']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"❌ Ошибка Ollama: {e}")
        return short_text[:400] + "..."

# ---------------- Gemini Summary with quota check ----------------
async def summarize(text, max_tokens=200):
    if not AI_STUDIO_KEY:
        logging.warning("⚠️ AI_STUDIO_KEY не задан, использую локальную Ollama")
        return await summarize_ollama(text)

    text = clean_text(text)
    short_text = ". ".join(text.split(".")[:2])
    logging.info(f"🤖 Gemini: готовлю резюме для текста: {short_text[:60]}...")

    # Проверка квоты
    try:
        quota_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:getQuota?key={AI_STUDIO_KEY}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(quota_url) as resp:
                if resp.status == 200:
                    quota_data = await resp.json()
                    remaining = quota_data.get("freeTierRemaining", 0)
                    logging.info(f"⏱ Gemini квота осталась: {remaining}")
                    if remaining <= 0:
                        logging.warning("⚠️ Квота Gemini исчерпана, используем Ollama")
                        return await summarize_ollama(text)
    except Exception as e:
        logging.warning(f"⚠️ Ошибка проверки квоты Gemini: {e}")

    # Вызов Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={AI_STUDIO_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"Сделай краткое резюме новости:\n{short_text}"}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                if resp.status == 429 or "quotaExceeded" in str(result).lower():
                    logging.warning("⚠️ Квота Gemini исчерпана, fallback на Ollama")
                    return await summarize_ollama(text)

                candidates = result.get("candidates")
                if not candidates or not isinstance(candidates, list):
                    logging.warning("⚠️ Нет candidates в ответе Gemini, fallback на Ollama")
                    return await summarize_ollama(text)

                text_out = candidates[0].get("content", {}).get("parts", [{}])[0].get("text")
                if not text_out:
                    logging.warning("⚠️ Пустой текст от Gemini, fallback на Ollama")
                    return await summarize_ollama(text)

                logging.info(f"✅ Получено резюме Gemini: {text_out[:100]}...")
                return text_out

    except Exception as e:
        logging.warning(f"⚠️ Ошибка Gemini: {e}, fallback на Ollama")
        return await summarize_ollama(text)

# ---------------- Other helpers ----------------
async def check_sources():
    results = await asyncio.gather(*[fetch_and_check(url, head_only=True) for url in RSS_URLS])
    logging.info("🔍 Проверка источников:")
    for u,s in results: logging.info(f"  {s} — {u}")

async def send_news():
    all_news=[]
    if os.path.exists("news_queue.json"):
        try:
            with open("news_queue.json","r",encoding="utf-8") as f:
                queued=json.load(f)
            all_news.extend([(t,l,s,datetime.fromisoformat(p)) for t,l,s,p in queued])
            os.remove("news_queue.json")
        except: pass

    results = await asyncio.gather(*[fetch_and_check(url) for url in RSS_URLS])
    for r in results: all_news.extend(r)
    if not all_news: return

    cutoff=datetime.now(timezone.utc)-timedelta(days=DAYS_LIMIT)
    all_news=[n for n in all_news if n[3] and n[3]>=cutoff]

    try:
        with open(SENT_LINKS_FILE,"r",encoding="utf-8") as f: data=json.load(f)
        sent_links=data.get("links",{}); last_index=data.get("last_source_index",0)
    except: sent_links,last_index={},0
    sent_links={k:v for k,v in sent_links.items() if datetime.strptime(v,"%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)>=cutoff}

    if ROUND_ROBIN_MODE:
        sources=defaultdict(deque)
        for t,l,s,p in sorted(all_news,key=lambda x:x[3],reverse=True): sources[s].append((t,l,s,p))
        src_list=list(sources.keys()); queue,i=[],last_index
        while any(sources.values()):
            s=src_list[i%len(src_list)]
            if sources[s]: queue.append(sources[s].popleft())
            i+=1
        new_items=[n for n in queue if n[1] not in sent_links]
    else: new_items=[n for n in sorted(all_news,key=lambda x:x[3],reverse=True) if n[1] not in sent_links]

    total = len(new_items)
    if total <= BATCH_SIZE_SMALL: pause=PAUSE_SMALL
    elif total <= BATCH_SIZE_MEDIUM: pause=PAUSE_MEDIUM
    else: pause=PAUSE_LARGE

    current_batch = new_items[:NEWS_LIMIT or total]
    queue_rest = new_items[NEWS_LIMIT or total:]

    sent_count=0
    for t,l,s,p in current_batch:
        local_time=(p or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        summary=await summarize(f"{t}\n{l}")
        text=f"<b>{t}</b>\n📡 {s}\n🗓 {local_time}\n\n{summary}\n🔗 {l}"
        if len(text)>4000: text=text[:3990]+"..."
        for _ in range(3):
            try: 
                await bot.send_message(chat_id=CHAT_ID,text=text,parse_mode="HTML")
                sent_links[l]=local_time
                sent_count+=1
                logging.info(f"📤 Новость отправлена в Telegram: {t[:50]}...")
                break
            except Exception as e: 
                logging.error(f"❌ Ошибка отправки: {e}")
                await asyncio.sleep(5)
        await asyncio.sleep(SINGLE_MESSAGE_PAUSE)

    if queue_rest:
        with open("news_queue.json","w",encoding="utf-8") as f:
            json.dump([(t,l,s,p.isoformat()) for t,l,s,p in queue_rest],f,ensure_ascii=False,indent=2)

    save={"links":sent_links}
    if ROUND_ROBIN_MODE and 'src_list' in locals() and src_list: save["last_source_index"]=(last_index+sent_count)%len(src_list)
    tmp=SENT_LINKS_FILE+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(save,f,ensure_ascii=False,indent=2)
    os.replace(tmp,SENT_LINKS_FILE)
    logging.info(f"✅ Отправлено {sent_count}/{len(current_batch)} новостей, пауза перед следующим батчем {pause} сек")
    await asyncio.sleep(pause)

# ---------------- MAIN LOOP ----------------
async def main():
    last_check=datetime.min
    while True:
        now=datetime.now()
        if (now-last_check)>timedelta(days=1):
            await check_sources()
            last_check=now
        logging.info("🔄 Проверка новостей...")
        await send_news()
        logging.info(f"⏰ Следующая проверка через {INTERVAL//60} мин\n")
        print("💤 цикл завершён, жду следующий", flush=True)
        await asyncio.sleep(INTERVAL)

if __name__=="__main__":
    logging.info("🚀 Запуск бота...")
    asyncio.run(main())
