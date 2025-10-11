import os, sys, json, time, asyncio, ssl, logging, subprocess, calendar
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from dotenv import load_dotenv
import aiohttp, feedparser
from telegram import Bot
from bs4 import BeautifulSoup
from article_parser import extract_article_text

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
GEMINI_MODEL = os.environ.get("AI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_MODEL_FALLBACK = os.environ.get("OLLAMA_MODEL_FALLBACK", "gpt-oss:120b")

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
if not RSS_URLS:
    sys.exit("❌ RSS_URLS не заданы")

bot = Bot(token=TELEGRAM_TOKEN)

# ---------------- LOG ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("parser.log", encoding="utf-8")
    ]
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
                if r.status != 200:
                    raise Exception(f"HTTP {r.status}")
                body = await r.text()
                feed = feedparser.parse(body)
                news = []
                for e in feed.entries:
                    pub = None
                    if getattr(e, "published_parsed", None):
                        pub = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=timezone.utc)
                    news.append((
                        e.get("title", "Без заголовка").strip(),
                        e.get("link", "").strip(),
                        feed.feed.get("title", "Неизвестный источник").strip(),
                        pub
                    ))
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

def parse_iso_utc(s):
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc)
    if not s:
        raise ValueError("empty date")
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        for fmt in ("%d.%m.%Y, %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            raise ValueError(f"Неверный формат даты: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

# ---------------- Ollama local ----------------
async def summarize_ollama(text: str):
    short_text = ". ".join(text.split(".")[:3])
    prompt = f"Сделай краткое резюме новости:\n{short_text}"
    logging.debug(f"🧠 [OLLAMA INPUT] {short_text[:500]}...")
    async def run_model(model_name: str):
        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        logging.error(f"⚠️ Ollama {model_name} HTTP {resp.status}")
                        return None, model_name
                    data = await resp.json()
                    output = data.get("response", "").strip()
                    if not output:
                        logging.warning(f"⚠️ Ollama ({model_name}) пустой ответ")
                        return None, model_name
                    elapsed = round(time.time() - start_time, 2)
                    logging.info(f"✅ Ollama ({model_name}) за {elapsed} сек")
                    logging.debug(f"🧠 [OLLAMA OUTPUT] {output[:500]}...")
                    return output, model_name
        except asyncio.TimeoutError:
            logging.error(f"⏰ Ollama ({model_name}) таймаут")
        except Exception as e:
            logging.error(f"❌ Ollama ({model_name}): {e}")
        return None, model_name

    result, used_model = await run_model(OLLAMA_MODEL)
    if not result:
        logging.warning(f"⚠️ Переключаюсь на резервную модель {OLLAMA_MODEL_FALLBACK}")
        result, used_model = await run_model(OLLAMA_MODEL_FALLBACK)

    if not result:
        return short_text[:400] + "...", "local-fallback"

    return result, used_model

# ---------------- Gemini ----------------
async def summarize(text, max_tokens=200, retries=3):
    if not AI_STUDIO_KEY:
        logging.debug(f"🧠 [GEMINI INPUT] {short_text[:500]}...")
        logging.warning("⚠️ AI_STUDIO_KEY не задан, fallback на Ollama")
        return await summarize_ollama(text)

    text = clean_text(text)
    short_text = ". ".join(text.split(".")[:2])
    payload = {
        "contents": [{"parts": [{"text": f"Сделай краткое резюме новости:\n{short_text}"}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": AI_STUDIO_KEY, "Content-Type": "application/json"}

    backoff = 1
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        logging.warning(f"⚠️ Gemini HTTP {resp.status}: {body}")
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    result = json.loads(body)
        except Exception as e:
            logging.warning(f"⚠️ Gemini error: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
            continue

        try:
            candidates = result.get("candidates")
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    text_out = parts[0]["text"]
                    logging.info(f"✅ Gemini OK ({GEMINI_MODEL}): {text_out[:100]}...")
                    logging.debug(f"🧠 [GEMINI OUTPUT] {text_out[:500]}...")
                    return text_out.strip(), GEMINI_MODEL
        except Exception as e:
            logging.warning(f"⚠️ Ошибка парсинга Gemini: {e}")
    logging.error("❌ Gemini не ответил, fallback на Ollama")
    return await summarize_ollama(text)

# ---------------- Проверка источников ----------------
async def check_sources():
    results = await asyncio.gather(*[fetch_and_check(url, head_only=True) for url in RSS_URLS])
    logging.info("🔍 Проверка источников:")
    for u, s in results:
        logging.info(f"  {s} — {u}")

# ---------------- Telegram helper ----------------
async def send_telegram(text):
    try:
        if asyncio.iscoroutinefunction(bot.send_message):
            await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML")
        else:
            await asyncio.to_thread(lambda: bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML"))
    except Exception as e:
        raise e

# ---------------- Основная логика ----------------
async def send_news():
    all_news = []
    if os.path.exists("news_queue.json"):
        try:
            with open("news_queue.json", "r", encoding="utf-8") as f:
                queued = json.load(f)
            all_news.extend([(t, l, s, datetime.fromisoformat(p)) for t, l, s, p in queued])
            os.remove("news_queue.json")
        except Exception:
            pass

    results = await asyncio.gather(*[fetch_and_check(url) for url in RSS_URLS])
    for r in results:
        all_news.extend(r)
    if not all_news:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_LIMIT)

    try:
        with open(SENT_LINKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        sent_links = data.get("links", {})
        last_index = data.get("last_source_index", 0)
    except Exception:
        sent_links, last_index = {}, 0

    clean_sent = {}
    for k, v in sent_links.items():
        try:
            if parse_iso_utc(v) >= cutoff:
                clean_sent[k] = v
        except Exception:
            continue
    sent_links = clean_sent

    if ROUND_ROBIN_MODE:
        sources = defaultdict(deque)
        for t, l, s, p in sorted(all_news, key=lambda x: x[3], reverse=True):
            sources[s].append((t, l, s, p))
        src_list = list(sources.keys())
        queue, i = [], last_index
        while any(sources.values()):
            s = src_list[i % len(src_list)]
            if sources[s]:
                queue.append(sources[s].popleft())
            i += 1
        new_items = [n for n in queue if n[1] not in sent_links]
    else:
        new_items = [n for n in sorted(all_news, key=lambda x: x[3], reverse=True)
                     if n[1] not in sent_links]

    total = len(new_items)
    if total <= BATCH_SIZE_SMALL:
        pause = PAUSE_SMALL
    elif total <= BATCH_SIZE_MEDIUM:
        pause = PAUSE_MEDIUM
    else:
        pause = PAUSE_LARGE

    current_batch = new_items[:NEWS_LIMIT or total]
    queue_rest = new_items[NEWS_LIMIT or total:]

    sent_count = 0
    for t, l, s, p in current_batch:
        local_time = (p or datetime.now(timezone.utc)).astimezone(timezone.utc)
        local_time_str = local_time.strftime("%d.%m.%Y, %H:%M")

        try:
            article_text = await extract_article_text(l, ssl_ctx)
        except Exception as e:
            logging.warning(f"extract_article_text error for {l}: {e}")
            article_text = None

        content = article_text if article_text else f"{t}\n{l}"
        summary_text, used_model = await summarize(content)

        MAX_SUMMARY_LEN = 600
        MAX_TITLE_LEN = 120
        title_clean = t.strip()
        if len(title_clean) > MAX_TITLE_LEN:
            title_clean = title_clean[:MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"
        summary_clean = summary_text.strip()
        if len(summary_clean) > MAX_SUMMARY_LEN:
            summary_clean = summary_clean[:MAX_SUMMARY_LEN].rsplit(" ", 1)[0] + "…"

        text = (
            f"<b>{title_clean}</b>\n"
            f"📡 <i>{s}</i> | 🗓 {local_time_str}\n"
            f"━━━━━━━━━━━━━━━\n\n"
            f"💬 {summary_clean}\n\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🤖 <i>Модель: {used_model}</i>\n"
            f"🔗 <a href=\"{l}\">Читать статью</a>"
        )

        for _ in range(3):
            try:
                await send_telegram(text)
                sent_links[l] = (p or datetime.now(timezone.utc)).isoformat()
                sent_count += 1
                logging.info(f"📤 Отправлено: {title_clean[:50]}...")
                break
            except Exception as e:
                logging.error(f"❌ Ошибка отправки: {e}")
                if "429" in str(e):
                    await asyncio.sleep(30)
        else:
            await asyncio.sleep(5)

    if queue_rest:
        with open("news_queue.json", "w", encoding="utf-8") as f:
            json.dump([(t, l, s, p.isoformat()) for t, l, s, p in queue_rest],
                      f, ensure_ascii=False, indent=2)

    save = {"links": sent_links}
    if ROUND_ROBIN_MODE and 'src_list' in locals() and src_list:
        save["last_source_index"] = (last_index + sent_count) % len(src_list)
    tmp = SENT_LINKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SENT_LINKS_FILE)

    logging.info(f"✅ Отправлено {sent_count}/{len(current_batch)} новостей. Пауза {pause} сек")
    await asyncio.sleep(pause)

# ---------------- MAIN LOOP ----------------
async def main():
    last_check = datetime.min
    while True:
        now = datetime.now()
        if (now - last_check) > timedelta(days=1):
            await check_sources()
            last_check = now
        logging.info("🔄 Проверка новостей...")
        await send_news()
        logging.info(f"⏰ Следующая проверка через {INTERVAL // 60} мин\n")
        print("💤 цикл завершён, жду следующий", flush=True)
        await asyncio.sleep(INTERVAL)

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger().setLevel(logging.DEBUG)

    print("🚀 Запуск бота...", flush=True)
    logging.info("🚀 Логгер инициализирован")
    asyncio.run(main())
