import os, sys, json, time, asyncio, ssl, logging, subprocess, calendar
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from dotenv import load_dotenv
import aiohttp, feedparser
from telegram import Bot
from bs4 import BeautifulSoup
from article_parser import extract_article_text
from utils import send_long_message

# ---- dedup seen links across restarts ----
SEEN_FILE = "seen.json"
if os.path.exists(SEEN_FILE):
    try:
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            seen_links = set(json.load(f))
    except Exception:
        seen_links = set()
else:
    seen_links = set()

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
PARSER_MAX_TEXT_LENGTH = int(os.environ.get("PARSER_MAX_TEXT_LENGTH",
                                           os.environ.get("MAX_TEXT_LENGTH", "10000")))
# legacy alias
MAX_TEXT_LENGTH = PARSER_MAX_TEXT_LENGTH
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 1200))

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

LOG_FILE = "parser.log"

# --- Цвета для терминала ---
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[94m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m"
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_color = COLORS.get(record.levelname, "")
        msg = super().format(record)
        return f"{level_color}{msg}{RESET}"

# --- Формат ---
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# --- Файл (без цвета) ---
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

# --- Терминал (с цветом) ---
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s | %(message)s"))

# --- Настройка корневого логгера ---
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# --- Отдельный логгер для моделей ---
model_logger = logging.getLogger("model")
model_logger.setLevel(logging.INFO)
model_logger.addHandler(console_handler)
model_logger.addHandler(file_handler)
model_logger.propagate = False

# ---------------- SSL CONTEXT ----------------
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

# ---------------- HELPERS ----------------
async def fetch_and_check(session, url, head_only=False):
    """Асинхронная загрузка RSS: HEAD или GET, с fallback."""
    try:
        if head_only:
            async with session.head(url, ssl=ssl_ctx) as r:
                if r.status == 405:  # fallback
                    async with session.get(url, ssl=ssl_ctx) as r2:
                        return url, "✅ OK" if r2.status == 200 else f"⚠️ HTTP {r2.status}"
                return url, "✅ OK" if r.status == 200 else f"⚠️ HTTP {r.status}"

        async with session.get(url, ssl=ssl_ctx) as r:
            if r.status != 200:
                raise Exception(f"HTTP {r.status}")
            body = await r.text()
            feed = feedparser.parse(body)
            news = []
            for e in feed.entries:
                pub = None
                if getattr(e, "published_parsed", None):
                    pub = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=timezone.utc)
                summary = e.get("summary", "") or e.get("description", "") or ""
                news.append((
                    e.get("title", "Без заголовка").strip(),
                    e.get("link", "").strip(),
                    feed.feed.get("title", "Неизвестный источник").strip(),
                    summary,
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

from html import escape
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
    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]
    prompt = f"Не делай вступлений. Сделай резюме новости на русском языке:\n{prompt_text}"
    logging.info(f"🧠 [OLLAMA INPUT] >>> {prompt_text[:5500]}")
    async def run_model(model_name: str):
        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "options": {"num_predict": MODEL_MAX_TOKENS}}
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
                    logging.info(f"🧠 [OLLAMA OUTPUT] <<< {output}")
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
        # Вернуть начало переданного prompt_text как fallback (увеличено до 2000 символов)
        return prompt_text[:2000] + "...", "local-fallback"

    return result, used_model

# ---------------- Gemini ----------------
async def summarize(text, max_tokens=200, retries=3):
    text = clean_text(text)
    # используем весь контент (но не более PARSER_MAX_TEXT_LENGTH)
    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]

    # --- добавляем явное указание на русский язык и формат результата ---
    prompt_text = f"Сделай краткое резюме новости на русском языке:\n{prompt_text}"

    if not AI_STUDIO_KEY:
        logging.debug(f"🧠 [GEMINI INPUT] {prompt_text[:500]}...")
        logging.warning("⚠️ AI_STUDIO_KEY не задан, fallback на Ollama")
        return await summarize_ollama(text)

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens or MODEL_MAX_TOKENS}
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": AI_STUDIO_KEY, "Content-Type": "application/json"}

    backoff = 1
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"🧠 [GEMINI INPUT] >>> {prompt_text[:500]}")
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
                    logging.info(f"✅ Gemini OK ({GEMINI_MODEL}): {text_out}")
                    logging.info(f"🧠 [GEMINI OUTPUT] <<< {text_out}")
                    return text_out.strip(), GEMINI_MODEL
        except Exception as e:
            logging.warning(f"⚠️ Ошибка парсинга Gemini: {e}")
    logging.error("❌ Gemini не ответил, fallback на Ollama")
    return await summarize_ollama(text)

# ---------------- Проверка источников ----------------
async def check_sources():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        results = await asyncio.gather(*[fetch_and_check(session, url, head_only=True) for url in RSS_URLS])
    logging.info("🔍 Проверка источников:")
    for u, s in results:
        logging.info(f"  {s} — {u}")

# ---------------- Telegram helper ----------------
async def send_telegram(text):
    # Делегируем разбиение и отправку в utils.send_long_message
    await send_long_message(bot, CHAT_ID, text, parse_mode="HTML", delay=SINGLE_MESSAGE_PAUSE)

# ---------------- Основная логика ----------------
async def send_news():
    all_news = []
    if os.path.exists("news_queue.json"):
        try:
            with open("news_queue.json", "r", encoding="utf-8") as f:
                queued = json.load(f)
            # Поддерживаем старый формат (t, l, s, p) и новый (t, l, s, summary, p)
            for item in queued:
                if len(item) == 4:
                    t, l, s, p = item
                    all_news.append((t, l, s, "", datetime.fromisoformat(p)))
                elif len(item) == 5:
                    t, l, s, summary, p = item
                    all_news.append((t, l, s, summary, datetime.fromisoformat(p)))
            os.remove("news_queue.json")
        except Exception:
            pass

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        results = await asyncio.gather(*[fetch_and_check(session, url) for url in RSS_URLS])
    for r in results:
        # `r` может быть либо [] при ошибке, либо список кортежей из fetch_and_check
        # Убедимся, что элементы имеют 5 элементов (t, l, s, summary, pub)
        for it in r:
            if len(it) == 4:
                # старый формат без summary
                t, l, s, p = it
                all_news.append((t, l, s, "", p))
            elif len(it) == 5:
                all_news.append(it)
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
        for t, l, s, summary, p in sorted(all_news, key=lambda x: x[4], reverse=True):
            sources[s].append((t, l, s, summary, p))
        src_list = list(sources.keys())
        queue, i = [], last_index
        while any(sources.values()):
            s = src_list[i % len(src_list)]
            if sources[s]:
                queue.append(sources[s].popleft())
            i += 1
        new_items = [n for n in queue if n[1] not in sent_links]
    else:
        # безопасная сортировка даже если p = None
        MIN_DT = datetime.fromtimestamp(0, tz=timezone.utc)
        new_items = [
            n for n in sorted(all_news, key=lambda x: x[4] or MIN_DT, reverse=True)
            if n[1] not in sent_links
        ]

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
    # ---------------- Обработка статьи и резюме ----------------
    for item in current_batch:
        if len(item) == 5:
            t, l, s, summary, p = item
        else:
            t, l, s, summary, p = item[0], item[1], item[2], "", item[3]

        local_time = (p or datetime.now(timezone.utc)).astimezone(timezone.utc)
        local_time_str = local_time.strftime("%d.%m.%Y, %H:%M")

        try:
            # --- Извлекаем текст статьи безопасно ---
            import inspect
            if inspect.iscoroutinefunction(extract_article_text):
                article_text = await extract_article_text(l, ssl_ctx, max_length=PARSER_MAX_TEXT_LENGTH)
            else:
                article_text = await asyncio.to_thread(extract_article_text, l, ssl_ctx, PARSER_MAX_TEXT_LENGTH)
        except Exception as e:
            logging.warning(f"extract_article_text error for {l}: {e}")
            article_text = None

        # --- Подготавливаем контент для модели ---
        if not article_text or len(article_text) < 300:
            # Если текста мало, берем summary или title
            content = f"{summary or t}"
        else:
            content = article_text

        # --- Усечение контента по PARSER_MAX_TEXT_LENGTH перед моделью ---
        content = content[:PARSER_MAX_TEXT_LENGTH]
        logging.debug(f"📝 Контент для модели ({len(content)} символов): {content}")

        # --- Генерация ---
        summary_text, used_model = await summarize(content, max_tokens=MODEL_MAX_TOKENS)

        # --- Усечение заголовка и резюме ---
        MAX_SUMMARY_LEN = 1200
        MAX_TITLE_LEN = 120
        title_clean = t.strip()
        if len(title_clean) > MAX_TITLE_LEN:
            title_clean = title_clean[:MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"
        summary_clean = summary_text.strip()
        if len(summary_clean) > MAX_SUMMARY_LEN:
            summary_clean = summary_clean[:MAX_SUMMARY_LEN].rsplit(" ", 1)[0] + "…"

        # --- Формируем сообщение для Telegram ---
        title_safe = escape(title_clean)
        summary_safe = escape(summary_clean)
        link_safe = escape(l, quote=True)

        # ✅ новый безопасный сплит сообщений: header/body/footer
        def split_message_simple(text: str, limit: int = 4096) -> list[str]:
            parts = []
            while len(text) > limit:
                pos = text.rfind("\n", 0, limit)
                if pos == -1:
                    pos = text.rfind(" ", 0, limit)
                if pos == -1:
                    pos = limit
                parts.append(text[:pos].strip())
                text = text[pos:].strip()
            if text:
                parts.append(text)
            return parts

        header = f"<b>{title_safe}</b>\n📡 <i>{s}</i> | 🗓 {local_time_str}\n━━━━━━━━━━━━━━━\n"
        body = f"💬 {summary_safe}"
        footer = f"\n━━━━━━━━━━━━━━━\n🤖 <i>Модель: {used_model}</i>\n🔗 <a href=\"{link_safe}\">Читать статью</a>"

        # Оставляем резерв в 200 символов на header/footer/markup
        parts = split_message_simple(body, limit=4096 - 200)
        assembled_parts = []
        for i, part in enumerate(parts):
            if len(parts) == 1:
                msg = header + part + footer
            elif i == 0:
                msg = header + part
            elif i == len(parts) - 1:
                msg = part + footer
            else:
                msg = part
            assembled_parts.append(msg)

        # Проверяем, отправляли ли уже ссылку ранее (между перезапусками)
        if l in seen_links:
            logging.debug(f"🔁 Пропускаю уже отправленную ссылку: {l}")
            continue

        for _ in range(3):
            try:
                # универсальная безопасная отправка (async/sync)
                import inspect
                async def send_msg(part_msg):
                    fn = getattr(bot, "send_message", None)
                    # Печатаем в терминал финальный текст сообщения для отладки
                    try:
                        print(part_msg)
                    except Exception:
                        pass
                    if inspect.iscoroutinefunction(fn):
                        await fn(chat_id=CHAT_ID, text=part_msg, parse_mode="HTML")
                    else:
                        await asyncio.to_thread(fn, chat_id=CHAT_ID, text=part_msg, parse_mode="HTML")

                for part_msg in assembled_parts:
                    await send_msg(part_msg)
                    await asyncio.sleep(SINGLE_MESSAGE_PAUSE)

                sent_links[l] = (p or datetime.now(timezone.utc)).isoformat()
                seen_links.add(l)
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
            safe_queue = []
            for item in queue_rest:
                if len(item) == 5:
                    t, l, s, summary, p = item
                elif len(item) == 4:
                    t, l, s, p = item
                    summary = ""
                else:
                    continue

                # если p может быть строкой — приводи к iso
                if isinstance(p, str):
                    try:
                        p = datetime.fromisoformat(p)
                    except Exception:
                        pass

                iso_p = p.isoformat() if hasattr(p, "isoformat") else str(p)
                safe_queue.append((t, l, s, summary, iso_p))

            with open("news_queue.json", "w", encoding="utf-8") as f:
                json.dump(safe_queue, f, ensure_ascii=False, indent=2)

          
    save = {"links": sent_links}
    if ROUND_ROBIN_MODE and 'src_list' in locals() and src_list:
        save["last_source_index"] = (last_index + sent_count) % len(src_list)
    tmp = SENT_LINKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SENT_LINKS_FILE)

    # Сохраняем список уже отправленных ссылок между перезапусками
    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(list(seen_links), f, ensure_ascii=False, indent=2)
    except Exception:
        logging.warning("⚠️ Не удалось сохранить seen.json")

    logging.info(f"✅ Отправлено {sent_count}/{len(current_batch)} новостей. Пауза {pause} сек")
    await asyncio.sleep(pause)

# ---------------- MAIN LOOP ----------------
async def main():
    last_check = datetime.now(timezone.utc)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        try:
            while True:
                now = datetime.now(timezone.utc)
                if (now - last_check) > timedelta(days=1):
                    await check_sources()
                    last_check = now
                logging.info("🔄 Проверка новостей...")
                await send_news()
                logging.info(f"⏰ Следующая проверка через {INTERVAL // 60} мин\n")
                print("💤 цикл завершён, жду следующий", flush=True)
                await asyncio.sleep(INTERVAL)
        except KeyboardInterrupt:
            logging.info("🛑 Завершение по Ctrl+C, сохраняем state…")
        finally:
            try:
                with open(SEEN_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(seen_links), f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning(f"⚠️ Не удалось сохранить seen.json при выходе: {e}")

if __name__ == "__main__":
    print("🚀 Запуск бота...", flush=True)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info(f"💬 MODEL_MAX_TOKENS = {MODEL_MAX_TOKENS}")
    logging.info(f"📰 PARSER_MAX_TEXT_LENGTH = {PARSER_MAX_TEXT_LENGTH}")
    asyncio.run(main())
