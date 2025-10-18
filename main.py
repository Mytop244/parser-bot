import os, sys, json, time, asyncio, ssl, logging, subprocess, calendar
import tempfile
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from dotenv import load_dotenv
import aiohttp, feedparser
from telegram import Bot
from bs4 import BeautifulSoup
from functools import partial
import re, html

# ---- small cache and helpers from utils.py ----
HTML_SAFE_LIMIT = 4096  # Telegram limit
_cache = {}

def split_text_safe(text: str, limit: int) -> list[str]:
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

async def send_long_message(bot, chat_id: int, text: str, parse_mode="HTML", delay: int = 1):
    """Отправляет длинный текст в Telegram частями, сохраняя HTML-разметку.

    Использует `_cache` для повторных вызовов с тем же текстом.
    """
    if text in _cache:
        parts = _cache[text]
    else:
        paragraphs = text.split("\n")
        parts, current = [], ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 1 < HTML_SAFE_LIMIT:
                current += ("" if not current else "\n") + para
            else:
                if current:
                    parts.append(current)
                if len(para) >= HTML_SAFE_LIMIT:
                    parts.extend(split_text_safe(para, HTML_SAFE_LIMIT))
                    current = ""
                else:
                    current = para
        if current:
            parts.append(current)
        _cache[text] = parts

    for part in parts:
        try:
            logging.info((part[:120] + "…") if len(part) > 120 else part)
        except Exception:
            pass
        try:
            await bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
        except Exception as e:
            logging.error(f"Ошибка при отправке сообщения: {e}")
        await asyncio.sleep(delay)

# ---- Inlined extract_article_text (from article_parser.py) ----
async def extract_article_text(
    url: str,
    ssl_context=None,
    max_length: int = 5000,
    session: aiohttp.ClientSession | None = None
) -> str:
    """Извлекает текст статьи с ретраями и fallback-экстракторами."""
    ctx = ssl_context if ssl_context is not None else ssl_ctx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NewsBot/1.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    MAX_DOWNLOAD = max(200_000, min(1_000_000, max_length * 200))

    async def _read_limited(resp, max_bytes):
        chunks, size = [], 0
        async for chunk in resp.content.iter_chunked(8192):
            chunks.append(chunk)
            size += len(chunk)
            if size >= max_bytes:
                logging.debug(f"⚠️ HTML truncated at {size} bytes for {url}")
                break
        try:
            return b"".join(chunks).decode(errors="ignore")
        except Exception:
            return b"".join(chunks).decode("utf-8", errors="ignore")

    html = ""
    backoff = 1
    for attempt in range(1, 4):
        try:
            if session is None:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), headers=headers) as s:
                    async with s.get(url, ssl=ctx) as r:
                        if r.status != 200:
                            logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                            return ""
                        html = await _read_limited(r, MAX_DOWNLOAD)
            else:
                async with session.get(url, ssl=ctx, headers=headers) as r:
                    if r.status != 200:
                        logging.warning(f"⚠️ HTTP {r.status} при загрузке {url}")
                        return ""
                    html = await _read_limited(r, MAX_DOWNLOAD)
            break
        except Exception as e:
            logging.debug(f"load attempt {attempt} failed for {url}: {e}")
            if attempt < 3:
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                logging.warning(f"⚠️ Ошибка загрузки {url}: {e}")
                return ""

    if not html.strip():
        logging.warning(f"⚠️ Пустой HTML для {url}")
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "aside", "form"]):
        tag.decompose()

    article = soup.find("article")
    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    else:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]

    joined = " ".join(paragraphs).strip()
    text = clean_text(joined)
    logging.debug(f"📝 <p> text length: {len(text)} for {url}")

    if len(text.split()) >= 50:
        out = text[:max_length].rsplit(" ", 1)[0]
        logging.info(f"✅ Returned <p> text ({len(out)} chars) for {url}")
        return out

    loop = asyncio.get_running_loop()

    try:
        import trafilatura
        def trafilatura_extract(html_inner):
            return trafilatura.extract(html_inner, include_comments=False, favor_recall=True)
        extracted = await loop.run_in_executor(None, partial(trafilatura_extract, html))
        logging.debug(f"🧩 trafilatura length: {len(extracted) if extracted else 0} for {url}")
        if extracted and len(extracted.split()) >= 30:
            out = clean_text(extracted)[:max_length]
            logging.info(f"✅ Returned trafilatura text ({len(out)} chars) for {url}")
            return out.rsplit(" ", 1)[0] if " " in out else out
    except Exception as e:
        logging.debug(f"trafilatura fail: {e}")

    try:
        from readability import Document
        def readability_extract(html_inner):
            doc = Document(html_inner)
            summary_html = doc.summary()
            return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
        extracted = await loop.run_in_executor(None, partial(readability_extract, html))
        logging.debug(f"📖 readability length: {len(extracted) if extracted else 0} for {url}")
        if extracted and len(extracted.split()) >= 30:
            out = clean_text(extracted)[:max_length]
            logging.info(f"✅ Returned readability text ({len(out)} chars) for {url}")
            return out.rsplit(" ", 1)[0] if " " in out else out
    except Exception as e:
        logging.debug(f"readability fail: {e}")

    meta = (soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", property="og:description") or
            soup.find("meta", property="twitter:description"))
    if meta and meta.get("content"):
        meta_text = clean_text(meta.get("content", ""))
        logging.debug(f"🪶 meta fallback length: {len(meta_text)} for {url}")
        out = meta_text[:min(max_length, 1000)].rsplit(" ", 1)[0]
        logging.info(f"✅ Returned meta fallback ({len(out)} chars) for {url}")
        return out

    logging.warning(f"⚠️ Не удалось извлечь текст для {url}")
    return ""

# ---- load env early, миграция старых файлов ----
load_dotenv()

# Переносим legacy файлов в единый state.json если они есть
LEGACY_SEEN = "seen.json"
LEGACY_SENT = "sent_links.json"

STATE_FILE = "state.json"
# STATE_DAYS_LIMIT нужно брать _после_ load_dotenv
STATE_DAYS_LIMIT = int(os.getenv("STATE_DAYS_LIMIT", "3"))

def migrate_legacy_files():
    """Перенос старых seen/sent файлов в новый state.json"""
    migrated = False
    state_local = {"seen": {}, "sent": {}}
    try:
        if os.path.exists(LEGACY_SEEN):
            with open(LEGACY_SEEN, "r", encoding="utf-8") as f:
                state_local["seen"] = json.load(f)
            migrated = True
        if os.path.exists(LEGACY_SENT):
            with open(LEGACY_SENT, "r", encoding="utf-8") as f:
                state_local["sent"] = json.load(f)
            migrated = True
        if migrated:
            try:
                save_state_atomic(state_local, STATE_FILE)
                # Удаляем legacy только после успешной записи
                if os.path.exists(LEGACY_SEEN): os.remove(LEGACY_SEEN)
                if os.path.exists(LEGACY_SENT): os.remove(LEGACY_SENT)
                logging.info("✅ Миграция выполнена")
            except Exception:
                logging.exception("⚠️ Ошибка при записи state.json во время миграции")
        else:
            logging.info("🟡 Миграция не требуется")
    except Exception as e:
        logging.exception(f"Не удалось мигрировать старые файлы: {e}")

# state stores timestamps (epoch seconds) for seen and sent links
state = {"seen": {}, "sent": {}}

if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            state["seen"] = data.get("seen", {}) or {}
            state["sent"] = data.get("sent", {}) or {}
    except Exception:
        state = {"seen": {}, "sent": {}}

def cleanup_state():
    now = time.time()
    cutoff = now - STATE_DAYS_LIMIT * 86400
    if "meta" in state:
        return
    for k in ("seen", "sent"):
        state[k] = {url: ts for url, ts in state.get(k, {}).items() if ts >= cutoff}

_state_lock = None  # создаётся лениво

async def _ensure_lock():
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


async def save_state_async():
    """Асинхронное сохранение state.json используя aiofiles (не блокирует event-loop).

    Пишем во временный файл, затем атомарно заменяем целевой.
    """
    import aiofiles
    lock = await _ensure_lock()
    async with lock:
        cleanup_state()
        tmp_path = None
        try:
            # создаём временный файл в той же директории
            fd, tmp_path = tempfile.mkstemp(prefix="state_", suffix=".json", dir=".")
            os.close(fd)
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(state, ensure_ascii=False, indent=2))
                # aiofiles не даёт fsync напрямую; закрытие файла выполнит запись в ОС
            # атомарная замена
            os.replace(tmp_path, STATE_FILE)
            tmp_path = None
        except Exception:
            logging.exception("Не удалось сохранить state.json")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


def save_state_atomic(data, path="state.json"):
    """Atomically save JSON to `path` by writing to a temp file and os.replace."""
    fd, tmp = tempfile.mkstemp(prefix="tmp_state_", dir=os.path.dirname(path) or ".")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def mark_state(kind: str, key: str, value):
    """Помечает состояние и сохраняет"""
    if kind not in state:
        state[kind] = {}
    state[kind][key] = value
    # безопасная проверка event loop
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.create_task(save_state_async())
    else:
        try:
            save_state_atomic(state, STATE_FILE)
        except Exception:
            try:
                asyncio.run(save_state_async())
            except Exception:
                logging.exception("Не удалось сохранить state при mark_state")


# Теперь можно безопасно вызывать миграцию (save_state_atomic уже определён)
migrate_legacy_files()


# ---------------- ENV ----------------
if hasattr(time, "tzset"):
    os.environ["TZ"] = os.environ.get("TIMEZONE", "UTC")
    time.tzset()
else:
    logging.info("⏰ Windows: пропускаем установку TZ (tzset недоступен)")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
# Безопасный разбор CHAT_ID: сначала читаем как строку, затем пытаемся привести к int.
raw_chat = os.environ.get("CHAT_ID")
CHAT_ID = None
if raw_chat is not None and raw_chat != "":
    try:
        CHAT_ID = int(raw_chat)
    except Exception:
        sys.exit("❌ CHAT_ID должен быть целым числом")
RSS_URLS = [u.strip() for u in os.environ.get("RSS_URLS", "").split(",") if u.strip()]
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT", 5))
INTERVAL = int(os.environ.get("INTERVAL", 600))
SENT_LINKS_FILE = STATE_FILE
DAYS_LIMIT = int(os.environ.get("DAYS_LIMIT", 1))
ROUND_ROBIN_MODE = int(os.environ.get("ROUND_ROBIN_MODE", 1))
AI_STUDIO_KEY = os.environ.get("AI_STUDIO_KEY")
GEMINI_MODEL = os.environ.get("AI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_MODEL_FALLBACK = os.environ.get("OLLAMA_MODEL_FALLBACK", "gpt-oss:120b")
PARSER_MAX_TEXT_LENGTH = int(os.environ.get("PARSER_MAX_TEXT_LENGTH",
                                           os.environ.get("MAX_TEXT_LENGTH", "10000")))
# Ollama timeout (seconds)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 180))
# legacy alias
MAX_TEXT_LENGTH = PARSER_MAX_TEXT_LENGTH
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 1200))

# ---------------- Prompt templates (can be overridden in .env)
GEMINI_PROMPT = os.getenv("GEMINI_PROMPT",
    "Сделай профессиональное краткое резюме новости на русском языке, без вступления, дели на абзацы:\n{content}")
OLLAMA_PROMPT = os.getenv("OLLAMA_PROMPT",
    "Не делай вступлений. Сделай резюме новости на русском языке:\n{content}")

# Модельные лимиты (можно переопределить в .env)
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", 500))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", 500))

# Какая модель считается активной по умолчанию (можно задать через .env)
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", GEMINI_MODEL)

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

PARSE_MODE = os.getenv("PARSE_MODE", "HTML")

# ---------------- Message templates (can be overridden in .env)
HEADER_TEMPLATE = os.getenv("HEADER_TEMPLATE",
    "<b>{title}</b>\n📡 <i>{source}</i> | 🗓 {date}\n━━━━━━━━━━━━━━━\n")
FOOTER_TEMPLATE = os.getenv("FOOTER_TEMPLATE",
    "\n━━━━━━━━━━━━━━━\n🤖 <i>Модель: {model}</i>\n🔗 <a href=\"{link}\">Читать статью</a>")
BODY_PREFIX = os.getenv("BODY_PREFIX", "💬 ")

async def send(chat_id, text):
    for part in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        await bot.send_message(chat_id, part, parse_mode=PARSE_MODE)

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
file_handler.setLevel(logging.INFO)

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

# --- Безопасный SSL ---
SSL_VERIFY = os.getenv("SSL_VERIFY", "1") not in ("0", "false", "False")
ssl_ctx = ssl.create_default_context()
if not SSL_VERIFY:
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
    """Парсит дату в timezone-aware UTC datetime"""
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
        dt = None
        for fmt in ("%d.%m.%Y, %H:%M", "%Y-%m-%d %H:%M",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
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
    prompt = OLLAMA_PROMPT.format(content=prompt_text)
    logging.info(f"🧠 [OLLAMA INPUT] >>> {prompt_text[:5500]}")
    async def run_model(model_name: str):
        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": model_name, "prompt": prompt, "options": {"num_predict": MODEL_MAX_TOKENS}}
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as resp:
                    if resp.status != 200:
                        logging.error(f"⚠️ Ollama {model_name} HTTP {resp.status}")
                        return None, model_name

                    # Поддержка NDJSON / stream: аккумулируем все поля `response` из приходящих JSON-объектов
                    text = ""
                    try:
                        async for chunk in resp.content:
                            if not chunk:
                                continue
                            try:
                                s = chunk.decode("utf-8")
                            except Exception:
                                # если не можем декодировать — пропускаем
                                continue
                            # строка может содержать несколько JSON-объектов, разберём по строкам
                            for line in s.splitlines():
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                except Exception:
                                    # если это не валидный JSON — пропускаем
                                    continue
                                text += data.get("response", "")
                    except Exception as e:
                        logging.error(f"❌ Ollama ({model_name}) stream error: {e}")
                        return None, model_name

                    output = text.strip()
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
    prompt_text = GEMINI_PROMPT.format(content=prompt_text)

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
                    if resp.status == 429:
                        logging.warning("⚠️ Gemini quota exceeded — fallback to Ollama")
                        return await summarize_ollama(text)
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
                    try:
                        if isinstance(p, str):
                            try:
                                p = parse_iso_utc(p)
                            except Exception:
                                try:
                                    p = parse_iso_utc(p)
                                except Exception:
                                    p = None
                        elif isinstance(p, datetime) and p.tzinfo is None:
                            p = p.replace(tzinfo=timezone.utc)
                        all_news.append((t, l, s, "", p))
                    except Exception:
                        continue
                elif len(item) == 5:
                    t, l, s, summary, p = item
                    try:
                        if isinstance(p, str):
                            try:
                                p_dt = parse_iso_utc(p)
                            except Exception:
                                p_dt = None
                        elif isinstance(p, datetime):
                            if p.tzinfo is None:
                                p_dt = p.replace(tzinfo=timezone.utc)
                            else:
                                p_dt = p.astimezone(timezone.utc)
                    except Exception:
                        p_dt = None
                    all_news.append((t, l, s, summary, p_dt))
            try:
                os.remove("news_queue.json")
            except Exception:
                pass
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

    # Используем единое хранилище state для sent links
    sent_links = state.get("sent", {})
    last_index = state.get("meta", {}).get("last_source_index", 0)

    clean_sent = {}
    cutoff_ts = cutoff.timestamp()
    for k, v in sent_links.items():
        try:
            if isinstance(v, (int, float)):
                if v >= cutoff_ts:
                    clean_sent[k] = v
            else:
                if parse_iso_utc(v) >= cutoff:
                    clean_sent[k] = v
        except Exception:
            continue
    sent_links = clean_sent

    if ROUND_ROBIN_MODE:
        sources = defaultdict(deque)
        MIN_DT = datetime.fromtimestamp(0, tz=timezone.utc)
        for t, l, s, summary, p in sorted(all_news, key=lambda x: x[4] or MIN_DT, reverse=True):

            sources[s].append((t, l, s, summary, p))
        src_list = list(sources.keys())
        queue, i = [], last_index
        if src_list:
            while any(sources.values()):
                s = src_list[i % len(src_list)]
                if sources[s]:
                    queue.append(sources[s].popleft())
                i += 1
        else:
            queue = []
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

        # Быстрая проверка — если ссылка уже была отправлена ранее, пропускаем
        if l in state.get("seen", {}):
            logging.debug(f"🔁 Пропускаю уже отправленную ссылку: {l}")
            continue

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
        try:
            active = (ACTIVE_MODEL or "").lower()
        except Exception:
            active = ""

        if "gemini" in active:
            logging.info(f"🧩 Используем GEMINI лимит {GEMINI_MAX_TOKENS} токенов для {ACTIVE_MODEL}")
            summary_text, used_model = await summarize(content, max_tokens=GEMINI_MAX_TOKENS)
        else:
            logging.info(f"🧩 Используем OLLAMA лимит {OLLAMA_MAX_TOKENS} токенов для {ACTIVE_MODEL}")
            # для локальной Ollama используем отдельную функцию; передаём усечённый контент
            summary_text, used_model = await summarize_ollama(content[:PARSER_MAX_TEXT_LENGTH])

        # --- Усечение заголовка и резюме ---
        MAX_SUMMARY_LEN = 1200
        MAX_TITLE_LEN = 120
        title_clean = t.strip()
        if len(title_clean) > MAX_TITLE_LEN:
            title_clean = title_clean[:MAX_TITLE_LEN].rsplit(" ", 1)[0] + "…"
        summary_clean = summary_text.strip()

        # --- списки → буллеты ---
        summary_clean = re.sub(r'(?m)^[\s]*[\*\-\u2013]\s+', '• ', summary_clean)
        summary_clean = re.sub(r'(?m)^[\s]*[\*\-]{2,}\s*$', '', summary_clean)
        summary_clean = re.sub(r'(?m)^[\s]*\*\s*', '• ', summary_clean)

        # --- базовое форматирование ---
        summary_clean = re.sub(r'(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', summary_clean)   # *текст* → курсив
        summary_clean = re.sub(r'(?<!\*)\*\*(?!\*)([^*\n]+?)(?<!\*)\*\*(?!\*)', r'<b>\1</b>', summary_clean) # **текст** → жирный
        summary_clean = re.sub(r'__([^_\n]+?)__', r'<u>\1</u>', summary_clean)                               # __текст__ → подчёркнутый
        summary_clean = re.sub(r'`([^`\n]+?)`', r'<code>\1</code>', summary_clean)                           # `код` → код
        summary_clean = re.sub(r'\[([^\]]+?)\]\((https?://[^\s)]+)\)', r'<a href="\2">\1</a>', summary_clean) # [текст](ссылка)

        # --- очистка ---
        summary_clean = re.sub(r'[ \t]{2,}', ' ', summary_clean).strip()

        # --- экранирование, кроме разрешённых HTML-тегов ---
        escaped = html.escape(summary_clean)

        # возвращаем поддерживаемые Telegram-теги
        for tag in ["b", "i", "u", "code", "a"]:
            escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>")
            escaped = escaped.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        # отдельно для href
        escaped = re.sub(r'&lt;a href=&quot;(https?://[^&]+)&quot;&gt;', r'<a href="\1">', escaped)

        # --- Формируем сообщение для Telegram ---
        title_safe = escape(title_clean)
        summary_safe = escaped
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

        header = HEADER_TEMPLATE.format(title=title_safe, source=s, date=local_time_str)
        body = f"{BODY_PREFIX}{summary_safe}"
        footer = FOOTER_TEMPLATE.format(model=used_model, link=link_safe)

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

        # (Проверка на seen_links была поднята выше чтобы избежать лишней работы)

        async def send_and_log(part_msg):
            import inspect
            logging.debug(part_msg[:800])  # не спамим полный текст в консоль
            fn = getattr(bot, "send_message", None)
            try:
                if inspect.iscoroutinefunction(fn):
                    await fn(chat_id=CHAT_ID, text=part_msg, parse_mode="HTML")
                else:
                    await asyncio.to_thread(fn, chat_id=CHAT_ID, text=part_msg, parse_mode="HTML")
            except Exception as e:
                if "429" in str(e):
                    logging.warning(f"⏳ Rate limited: {e}")
                    raise
                else:
                    logging.exception("❌ Ошибка при отправке сообщения")
                    raise

        for _ in range(3):
            try:
                for part_msg in assembled_parts:
                    await send_and_log(part_msg)
                    await asyncio.sleep(SINGLE_MESSAGE_PAUSE)

                ts_now = int(time.time())
                mark_state("sent", l, ts_now)
                mark_state("seen", l, ts_now)
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
                        p = parse_iso_utc(p)
                    except Exception:
                        pass

                iso_p = p.isoformat() if hasattr(p, "isoformat") else str(p)
                safe_queue.append((t, l, s, summary, iso_p))

            with open("news_queue.json", "w", encoding="utf-8") as f:
                json.dump(safe_queue, f, ensure_ascii=False, indent=2)

          
    # Сохраняем обновлённый state (sent + last_source_index)
    state.setdefault("sent", {})
    state["sent"].update(sent_links)
    if ROUND_ROBIN_MODE and 'src_list' in locals() and src_list:
        state.setdefault("meta", {})
        state["meta"]["last_source_index"] = (last_index + sent_count) % len(src_list)


    # Сохраняем список уже отправленных ссылок между перезапусками
    try:
        await save_state_async()
        cleanup_state()

    except Exception:
        logging.warning("⚠️ Не удалось сохранить state.json")

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
                try:
                    # ограничиваем выполнение send_news до INTERVAL секунд чтобы не блокировать цикл
                    await asyncio.wait_for(send_news(), timeout=INTERVAL)
                except asyncio.TimeoutError:
                    logging.warning("⏰ send_news превысил лимит времени и был прерван")
                except Exception as e:
                    logging.exception(f"❌ Ошибка в send_news: {e}")
                logging.info(f"⏰ Следующая проверка через {INTERVAL // 60} мин\n")
                print("💤 цикл завершён, жду следующий", flush=True)
                await asyncio.sleep(INTERVAL)
        except KeyboardInterrupt:
            logging.info("🛑 Завершение по Ctrl+C, сохраняем state…")
        finally:
            try:
                await save_state_async()
            except Exception as e:
                logging.warning(f"⚠️ Не удалось сохранить state.json при выходе: {e}")

if __name__ == "__main__":
    print("🚀 Запуск бота...", flush=True)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info(f"💬 MODEL_MAX_TOKENS = {MODEL_MAX_TOKENS}")
    logging.info(f"📰 PARSER_MAX_TEXT_LENGTH = {PARSER_MAX_TEXT_LENGTH}")
    asyncio.run(main())
