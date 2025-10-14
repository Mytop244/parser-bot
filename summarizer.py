import time, json, logging, asyncio, aiohttp, random, atexit
from datetime import timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# ---------------- ПАРАМЕТРЫ ----------------
try:
    from main import (
        PARSER_MAX_TEXT_LENGTH,
        OLLAMA_TIMEOUT,
        MODEL_MAX_TOKENS,
        OLLAMA_MODEL,
        OLLAMA_MODEL_FALLBACK,
        AI_STUDIO_KEY,
        GEMINI_MODEL,
    )
except Exception:
    PARSER_MAX_TEXT_LENGTH = 10000
    OLLAMA_TIMEOUT = 180
    MODEL_MAX_TOKENS = 1200
    OLLAMA_MODEL = "gpt-oss:20b"
    OLLAMA_MODEL_FALLBACK = "gpt-oss:120b"
    AI_STUDIO_KEY = None
    GEMINI_MODEL = "gemini-2.5-flash"


# ---------------- СЕССИЯ (улучшение 1) ----------------
_AIO_CONN = aiohttp.TCPConnector(limit=10)
_session = aiohttp.ClientSession(connector=_AIO_CONN)

def get_session():
    return _session

@atexit.register
def _close_session():
    try:
        loop = asyncio.get_event_loop()
        if not _session.closed:
            loop.run_until_complete(_session.close())
    except Exception:
        pass


# ---------------- ОГРАНИЧЕНИЕ (улучшение 2) ----------------
_semaphore = asyncio.Semaphore(3)

# ---------------- КЭШ OLLAMA (улучшение 3) ----------------
_cache = {}


# ---------------- RETRY ----------------
async def async_retry(func, attempts=3, base=1.0, max_delay=30.0):
    delay = base
    for i in range(attempts):
        try:
            return await func()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if i == attempts - 1:
                raise
            jitter = random.uniform(0, delay * 0.5)
            logging.warning(f"⚠️ Повтор через {round(delay + jitter, 1)}с: {e}")
            await asyncio.sleep(min(delay + jitter, max_delay))
            delay *= 2


# ---------------- OLLAMA ----------------
async def summarize_ollama(text: str):
    if text in _cache:
        logging.info("♻️ Ollama cache hit")
        return _cache[text], "cache"

    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]
    prompt = f"Не делай вступлений. Сделай резюме новости на русском языке:\n{prompt_text}"
    logging.info(f"🧠 [OLLAMA INPUT] >>> {prompt_text[:5500]}")

    async def run_model(model_name: str):
        async with _semaphore:
            url = "http://127.0.0.1:11434/api/generate"
            payload = {"model": model_name, "prompt": prompt, "options": {"num_predict": MODEL_MAX_TOKENS}}
            start_time = time.time()
            session = get_session()
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT)) as resp:
                    if resp.status != 200:
                        logging.error(f"⚠️ Ollama {model_name} HTTP {resp.status}")
                        return None, model_name

                    text_out, buffer, total = "", "", 0
                    MAX_STREAM_CHARS = PARSER_MAX_TEXT_LENGTH * 2
                    async for raw in resp.content:
                        if not raw:
                            continue
                        try:
                            chunk = raw.decode("utf-8")
                        except Exception:
                            continue
                        buffer += chunk
                        parts = buffer.splitlines()
                        if buffer and not buffer.endswith("\n"):
                            *lines, buffer = parts
                        else:
                            lines, buffer = parts, ""
                        for line in lines:
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            frag = data.get("response", "")
                            text_out += frag
                            total += len(frag)
                            if total > MAX_STREAM_CHARS:
                                logging.warning("⚠️ Ollama stream truncated (limit reached)")
                                raise asyncio.CancelledError("stream too long")

                    out = text_out.strip()
                    if not out:
                        logging.warning(f"⚠️ Ollama ({model_name}) пустой ответ")
                        return None, model_name
                    elapsed = round(time.time() - start_time, 2)
                    logging.info(f"✅ Ollama ({model_name}) за {elapsed} сек")
                    _cache[text] = out
                    return out, model_name

            except asyncio.TimeoutError:
                logging.error(f"⏰ Ollama ({model_name}) таймаут")
            except asyncio.CancelledError:
                logging.warning(f"⚠️ Ollama ({model_name}) остановлен по лимиту")
            except Exception as e:
                logging.error(f"❌ Ollama ({model_name}): {e}")
            return None, model_name

    try:
        result, used_model = await async_retry(lambda: run_model(OLLAMA_MODEL), attempts=3, base=1.5)
    except Exception:
        logging.warning(f"⚠️ Переключаюсь на резервную модель {OLLAMA_MODEL_FALLBACK}")
        result, used_model = await async_retry(lambda: run_model(OLLAMA_MODEL_FALLBACK), attempts=3, base=1.5)

    if not result:
        return prompt_text[:2000] + "...", "local-fallback"
    return result, used_model


# ---------------- GEMINI ----------------
async def summarize(text, max_tokens=200, retries=3):
    text = BeautifulSoup(text, "html.parser").get_text() if text and "<" in text else (text or "")
    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]
    prompt_text = f"Сделай профессиональное краткое резюме новости на русском языке, без вступления, дели на абзацы:\n{prompt_text}"

    if not AI_STUDIO_KEY:
        logging.debug(f"🧠 [GEMINI INPUT] {prompt_text[:500]}...")
        logging.warning("⚠️ AI_STUDIO_KEY не задан, fallback на Ollama")
        return await summarize_ollama(text)

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens or MODEL_MAX_TOKENS},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {"x-goog-api-key": AI_STUDIO_KEY, "Content-Type": "application/json"}

    async def call_gemini():
        async with _semaphore:
            session = get_session()
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.text()
                if resp.status == 429:
                    logging.warning("⚠️ Gemini quota exceeded — fallback to Ollama")
                    return None
                if resp.status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=body,
                    )
                result = json.loads(body)
                candidates = result.get("candidates")
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]

    try:
        text_out = await async_retry(call_gemini, attempts=retries, base=2.0)
        if text_out:
            logging.info(f"✅ Gemini OK ({GEMINI_MODEL}): {text_out}")
            return text_out.strip(), GEMINI_MODEL
    except Exception as e:
        logging.warning(f"⚠️ Gemini error: {e}")

    logging.error("❌ Gemini не ответил, fallback на Ollama")
    return await summarize_ollama(text)
