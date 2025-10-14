import time
import json
import logging
import asyncio
import aiohttp
from datetime import timezone

from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

# these names will be provided by main when imported (or can be read from env here)
try:
    from main import PARSER_MAX_TEXT_LENGTH, OLLAMA_TIMEOUT, MODEL_MAX_TOKENS, OLLAMA_MODEL, OLLAMA_MODEL_FALLBACK, AI_STUDIO_KEY, GEMINI_MODEL
except Exception:
    PARSER_MAX_TEXT_LENGTH = int(10000)
    OLLAMA_TIMEOUT = 180
    MODEL_MAX_TOKENS = 1200
    OLLAMA_MODEL = "gpt-oss:20b"
    OLLAMA_MODEL_FALLBACK = "gpt-oss:120b"
    AI_STUDIO_KEY = None
    GEMINI_MODEL = "gemini-2.5-flash"


async def summarize_ollama(text: str):
    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]
    prompt = f"Не делай вступлений. Сделай резюме новости на русском языке:\n{prompt_text}"
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

                    text_out = ""
                    try:
                        async for chunk in resp.content:
                            if not chunk:
                                continue
                            try:
                                s = chunk.decode("utf-8")
                            except Exception:
                                continue
                            for line in s.splitlines():
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line)
                                except Exception:
                                    continue
                                text_out += data.get("response", "")
                    except Exception as e:
                        logging.error(f"❌ Ollama ({model_name}) stream error: {e}")
                        return None, model_name

                    out = text_out.strip()
                    if not out:
                        logging.warning(f"⚠️ Ollama ({model_name}) пустой ответ")
                        return None, model_name
                    elapsed = round(time.time() - start_time, 2)
                    logging.info(f"✅ Ollama ({model_name}) за {elapsed} сек")
                    logging.info(f"🧠 [OLLAMA OUTPUT] <<< {out}")
                    return out, model_name
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
        return prompt_text[:2000] + "...", "local-fallback"

    return result, used_model


async def summarize(text, max_tokens=200, retries=3):
    text = BeautifulSoup(text, "html.parser").get_text() if text and "<" in text else (text or "")
    prompt_text = text[:PARSER_MAX_TEXT_LENGTH]
    prompt_text = f"Сделай профессиональное краткое резюме новости на русском языке, без вступления, дели на абзацы:\n{prompt_text}"

    if not AI_STUDIO_KEY:
        logging.debug(f"🧠 [GEMINI INPUT] {prompt_text[:500]}...")
        logging.warning("⚠️ AI_STUDIO_KEY не задан, fallback на Ollama")
        return await summarize_ollama(text)

    payload = {"contents": [{"parts": [{"text": prompt_text}]}],
               "generationConfig": {"maxOutputTokens": max_tokens or MODEL_MAX_TOKENS}}
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
