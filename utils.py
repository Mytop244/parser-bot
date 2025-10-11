# utils.py
import logging, re

# Настройка логов (если не переопределена в main)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def clean_text(text: str) -> str:
    """Убирает лишние пробелы, переносы, невидимые символы."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def short_preview(text: str, limit: int = 120) -> str:
    """Короткий превью для логов/отладки."""
    if not text:
        return ""
    return (text[:limit] + "…") if len(text) > limit else text

def safe_get(d: dict, *keys, default=None):
    """Безопасно извлекает вложенные значения из словаря."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


import asyncio
from html import escape

async def send_long_message(bot, chat_id: int, text: str, parse_mode="HTML", delay: float = 0.5):
    """
    Отправляет длинный текст в Telegram, разбивая на части ≤ 4096 символов.
    - bot: экземпляр telegram.Bot
    - chat_id: id чата
    - text: текст для отправки
    - parse_mode: режим (HTML или Markdown)
    - delay: пауза между частями
    """
    MAX_LEN = 4096

    # Экранируем текст если parse_mode=HTML
    if parse_mode and parse_mode.upper() == "HTML":
        text = escape(text)

    parts = []
    while len(text) > MAX_LEN:
        split_pos = text.rfind('\n', 0, MAX_LEN)
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, MAX_LEN)
        if split_pos == -1:
            split_pos = MAX_LEN
        parts.append(text[:split_pos].strip())
        text = text[split_pos:].strip()
    if text:
        parts.append(text)

    for part in parts:
        # Поддержка как async send_message, так и sync через to_thread
        if asyncio.iscoroutinefunction(bot.send_message):
            await bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
        else:
            await asyncio.to_thread(lambda p=part: bot.send_message(chat_id=chat_id, text=p, parse_mode=parse_mode))
        await asyncio.sleep(delay)
