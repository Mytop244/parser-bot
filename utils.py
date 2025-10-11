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
import re
from html import escape
from telegram import Bot

HTML_SAFE_LIMIT = 4096  # Telegram limit

async def send_long_message(bot: Bot, chat_id: int, text: str, parse_mode="HTML", delay: int = 1):
    """
    Разбивает длинный текст на безопасные части для Telegram и отправляет их.
    Сохраняет HTML-разметку и ссылки.
    """
    # Разбиваем по абзацам, чтобы минимизировать разрыв тегов
    paragraphs = text.split("\n")
    parts = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Если добавление нового абзаца не превышает лимит — добавляем
        if len(current) + len(para) + 1 < HTML_SAFE_LIMIT:
            current += ("" if not current else "\n") + para
        else:
            # Если текущая часть уже есть — добавляем в список
            if current:
                parts.append(current)
            # Если абзац слишком длинный — делим его по предложениям или пробелам
            if len(para) >= HTML_SAFE_LIMIT:
                sub_parts = split_text_safe(para, HTML_SAFE_LIMIT)
                parts.extend(sub_parts)
                current = ""
            else:
                current = para

    if current:
        parts.append(current)

    # Отправляем по частям
    for part in parts:
        await bot.send_message(chat_id=chat_id, text=part, parse_mode=parse_mode)
        await asyncio.sleep(delay)


def split_text_safe(text: str, limit: int) -> list[str]:
    """
    Безопасно разбивает длинный текст на части ≤ limit, не ломая слова.
    """
    parts = []
    while len(text) > limit:
        # Ищем последний перенос строки или пробел в пределах лимита
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
