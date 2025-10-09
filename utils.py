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
