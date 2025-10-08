# 📰 RSS News Bot

Асинхронный бот для парсинга RSS-лент и отправки новостей в Telegram с использованием AI-суммаризации.

## 🚀 Особенности

- **📡 Мульти-источники** - поддержка неограниченного количества RSS-лент
- **🤖 AI-суммаризация** - автоматическое создание кратких содержаний через Gemini API или локальный Ollama
- **⚡ Асинхронная обработка** - высокая производительность с aiohttp
- **🔄 Round-Robin режим** - равномерное распределение новостей по источникам
- **💾 Умный кеш** - предотвращение дублирования отправленных новостей
- **📊 Батчевая отправка** - адаптивная пауза между сообщениями
- **⏰ Фильтрация по времени** - настраиваемый лимит дней для новостей

## 🛠 Технологии

- **Python 3.7+** с asyncio
- **AI-модели**: Google Gemini API / Ollama (локально)
- **Парсинг**: feedparser, BeautifulSoup4
- **HTTP-клиент**: aiohttp
- **Telegram Bot API**
- **Обработка дат**: datetime, timezone

## 📦 Установка

### 1. Клонирование и настройка
```bash
git clone <your-repo-url>
cd rss-news-bot
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Настройка окружения
Создайте файл `.env`:
```env
# Обязательные настройки
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_chat_id

# RSS-ленты (через запятую)
RSS_URLS=https://example1.com/rss,https://example2.com/feed

# Настройки парсера
INTERVAL=600
NEWS_LIMIT=5
DAYS_LIMIT=1
ROUND_ROBIN_MODE=1

# AI-суммаризация (опционально)
AI_STUDIO_KEY=your_gemini_api_key
AI_MODEL=gemini-2.5-flash

# Локальная суммаризация через Ollama
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_MODEL_FALLBACK=gpt-oss:120b

# Настройки батчей
BATCH_SIZE_SMALL=5
PAUSE_SMALL=3
BATCH_SIZE_MEDIUM=15
PAUSE_MEDIUM=5
BATCH_SIZE_LARGE=25
PAUSE_LARGE=10
SINGLE_MESSAGE_PAUSE=1

# Файлы
SENT_LINKS_FILE=sent_links.json
TIMEZONE=Europe/Moscow
```

## 🚀 Запуск

### Базовый запуск:
```bash
python bot.py
```

### С Docker:
```dockerfile
FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "bot.py"]
```

## ⚙️ Конфигурация

### Основные параметры:
- `TELEGRAM_TOKEN` - токен бота от @BotFather
- `CHAT_ID` - ID чата для отправки
- `RSS_URLS` - список RSS-лент через запятую
- `INTERVAL` - интервал проверки в секундах (по умолчанию 600)
- `NEWS_LIMIT` - лимит новостей за одну итерацию
- `DAYS_LIMIT` - максимальный возраст новостей в днях

### Режимы работы:
- `ROUND_ROBIN_MODE=1` - чередование источников
- `ROUND_ROBIN_MODE=0` - хронологический порядок

### AI-суммаризация:
- **Gemini API**: установите `AI_STUDIO_KEY` для использования Google Gemini
- **Ollama локально**: запустите Ollama сервер на `127.0.0.1:11434`

## 📝 Формат сообщений

Бот отправляет новости в формате:
```
━━━━━━━━━━━━━━━
📰 Заголовок новости
📡 Источник | 🗓 25.12.2024, 14:30
━━━━━━━━━━━━━━━

💬 Краткое содержание новости, сгенерированное AI...

🤖 Модель: gemini-2.5-flash
🔗 Читать статью
```

## 🔧 Файлы данных

- `sent_links.json` - история отправленных ссылок
- `news_queue.json` - очередь неотправленных новостей  
- `parser.log` - лог работы бота

## 🐛 Логирование

Бот ведет подробное логирование:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("parser.log", encoding="utf-8")]
)
```

## 🤝 Разработка

### Добавление новых источников:
Просто добавьте RSS-URL в переменную `RSS_URLS` в `.env` файле.

### Кастомизация суммаризации:
Измените промпт в функциях `summarize()` или `summarize_ollama()`.

### Расширение функционала:
Основные точки расширения:
- `fetch_and_check()` - парсинг RSS
- `summarize()` - AI-суммаризация  
- `send_news()` - логика отправки

## ⚠️ Примечания

- Для работы с Gemini API необходим аккаунт в Google AI Studio
- Локальная суммаризация требует установленного Ollama
- Бот корректно обрабатывает временные зоны и форматы дат
- Реализована обработка ошибок и повторные попытки при сетевых сбоях