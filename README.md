### 📘 README.md

````markdown
# 📰 Telegram AI News Bot

**Telegram AI News Bot** — это асинхронный бот, который:
- парсит RSS-ленты,
- извлекает тексты новостей,
- делает их краткое резюме при помощи **AI (Gemini / Ollama)**,
- и публикует результаты в Telegram-канал или чат.

В комплекте идёт вспомогательный скрипт `rss_checker.py` — для проверки живости RSS-источников и автоматического комментирования нерабочих ссылок в `rss.txt`.

---

## 🚀 Возможности

- 📡 Поддержка любых RSS-лент (в `rss.txt`)
- 🧠 Резюмирование новостей с помощью **Google Gemini API** или **Ollama LLM**
- 🤖 Автоматическая отправка в Telegram через **Bot API**
- ⚙️ Асинхронная работа (`asyncio` + `aiohttp`)
- 🧩 Проверка RSS-источников (`rss_checker.py`)
- 💾 Сохранение состояния, чтобы избежать дубликатов
- 🧱 Простая настройка через `.env`

---

## 🛠 Установка

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/yourname/telegram-ai-news-bot.git
cd telegram-ai-news-bot
````

### 2. Создайте виртуальное окружение и установите зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Настройте окружение

Создайте файл `.env` и укажите ключевые переменные:

```env
# Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCDEF...
TELEGRAM_CHAT_ID=-1001234567890

# AI (один из вариантов)
# Для Gemini
GEMINI_API_KEY=your_gemini_api_key

# Для Ollama
OLLAMA_MODEL=llama3
OLLAMA_API_BASE=http://localhost:11434

# Прочее
UPDATE_INTERVAL=300        # Период обновления RSS (сек)
SUMMARY_SENTENCES=3        # Кол-во предложений в саммари
```

---

## 📄 Файл `rss.txt`

Добавьте в `rss.txt` список RSS-источников — по одному в строке.

Пример:

```
https://meduza.io/rss/all
https://habr.com/ru/rss/all/all/
https://lenta.ru/rss/news
```

---

## 🧪 Проверка источников

Перед запуском бота рекомендуется проверить ленты:

```bash
python rss_checker.py
```

Скрипт проверит все ссылки из `rss.txt` и автоматически закомментирует нерабочие (`# ...`) в том же файле.

---

## ▶️ Запуск бота

После настройки `.env` и `rss.txt` запустите:

```bash
python main.py
```

Бот начнёт:

* читать RSS-ленты,
* формировать краткие AI-резюме,
* и отправлять их в указанный Telegram-чат.

---

## 🧰 Структура проекта

```
.
├── main.py              # Основная логика Telegram AI бота
├── rss_checker.py       # Проверка и фильтрация RSS источников
├── rss.txt              # Список RSS-лент
├── requirements.txt     # Зависимости Python
├── .env.example         # Пример настроек окружения
└── README.md            # Этот файл 🙂
```

---

## ⚙️ Зависимости

* Python ≥ 3.10
* `aiohttp`, `feedparser`, `python-dotenv`, `requests`, `asyncio`, `aiogram`
* Для AI:

  * **Gemini API** (`google-generativeai`)
  * или **Ollama API** (локальная модель)

---

## 🧩 Пример вывода

```
✅ https://meduza.io/rss/all
❌ https://old-dead-feed.com/rss
```

В Telegram:

```
🗞 Новость: "Apple представила новые MacBook Pro"
🧠 Саммари: Компания представила обновлённую линейку ноутбуков с процессорами M4 и новым дизайном...
```

---

## 🐳 Docker (опционально)

Можно запустить через Docker:

```bash
docker build -t telegram-ai-news-bot .
docker run --env-file .env telegram-ai-news-bot
```

---

## 🧠 Автор

**Winters89**
📧 Winters89@mail.ru
💬 Telegram: @dvprokhorov

---

## 📜 Лицензия

MIT License © 2025
Свободно используйте, модифицируйте и распространяйте с указанием автора.

```

