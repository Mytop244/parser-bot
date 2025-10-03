# Telegram RSS Parser Bot

Бот для отправки новостей из RSS в Telegram с поддержкой логирования и сохранением уже отправленных ссылок.

---

## ⚙️ Возможности

* Парсинг RSS-ленты (по умолчанию Wired RSS)
* Отправка новостей в Telegram
* Локальное хранение отправленных ссылок (`sent_links.json`)
* Логи с ежедневной ротацией (`log/parser-YYYY-MM-DD.log`)
* Игнорирование секретов и временных файлов через `.gitignore`
* Настраиваемый интервал проверки новостей

---

## 📦 Установка

1. Клонируем репозиторий:

```bash
git clone <URL_репозитория>
cd parser-bot
```

2. Создаем и активируем виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows
```

3. Устанавливаем зависимости:

```bash
pip install -r requirements.txt
```

4. Создаем файл `.env` с настройками:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_chat_id
RSS_URL=https://www.wired.com/feed/rss
NEWS_LIMIT=5
INTERVAL=600
SENT_LINKS_FILE=sent_links.json
```

---

## 🚀 Запуск

```bash
source .venv/bin/activate
python main.py
```

Бот будет проверять новости каждые `INTERVAL` секунд и отправлять новые в Telegram.

---

## 📂 Структура проекта

```
parser-bot/
├─ main.py               # основной скрипт бота
├─ requirements.txt      # зависимости Python
├─ .env                  # токены и настройки (не коммитится)
├─ sent_links.json       # локальные отправленные ссылки (не коммитится)
├─ log/                  # лог-файлы с ротацией по дням (не коммитится)
├─ README.md
```

---

## 📌 Git Ignore

Файлы и папки, которые не отслеживаются:

```
.env
sent_links.json
log/
*.log
venv/
.venv/
__pycache__/
```

---

## ⚠️ Примечания

* Логи хранятся максимум 7 дней.
* RSS URL можно поменять в `.env`.
* Бот использует асинхронный `aiohttp` и Telegram Bot API.
