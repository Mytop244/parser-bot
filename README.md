# 🧠 AI News Telegram Bot

Телеграм-бот, который собирает новости из RSS-источников, анализирует статьи с помощью **Gemini API** или **локальной Ollama**, формирует краткие резюме и отправляет их в Telegram.

---

## ⚙️ Возможности
- Поддержка **нескольких RSS-источников**
- Краткое резюме с помощью:
  - 🌐 **Google Gemini (AI Studio)**
  - 💻 **Ollama (локальная LLM)** — fallback при недоступности Gemini
- Фильтр по дате и количеству новостей
- Очередь и round-robin по источникам
- Логирование и сохранение отправленных ссылок
- Полностью управляется через `.env`

---

## 📁 Структура проекта

```

project/
│
├─ main.py               # Основная логика бота
├─ article_parser.py     # Извлечение текста статьи
├─ .env                  # Настройки окружения
├─ sent_links.json       # История отправленных ссылок
├─ news_queue.json       # Очередь новостей
└─ README.md

````

---

## 🧩 Установка

1. Установи зависимости:
   ```bash
   pip install -r requirements.txt
````

2. Создай файл `.env`:

   ```env
   # 🌍 Время
   TIMEZONE=Europe/Moscow

   # 🤖 Telegram
   TELEGRAM_TOKEN=Твой_токен
   CHAT_ID=Твой_CHAT_ID

   # 📰 RSS источники (через запятую)
   RSS_URLS=https://www.wired.com/feed/tag/ai/latest/rss,https://magazine.sebastianraschka.com/feed

   # ⚙️ Настройки
   INTERVAL=600
   NEWS_LIMIT=5
   DAYS_LIMIT=1
   ROUND_ROBIN_MODE=1

   # 🧠 AI
   AI_STUDIO_KEY=твой_Gemini_API_ключ
   AI_MODEL=gemini-2.5-flash
   OLLAMA_MODEL=gpt-oss:20b
   OLLAMA_MODEL_FALLBACK=gpt-oss:120b

   # 💤 Паузы и батчи
   BATCH_SIZE_SMALL=5
   PAUSE_SMALL=3
   BATCH_SIZE_MEDIUM=15
   PAUSE_MEDIUM=5
   BATCH_SIZE_LARGE=25
   PAUSE_LARGE=10
   SINGLE_MESSAGE_PAUSE=1

   # 💾 Файлы
   SENT_LINKS_FILE=sent_links.json
   ```

---

## 🚀 Запуск

```bash
python main.py
```

При запуске бот:

1. Проверяет RSS-источники.
2. Извлекает новые статьи.
3. Делает краткое резюме с помощью Gemini или Ollama.
4. Отправляет сообщение в Telegram.

---

## 🧠 Пример сообщения

```
The Big LLM Architecture Comparison  
📡 Ahead of AI | 🗓 19.07.2025, 11:11  
━━━━━━━━━━━━━━━  
💬 Несмотря на семилетний прогресс с момента создания архитектуры GPT, современные модели сохраняют сходство с GPT-2.  
━━━━━━━━━━━━━━━  
🤖 Модель: gemini-2.5-flash  
🔗 Читать статью
```

---

## 🔍 Логирование

Все логи сохраняются в `parser.log` и выводятся в консоль:

```
INFO | 🔄 Проверка новостей...
INFO | ✅ Gemini OK (gemini-2.5-flash)
INFO | 📤 Отправлено: The Big LLM Architecture Comparison...
```

---

## 💡 Совет

* Если нет ключа Gemini, бот автоматически перейдёт на Ollama.
* Для стабильности можно запустить через **pm2** или **systemd**:

  ```bash
  pm2 start main.py --name ai-news-bot --interpreter python
  ```

---

## 📜 Лицензия

MIT License © 2025
