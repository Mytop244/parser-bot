Sure — here’s a clean, professional English `README.md` you can put on GitHub for your project 👇

---

````markdown
# 📰 Async Telegram RSS News Bot

A powerful **asynchronous Python bot** that fetches, summarizes, and delivers news from multiple **RSS feeds** directly to a Telegram channel or chat.  
It supports both **Google Gemini** and **Ollama local models** for AI-based news summarization, along with advanced error handling, caching, and state persistence.

---

## ✨ Features

- 📡 Fetches and parses multiple RSS feeds concurrently using `aiohttp` and `feedparser`
- 🧠 AI-based summarization:
  - **Google Gemini API** (`gemini-2.0-flash` by default)
  - **Ollama local models** (e.g. `gpt-oss:20b`, `gpt-oss:120b`)
- 🤖 Sends formatted messages directly to **Telegram**
- 🕓 Filters recent news (by `DAYS_LIMIT`)
- 🪶 Automatically extracts full article text using:
  - `<article>` and `<p>` tags
  - `trafilatura` or `readability` fallback
- 🔄 Smart state persistence and migration (`state.json`)
- 🧩 Supports adaptive pause and batching to avoid Telegram rate limits
- 🧰 Configurable through `.env` and `rss.txt`

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rss-telegram-bot.git
cd rss-telegram-bot
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

Example:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=123456789
RSS_URLS=https://feeds.bbci.co.uk/news/rss.xml,https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml
AI_STUDIO_KEY=your_google_api_key
AI_MODEL=gemini-2.5-flash
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_MODEL_FALLBACK=gpt-oss:120b
INTERVAL=600
DAYS_LIMIT=1
NEWS_LIMIT=5
```

### 4. Optional: add `rss.txt`

If present, this file overrides `RSS_URLS`:

```
https://feeds.bbci.co.uk/news/rss.xml
https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml
# comments are ignored
```

---

## 🚀 Run the Bot

```bash
python main.py
```

The bot will:

1. Check all RSS feeds
2. Parse and summarize recent news
3. Send formatted updates to your Telegram channel

---

## 🧠 AI Models

| Model Type | Source                  | Description                             |
| ---------- | ----------------------- | --------------------------------------- |
| Gemini     | Google AI Studio        | Fast cloud summarization                |
| Ollama     | Local (localhost:11434) | Offline summarization with local models |

You can switch models dynamically using `.env`.

---

## 🪪 Logging

All activity is logged to:

```
parser.log
```

Example:

```
2025-10-27 08:40:53 | INFO | 🔍 Checking source: https://www.bbc.co.uk/news/rss.xml
2025-10-27 08:40:58 | INFO | ✅ Gemini OK (gemini-2.5-flash)
```

---

## 🧩 Dependencies

* Python 3.10+
* `aiohttp`, `feedparser`, `python-dotenv`, `beautifulsoup4`
* `python-telegram-bot`
* Optional: `trafilatura`, `readability-lxml`

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📦 File Structure

```
.
├── main.py              # Main async logic
├── rss.txt              # Optional list of RSS feeds
├── state.json           # Persistent cache (auto-generated)
├── parser.log           # Log output
├── .env                 # Configuration
└── requirements.txt
```

---

## 📜 License

MIT License © 2025 Mytop244

---

## 💡 Tips

* Use `SMART_PAUSE=1` to enable adaptive pauses when models fail
* Run locally with Ollama for privacy and offline operation
* Add `ROUND_ROBIN_MODE=1` for fair feed cycling

---

### 🧭 Example Output

**Telegram message format:**

```
<b>Breaking News Title</b>
📡 <i>BBC News</i> | 🗓 2025-10-27
━━━━━━━━━━━━━━━
💬 AI-generated summary text...

━━━━━━━━━━━━━━━
🤖 <i>Model: gemini-2.0-flash</i>
🔗 <a href="https://bbc.co.uk/news/article">Read full article</a>
