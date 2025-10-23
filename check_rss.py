import feedparser, aiohttp, asyncio, logging, os

RSS_FILE = "rss.txt"

async def check_feed(session, url):
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return False
            text = await resp.text()
            feed = feedparser.parse(text)
            return bool(feed.entries)
    except Exception:
        return False

async def main():
    if not os.path.exists(RSS_FILE):
        print("Файл rss.txt не найден.")
        return

    with open(RSS_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]

    async with aiohttp.ClientSession() as session:
        new_lines = []
        for line in lines:
            if not line or line.startswith("#"):
                new_lines.append(line)
                continue
            ok = await check_feed(session, line)
            mark = "✅" if ok else "❌"
            print(f"{mark} {line}")
            new_lines.append(line if ok else f"# {line}")

    with open(RSS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
