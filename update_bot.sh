#!/bin/bash

# --- НАСТРОЙКИ ---
# Базовая часть URL для удобства (чтобы не менять в 3 местах)
BASE_URL="https://raw.githubusercontent.com/Mytop244/parser-bot/refs/heads/main"

GITHUB_RAW_URL="${BASE_URL}/main.py"
GITHUB_REQ_URL="${BASE_URL}/requirements.txt"
GITHUB_RSS_URL="${BASE_URL}/rss.txt" # Ссылка на файл RSS

SCRIPT_NAME="main.py"
REQUIREMENTS="requirements.txt"
RSS_FILE="rss.txt"
BACKUP_DIR="backups"
PYTHON_CMD="python" # В Termux обычно просто python

# Переходим в директорию скрипта
cd "$(dirname "$0")"

# --- ФУНКЦИЯ ДЛЯ БЕЗОПАСНОГО ОБНОВЛЕНИЯ ---
update_file() {
    local url=$1
    local filename=$2
    local description=$3
    
    echo "⬇️ Скачивание $description ($filename)..."
    curl -s -L "$url" -o "${filename}.new"

    # Проверка: файл не пустой и не содержит HTML тега (ошибка 404)
    if [ -s "${filename}.new" ] && ! grep -q "<html" "${filename}.new"; then
        # Бэкап, если файл уже существует
        if [ -f "$filename" ]; then
            mkdir -p "$BACKUP_DIR"
            TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
            cp "$filename" "$BACKUP_DIR/${filename}_$TIMESTAMP.bak"
        fi
        
        mv "${filename}.new" "$filename"
        echo "✅ $description обновлен."
        return 0 # Успех
    else
        echo "⚠️ Не удалось обновить $description (ошибка загрузки или файл не изменился)."
        rm -f "${filename}.new"
        return 1 # Ошибка
    fi
}

# --- ЛОГИКА ---

echo "🔄 Начинаем обновление..."

# 1. Обновляем requirements.txt
update_file "$GITHUB_REQ_URL" "$REQUIREMENTS" "Файл зависимостей"

# 2. Обновляем rss.txt
update_file "$GITHUB_RSS_URL" "$RSS_FILE" "Список RSS"

# 3. Обновляем основной скрипт бота
if update_file "$GITHUB_RAW_URL" "$SCRIPT_NAME" "Скрипт бота"; then
    MAIN_UPDATED=true
else
    MAIN_UPDATED=false
fi

# 4. Установка зависимостей (специально для Termux и Linux)
if [ -f "$REQUIREMENTS" ]; then
    echo "📦 Проверка и установка библиотек из $REQUIREMENTS..."
    # Флаг --upgrade позволяет обновить библиотеки, если версии изменились
    pip install -r "$REQUIREMENTS" --upgrade
    echo "✅ Библиотеки проверены."
fi

# 5. Перезапуск
# Перезапускаем только если обновился main.py или если мы просто хотим перезагрузить процесс
echo "🛑 Остановка текущего бота..."
pkill -f "$PYTHON_CMD $SCRIPT_NAME"

sleep 2

echo "🚀 Запуск бота..."
nohup $PYTHON_CMD "$SCRIPT_NAME" > bot_output.log 2>&1 &

# Сохраняем PID
echo $! > bot.pid

echo "✅ Все готово! Бот запущен (PID: $(cat bot.pid))."
echo "📝 Следить за логами: tail -f bot_output.log"