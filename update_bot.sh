#!/bin/bash

# --- НАСТРОЙКИ ---
BASE_URL="https://raw.githubusercontent.com/Mytop244/parser-bot/refs/heads/main"
GITHUB_RAW_URL="${BASE_URL}/main.py"
GITHUB_REQ_URL="${BASE_URL}/requirements.txt"
GITHUB_RSS_URL="${BASE_URL}/rss.txt"

SCRIPT_NAME="main.py"
REQUIREMENTS="requirements.txt"
RSS_FILE="rss.txt"
BACKUP_DIR="backups"
LOGS_DIR="logs"
PYTHON_CMD="python"

cd "$(dirname "$0")"

# 1. ЗАЩИТА ОТ УСЫПЛЕНИЯ (TERMUX)
if command -v termux-wake-lock > /dev/null; then
    termux-wake-lock
    echo "🔋 Termux Wake Lock активирован."
fi

# 2. ПРОВЕРКА ИНТЕРНЕТА
echo "🌐 Проверка соединения..."
if ! ping -c 1 google.com &> /dev/null; then
    echo "❌ Нет интернета! Обновление отменено."
    exit 1
fi

# --- ФУНКЦИЯ ЗАГРУЗКИ ---
update_file() {
    local url=$1
    local filename=$2
    local description=$3
    
    echo "⬇️ Скачивание $description..."
    curl -s -L "$url" -o "${filename}.new"

    if [ -s "${filename}.new" ] && ! grep -q "<html" "${filename}.new"; then
        # Если это python-файл, проверяем синтаксис ПЕРЕД заменой
        if [[ "$filename" == *.py ]]; then
            if ! $PYTHON_CMD -m py_compile "${filename}.new"; then
                echo "❌ ОШИБКА: В новом файле $filename синтаксическая ошибка! Отмена."
                rm -f "${filename}.new"
                return 1
            fi
            echo "🧠 Синтаксис $filename в порядке."
        fi

        # Бэкап
        if [ -f "$filename" ]; then
            mkdir -p "$BACKUP_DIR"
            cp "$filename" "$BACKUP_DIR/${filename}_$(date +"%Y%m%d_%H%M%S").bak"
        fi
        
        mv "${filename}.new" "$filename"
        echo "✅ $description обновлен."
        return 0
    else
        echo "⚠️ Ошибка загрузки $description. Пропуск."
        rm -f "${filename}.new"
        return 1
    fi
}

# --- ОБНОВЛЕНИЕ ---

update_file "$GITHUB_RSS_URL" "$RSS_FILE" "RSS список"
update_file "$GITHUB_REQ_URL" "$REQUIREMENTS" "Requirements"

# Проверяем, обновился ли main.py
MAIN_UPDATED=false
if update_file "$GITHUB_RAW_URL" "$SCRIPT_NAME" "Скрипт бота"; then
    MAIN_UPDATED=true
fi

# --- УСТАНОВКА ЗАВИСИМОСТЕЙ ---
if [ -f "$REQUIREMENTS" ]; then
    echo "📦 Синхронизация библиотек..."
    pip install -r "$REQUIREMENTS" --upgrade --prefer-binary > /dev/null
fi

# --- ПЕРЕЗАПУСК (РОТАЦИЯ ЛОГОВ) ---

# Перезапускаем, если обновился скрипт ИЛИ процесс бота мертв
PID=$(cat bot.pid 2>/dev/null)
IS_RUNNING=false
if [ -n "$PID" ] && ps -p "$PID" > /dev/null; then
    IS_RUNNING=true
fi

if [ "$MAIN_UPDATED" = true ] || [ "$IS_RUNNING" = false ]; then
    echo "🔄 Перезапуск бота..."
    
    if [ "$IS_RUNNING" = true ]; then
        kill "$PID"
        sleep 2
        # Жесткое убийство, если не умер
        pkill -f "$PYTHON_CMD $SCRIPT_NAME"
    fi

    # Ротация логов
    mkdir -p "$LOGS_DIR"
    if [ -f "bot_output.log" ]; then
        mv "bot_output.log" "$LOGS_DIR/log_$(date +"%Y%m%d_%H%M%S").txt"
    fi

    # Запуск
    nohup $PYTHON_CMD "$SCRIPT_NAME" > bot_output.log 2>&1 &
    echo $! > bot.pid
    
    echo "✅ Бот успешно перезапущен (PID: $(cat bot.pid))"
else
    echo "💤 Обновлений кода не было, бот работает. Ничего не делаем."
fi