#!/bin/bash

# --- НАСТРОЙКИ ---
GITHUB_RAW_URL="https://raw.githubusercontent.com/Mytop244/parser-bot/refs/heads/main/main.py"
GITHUB_REQ_URL="https://raw.githubusercontent.com/Mytop244/parser-bot/refs/heads/main/requirements.txt" # Ссылка на requirements

SCRIPT_NAME="main.py"
REQUIREMENTS="requirements.txt"
BACKUP_DIR="backups"
PYTHON_CMD="python" # Или python3, в зависимости от системы

# Переходим в директорию, где лежит этот скрипт, чтобы пути не сломались
cd "$(dirname "$0")"

# --- ЛОГИКА ---

echo "🔄 Проверка обновлений..."

# 1. Скачиваем новый файл
curl -s -L "$GITHUB_RAW_URL" -o "${SCRIPT_NAME}.new"

# 2. Проверяем валидность (размер > 0 и отсутствие HTML тегов ошибки 404)
# grep ищет "<html", чтобы убедиться, что GitHub не вернул страницу ошибки вместо кода
if [ -s "${SCRIPT_NAME}.new" ] && ! grep -q "<html" "${SCRIPT_NAME}.new"; then
    echo "✅ Загрузка завершена."

    mkdir -p "$BACKUP_DIR"

    # 3. Бэкап
    if [ -f "$SCRIPT_NAME" ]; then
        TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
        cp "$SCRIPT_NAME" "$BACKUP_DIR/${SCRIPT_NAME}_$TIMESTAMP.bak"
        echo "📦 Бэкап сохранен: $BACKUP_DIR/${SCRIPT_NAME}_$TIMESTAMP.bak"
    fi

    # 4. Замена файла
    mv "${SCRIPT_NAME}.new" "$SCRIPT_NAME"
    echo "📄 Основной файл обновлен."

    # 4.1 Проверка зависимостей (раскомментируйте, если нужно авто-обновление библиотек)
    # echo "📦 Проверка requirements.txt..."
    # curl -s -L "$GITHUB_REQ_URL" -o "${REQUIREMENTS}.new"
    # if [ -s "${REQUIREMENTS}.new" ]; then
    #     mv "${REQUIREMENTS}.new" "$REQUIREMENTS"
    #     pip install -r "$REQUIREMENTS" | grep -v 'Requirement already satisfied'
    # fi

    # 5. Перезапуск
    echo "🛑 Остановка бота..."
    pkill -f "$PYTHON_CMD $SCRIPT_NAME"
    
    sleep 2

    echo "🚀 Запуск новой версии..."
    nohup $PYTHON_CMD "$SCRIPT_NAME" > bot_output.log 2>&1 &
    
    # Сохраняем PID нового процесса (полезно для отладки)
    echo $! > bot.pid
    
    echo "✅ Готово! Бот запущен (PID: $(cat bot.pid))."
    echo "📝 Логи: tail -f bot_output.log"

else
    echo "❌ Ошибка: Файл пуст или ссылка неверна (возможно 404)."
    rm -f "${SCRIPT_NAME}.new"
fi