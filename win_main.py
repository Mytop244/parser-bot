import subprocess      # Модуль для запуска внешних процессов (например, python main.py)
import sys             # Доступ к системной информации (аргументы запуска, путь к exe)
import os              # Работа с путями, файлами и директориями
import time            # Модуль времени (но в коде НЕ используется)

# ---- КОНФИГУРАЦИЯ -----------------------------------

# Абсолютный путь к Python (жёстко прописан, должен существовать на ПК)
PYTHON = r"C:\Users\MSI\AppData\Local\Programs\Python\Python313\python.exe"

# Путь к директории, где лежит этот файл (берётся из sys.argv[0], что иногда НЕПРАВИЛЬНО для exe)
BASE = os.path.dirname(sys.argv[0])

# Полный путь к main.py, который должен быть в этой же папке
MAIN_FILE = os.path.join(BASE, "main.py")

# ------------------------------------------------------

def start_main():
    print("Starting main.py...")                  # Лог в консоль
    return subprocess.Popen([PYTHON, MAIN_FILE]) # Запуск main.py через внешний Python

if __name__ == "__main__":                       # Проверка: код запущен напрямую, а не импортирован
    print("Launcher: starting main.py once.")    # Лог о начале работы лаунчера
    p = start_main()                              # Запуск main.py
    p.wait()                                      # Ожидание завершения main.py (блокировка до конца)
