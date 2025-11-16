import threading
import subprocess
import sys
import os
import time
import signal
import PySimpleGUI as sg
import pystray
from PIL import Image, ImageDraw
import winreg

BASE_DIR = os.path.dirname(__file__)
MAIN_FILE = os.path.join(BASE_DIR, "main.py")
RSS_FILE = os.path.join(BASE_DIR, "rss.txt")
LOG_FILE = os.path.join(BASE_DIR, "parser.log")

process = None
process_lock = threading.Lock()
need_restart = True
last_log_mtime = 0
start_time = 0
last_rss_mtime = 0
log_rotation_size = 20 * 1024 * 1024  # 20MB

# ---------------- ICONS ----------------
def make_icon(color):
    img = Image.new("RGB", (64, 64), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 54, 54), fill=color)
    return img

icon_green = make_icon("green")
icon_red = make_icon("red")
icon_yellow = make_icon("yellow")

# ---------------- LOG ROTATION ----------------
def rotate_logs():
    if not os.path.exists(LOG_FILE):
        return

    if os.path.getsize(LOG_FILE) > log_rotation_size:
        backup = LOG_FILE + ".1"
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(LOG_FILE, backup)
        with open(LOG_FILE, "w", encoding="utf-8"):
            pass  # create empty file

# ---------------- AUTOSTART ----------------
def enable_autostart():
    exe = sys.executable
    script = os.path.abspath(__file__)
    launcher_cmd = f"\"{exe}\" \"{script}\""
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0, winreg.KEY_SET_VALUE
    )
    winreg.SetValueEx(key, "NewsBotLauncher", 0, winreg.REG_SZ, launcher_cmd)
    winreg.CloseKey(key)

def disable_autostart():
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, "NewsBotLauncher")
        winreg.CloseKey(key)
    except:
        pass

# ---------------- PROCESS CONTROL ----------------
def start_bot():
    global process, need_restart, start_time
    with process_lock:
        if process and process.poll() is None:
            return
        rotate_logs()
        process = subprocess.Popen([sys.executable, MAIN_FILE])
        start_time = time.time()
        need_restart = True

def stop_bot():
    global process, need_restart
    need_restart = False
    with process_lock:
        if process and process.poll() is None:
            try:
                process.terminate()
            except:
                pass

# ---------------- UI WINDOWS ----------------
def open_rss_editor():
    global last_rss_mtime

    if not os.path.exists(RSS_FILE):
        with open(RSS_FILE, "w", encoding="utf-8") as f:
            f.write("")

    with open(RSS_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    layout = [
        [sg.Text("RSS источники:")],
        [sg.Multiline(text, size=(60, 15), key="-RSS-")],
        [sg.Button("Сохранить и применить"), sg.Button("Закрыть")]
    ]
    window = sg.Window("RSS Editor", layout)

    while True:
        ev, val = window.read()
        if ev in (sg.WIN_CLOSED, "Закрыть"):
            break

        if ev == "Сохранить и применить":
            with open(RSS_FILE, "w", encoding="utf-8") as f:
                f.write(val["-RSS-"])

            last_rss_mtime = os.path.getmtime(RSS_FILE)
            sg.popup("Сохранено.\nmain.py будет перезапущен при следующем цикле.")
            break

    window.close()


def open_log_viewer():
    layout = [
        [sg.Text("Логи parser.log (auto-refresh):")],
        [sg.Multiline("", size=(90, 30), key="-LOG-", disabled=True)],
        [sg.Button("Закрыть")]
    ]
    window = sg.Window("Live Logs", layout, finalize=True)

    while True:
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    window["-LOG-"].update("".join(f.readlines()[-5000:]))
        except:
            pass

        ev, _ = window.read(timeout=1000)
        if ev in (sg.WIN_CLOSED, "Закрыть"):
            break

    window.close()


def open_status_window():
    layout = [
        [sg.Text("Live Status Monitor", font=("Arial", 14))],
        [sg.Text("Статус:"), sg.Text("", key="-STATE-")],
        [sg.Text("Работает секунд:"), sg.Text("0", key="-RUN-")],
        [sg.Text("Лог обновлён:"), sg.Text("—", key="-LOGTIME-")],
        [sg.Button("Force Restart"), sg.Button("Закрыть")]
    ]

    window = sg.Window("Bot Status", layout, finalize=True)

    while True:
        with process_lock:
            running = process and process.poll() is None

        if running:
            state = "RUNNING"
            run_seconds = int(time.time() - start_time)
        else:
            state = "STOPPED"
            run_seconds = 0

        try:
            if os.path.exists(LOG_FILE):
                log_mtime = os.path.getmtime(LOG_FILE)
                log_age = int(time.time() - log_mtime)
                log_age_txt = f"{log_age} сек назад"
            else:
                log_age_txt = "нет логов"
        except:
            log_age_txt = "нет данных"

        window["-STATE-"].update(state)
        window["-RUN-"].update(str(run_seconds))
        window["-LOGTIME-"].update(log_age_txt)

        ev, _ = window.read(timeout=1000)
        if ev in (sg.WIN_CLOSED, "Закрыть"):
            break
        if ev == "Force Restart":
            stop_bot()
            time.sleep(0.5)
            start_bot()

    window.close()

# ---------------- WATCHDOG ----------------
def watchdog_loop(tray_icon):
    global last_log_mtime, last_rss_mtime

    while True:
        time.sleep(3)

        with process_lock:
            running = process and process.poll() is None

        # Иконка
        tray_icon.icon = icon_green if running else icon_red

        # Перезапуск при падении
        if not running and need_restart:
            start_bot()

        # Проверяем лог (зависание)
        if running:
            if os.path.exists(LOG_FILE):
                mtime = os.path.getmtime(LOG_FILE)

                if last_log_mtime == 0:
                    last_log_mtime = mtime
                else:
                    if mtime == last_log_mtime:  # лог не меняется
                        if time.time() - mtime > 25:
                            try:
                                process.kill()
                            except:
                                pass
                            continue

                last_log_mtime = mtime

        # Перезапуск при изменении RSS
        if os.path.exists(RSS_FILE):
            mtime = os.path.getmtime(RSS_FILE)
            if last_rss_mtime == 0:
                last_rss_mtime = mtime
            else:
                if mtime != last_rss_mtime:
                    stop_bot()
                    time.sleep(1)
                    start_bot()
                    last_rss_mtime = mtime

# ---------------- TRAY ----------------
icon = pystray.Icon(
    "newsbot",
    icon_green,
    "NewsBot",
    menu=pystray.Menu(
        pystray.MenuItem("Start", lambda i, item: start_bot()),
        pystray.MenuItem("Stop", lambda i, item: stop_bot()),
        pystray.MenuItem("Status", lambda i, item: open_status_window()),
        pystray.MenuItem("Edit RSS", lambda i, item: open_rss_editor()),
        pystray.MenuItem("View Logs", lambda i, item: open_log_viewer()),
        pystray.MenuItem("Quit", lambda i, item: (stop_bot(), icon.stop(), os._exit(0))),
    )
)

# ---------------- START ----------------
if __name__ == "__main__":
    enable_autostart()  # включаем автозапуск
    start_bot()

    t = threading.Thread(target=watchdog_loop, args=(icon,), daemon=True)
    t.start()

    icon.run()
