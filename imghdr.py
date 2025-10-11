# Минимальная замена модуля imghdr.what для Python 3.13
def _match_magic(h, magics):
    for name, sigs in magics.items():
        for sig in sigs:
            if h.startswith(sig):
                return name
    return None


def what(file, h=None):
    """
    Минимальная замена imghdr.what для Python 3.13.
    `file` может быть путём или объектом с read(); `h` — опциональные первые байты.
    """
    magics = {
        "jpeg": [b"\xff\xd8\xff"],
        "png": [b"\x89PNG\r\n\x1a\n"],
        "gif": [b"GIF87a", b"GIF89a"],
        "webp": [b"RIFF"],  # потребует доп. проверки 'WEBP' в заголовке
        "bmp": [b"BM"],
    }

    # получаем первые 64 байта
    header = b""
    if h:
        header = h if isinstance(h, (bytes, bytearray)) else bytes(h)
    else:
        try:
            if hasattr(file, "read"):
                pos = None
                try:
                    pos = file.tell()
                except Exception:
                    pos = None
                header = file.read(64)
                if pos is not None:
                    try:
                        file.seek(pos)
                    except Exception:
                        pass
            else:
                with open(file, "rb") as f:
                    header = f.read(64)
        except Exception:
            return None

    if not header:
        return None

    # webp special-case: 'RIFF' + 4 bytes + 'WEBP'
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "webp"

    return _match_magic(header, magics)
