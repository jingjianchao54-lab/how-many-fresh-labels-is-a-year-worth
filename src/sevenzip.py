"""Minimal read-only access to .7z members through the system libarchive (ctypes); no third-party package needed."""
import ctypes
import ctypes.util

_lib = ctypes.CDLL(ctypes.util.find_library("archive") or "libarchive.so.13")
_lib.archive_read_new.restype = ctypes.c_void_p
_lib.archive_read_support_format_7zip.argtypes = [ctypes.c_void_p]
_lib.archive_read_support_filter_all.argtypes = [ctypes.c_void_p]
_lib.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.archive_entry_pathname.argtypes = [ctypes.c_void_p]
_lib.archive_entry_pathname.restype = ctypes.c_char_p
_lib.archive_read_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.archive_read_data.restype = ctypes.c_ssize_t
_lib.archive_read_free.argtypes = [ctypes.c_void_p]
_lib.archive_error_string.argtypes = [ctypes.c_void_p]
_lib.archive_error_string.restype = ctypes.c_char_p


def iter_lines(path, wanted):
    """Yield (member name, line bytes) for every line of the members whose names are in `wanted`."""
    a = _lib.archive_read_new()
    _lib.archive_read_support_format_7zip(a)
    _lib.archive_read_support_filter_all(a)
    if _lib.archive_read_open_filename(a, str(path).encode(), 1 << 20) != 0:
        raise OSError(_lib.archive_error_string(a))
    entry = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(1 << 20)
    try:
        while _lib.archive_read_next_header(a, ctypes.byref(entry)) == 0:
            name = _lib.archive_entry_pathname(entry).decode()
            if name not in wanted:
                continue
            tail = b""
            while True:
                n = _lib.archive_read_data(a, buf, len(buf))
                if n < 0:
                    raise OSError(_lib.archive_error_string(a))
                if n == 0:
                    break
                parts = (tail + buf.raw[:n]).split(b"\n")
                tail = parts.pop()
                for p in parts:
                    yield name, p
            if tail:
                yield name, tail
    finally:
        _lib.archive_read_free(a)
