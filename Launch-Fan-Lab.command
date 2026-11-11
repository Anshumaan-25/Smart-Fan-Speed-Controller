#!/bin/sh
# Use a Python with modern Tk, including when Finder supplies a minimal PATH.
set -eu
cd "$(dirname "$0")"

for candidate in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 \
    /opt/homebrew/Caskroom/miniconda/base/bin/python3 \
    /usr/local/Caskroom/miniconda/base/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys, tkinter; sys.exit(sys.version_info < (3, 9) or tkinter.TkVersion < 8.6)' 2>/dev/null; then
        exec "$candidate" -B fan_lab.py
    fi
done

printf '%s\n' 'Fan Lab needs Python 3.9 or newer with Tk 8.6 or newer.' >&2
exit 1
