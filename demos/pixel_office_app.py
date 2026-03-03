"""
NovaAware Consciousness Dashboard — Desktop Application
像素风办公室 + 意识仪表盘整合版桌面应用

Usage / 用法:
  python3 demos/pixel_office_app.py
"""

import os
import sys
from pathlib import Path

try:
    import webview
except ImportError:
    print("[!] Installing pywebview...")
    os.system(f"{sys.executable} -m pip install pywebview")
    import webview

HTML_PATH = Path(__file__).parent / "pixel_office.html"


def main():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   NovaAware — Consciousness Dashboard (Desktop)      ║")
    print("  ║   Pixel Office + C1-C4 Conditions + Reactions        ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    if not HTML_PATH.exists():
        print(f"  [Error] {HTML_PATH} not found!")
        sys.exit(1)

    print(f"  [UI] Loading {HTML_PATH.name}")
    print("  [Keys] 1-5 switch emotion, Space auto-cycle")
    print()

    window = webview.create_window(
        title="NovaAware — Consciousness Dashboard",
        url=str(HTML_PATH),
        width=1360,
        height=860,
        min_size=(1024, 700),
        background_color="#06081a",
        text_select=False,
    )

    webview.start(debug=False)
    print("\n  [Stopped] Goodbye.")


if __name__ == "__main__":
    main()
