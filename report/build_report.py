"""Render report/report.html to report/Deepfake_Detection_Report.pdf with headless Chrome (or Edge).

    python report/make_figures.py
    python report/build_report.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "Deepfake_Detection_Report.pdf"
BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
]

browser = next((b for b in BROWSERS if b.exists()), None)
if browser is None:
    raise SystemExit("Chrome/Edge not found - open report.html in a browser and use Print -> Save as PDF")

with tempfile.TemporaryDirectory() as tmp:
    pdf = Path(tmp) / "report.pdf"
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}",
                    (HERE / "report.html").as_uri()], check=True, capture_output=True)
    shutil.copy(pdf, OUT)
print("written", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
