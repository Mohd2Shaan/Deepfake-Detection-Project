"""Render report/report.html to PDF with headless Chrome (or Edge).

    python report/make_figures.py
    python report/build_report.py               # public version, no personal names -> report/
    python report/build_report.py --with-names  # adds author/professor from report/private/details.json
                                                #   -> report/private/ (git-ignored, never pushed)
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"
BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--with-names", action="store_true", help="include author/professor (private copy)")
args = parser.parse_args()

html = (HERE / "report.html").read_text(encoding="utf-8")
rows = ""
if args.with_names:
    details = json.loads((PRIVATE / "details.json").read_text(encoding="utf-8"))
    rows = (f"    <tr><td>Author</td><td>{details['author']}</td></tr>\n"
            f"    <tr><td>Submitted to</td><td>{details['submitted_to']}</td></tr>\n")
html = html.replace("    <!--PERSONAL_DETAILS-->\n", rows)
OUT = (PRIVATE if args.with_names else HERE) / "Deepfake_Detection_Report.pdf"

browser = next((b for b in BROWSERS if b.exists()), None)
if browser is None:
    raise SystemExit("Chrome/Edge not found - open report.html in a browser and use Print -> Save as PDF")

with tempfile.TemporaryDirectory() as tmp:
    pdf = Path(tmp) / "report.pdf"
    page = HERE / "_render.html"              # temporary copy next to figures/ so relative paths work
    page.write_text(html, encoding="utf-8")
    try:
        subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}",
                        page.as_uri()], check=True, capture_output=True)
    finally:
        page.unlink()
    shutil.copy(pdf, OUT)
print("written", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
