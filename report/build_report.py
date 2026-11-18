"""
Build the submission report from the Markdown template.

Steps
-----
1. Read ``report/report_meta.json`` and fill the author block.
2. Render ``report/report_template.md`` to ``report/report.md``.
3. Convert the Markdown to a print-ready HTML file with journal-style CSS.
4. If Edge, Chrome or Chromium is available, print the HTML to PDF.
   Otherwise the HTML is kept and can be printed from any browser.

Usage
-----
    python report/build_report.py

The script is cross-platform.  On Linux it looks for ``google-chrome``,
``chromium`` or ``chromium-browser`` on PATH; on Windows it looks for Edge and
Chrome in their standard install locations.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "report_template.md"
MD_OUT = HERE / "report.md"
HTML_OUT = HERE / "report.html"
PDF_OUT = HERE / "Problem18_GER_Report.pdf"

FALLBACKS = {
    "author": "Author name — set in report/report_meta.json",
    "college": "College name — set in report/report_meta.json",
    "roll": "Roll number — set in report/report_meta.json",
    "email": "E-mail — set in report/report_meta.json",
    "supervisor": "Supervisor — set in report/report_meta.json",
}

CSS = r"""
@page {
  size: A4;
  margin: 15mm 15mm 16mm 15mm;
}
* { box-sizing: border-box; }
html { font-size: 10.3pt; }
body {
  font-family: "Cambria", "Georgia", "Times New Roman", serif;
  color: #111;
  line-height: 1.42;
  margin: 0;
  text-align: justify;
  hyphens: auto;
}
h1.title {
  font-size: 17.5pt;
  line-height: 1.2;
  margin: 0 0 6pt 0;
  text-align: left;
  font-weight: 700;
}
p.authors { font-size: 11pt; margin: 2pt 0 1pt 0; text-align: left; }
p.affil { font-size: 9pt; margin: 0 0 2pt 0; text-align: left; color: #222; }
p.meta { font-size: 8.6pt; margin: 0 0 8pt 0; text-align: left; color: #333; }
hr.rule { border: 0; border-top: 1.2pt solid #222; margin: 6pt 0 8pt 0; }
h2 {
  font-size: 12.2pt;
  margin: 12pt 0 4pt 0;
  text-align: left;
  page-break-after: avoid;
}
h3 {
  font-size: 10.6pt;
  margin: 9pt 0 3pt 0;
  text-align: left;
  font-style: italic;
  page-break-after: avoid;
}
p { margin: 0 0 5pt 0; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 16pt; }
li { margin-bottom: 2pt; }
strong { font-weight: 700; }
em { font-style: italic; }
code {
  font-family: "Consolas", "DejaVu Sans Mono", monospace;
  font-size: 8.6pt;
  background: #f4f4f4;
  padding: 0 1.5pt;
}
pre {
  font-family: "Consolas", "DejaVu Sans Mono", monospace;
  font-size: 8.2pt;
  background: #f6f6f6;
  border-left: 2pt solid #999;
  padding: 5pt 7pt;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 7.9pt;
  line-height: 1.22;
  margin: 4pt 0 7pt 0;
  page-break-inside: avoid;
}
table caption {
  caption-side: top;
  text-align: left;
  font-size: 8.4pt;
  font-style: italic;
  padding-bottom: 2pt;
}
th {
  border-top: 1.1pt solid #222;
  border-bottom: 0.7pt solid #222;
  padding: 2.2pt 2.6pt;
  text-align: left;
  vertical-align: bottom;
  font-weight: 700;
}
td {
  border-bottom: 0.35pt solid #ccc;
  padding: 2pt 2.6pt;
  vertical-align: top;
}
tr:last-child td { border-bottom: 1.1pt solid #222; }
figure {
  margin: 6pt 0 8pt 0;
  page-break-inside: avoid;
  text-align: center;
}
figure img {
  max-width: 100%;
  max-height: 108mm;
  display: block;
  margin: 0 auto 3pt auto;
}
figcaption {
  font-size: 8.2pt;
  font-style: italic;
  text-align: left;
  line-height: 1.3;
  color: #222;
}
.small { font-size: 8.6pt; }
.note {
  background: #f7f7f7;
  border-left: 2pt solid #777;
  padding: 5pt 7pt;
  font-size: 8.8pt;
  margin: 6pt 0 8pt 0;
  page-break-inside: avoid;
}
.eq {
  text-align: center;
  font-style: italic;
  margin: 5pt 0 7pt 0;
  page-break-inside: avoid;
}
.refs p {
  font-size: 8.7pt;
  text-indent: -14pt;
  padding-left: 14pt;
  margin-bottom: 3pt;
  text-align: left;
}
.appendix table { font-size: 6.9pt; page-break-inside: auto; }
.appendix td, .appendix th { padding: 1.1pt 1.8pt; }
.appendix tr { page-break-inside: avoid; }
a { color: #111; text-decoration: none; }
"""


def load_meta() -> dict:
    meta = json.loads((HERE / "report_meta.json").read_text(encoding="utf-8"))
    for key, fallback in FALLBACKS.items():
        if not str(meta.get(key, "")).strip():
            meta[key] = fallback
    return meta


def render_markdown(meta: dict) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    for key, value in meta.items():
        text = text.replace("{{" + key.upper() + "}}", str(value))
    MD_OUT.write_text(text, encoding="utf-8")
    return text


def to_html(text: str) -> str:
    body = markdown.markdown(
        text,
        extensions=["tables", "sane_lists", "attr_list", "md_in_html"],
        output_format="html5",
    )
    # Turn Markdown images (alt text = caption) into numbered journal figures.
    body = re.sub(
        r'<p><img alt="([^"]*)" src="([^"]*)"\s*/?></p>',
        lambda m: f'<figure><img alt="{m.group(1)}" src="{m.group(2)}">'
                  f"<figcaption>{m.group(1)}</figcaption></figure>",
        body,
    )
    body = re.sub(
        r'<p><img alt="([^"]*)" src="([^"]*)"\s*/?>\s*</p>',
        lambda m: f'<figure><img alt="{m.group(1)}" src="{m.group(2)}">'
                  f"<figcaption>{m.group(1)}</figcaption></figure>",
        body,
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Modelling State-Level GER in Indian Higher Education</title>
<style>{CSS}</style></head>
<body>{body}</body></html>"""


def find_browser() -> str | None:
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    for name in ("microsoft-edge", "msedge", "google-chrome", "chrome",
                 "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    return None


def print_pdf(browser: str) -> bool:
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={PDF_OUT}",
            HTML_OUT.as_uri(),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"PDF build failed: {exc}")
            return False
    if not PDF_OUT.exists() or PDF_OUT.stat().st_size < 20_000:
        print("PDF was not produced; the HTML is ready for manual printing.")
        if proc.stderr:
            print(proc.stderr[-800:])
        return False
    return True


def main() -> int:
    if not TEMPLATE.exists():
        print(f"Missing template: {TEMPLATE}")
        return 1
    meta = load_meta()
    text = render_markdown(meta)
    html = to_html(text)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {MD_OUT.name} and {HTML_OUT.name}")

    browser = find_browser()
    if not browser:
        print("No Edge/Chrome/Chromium found. Open report.html and print to PDF.")
        return 0
    if not print_pdf(browser):
        return 1
    print(f"Wrote {PDF_OUT.name} ({PDF_OUT.stat().st_size / 1024:.0f} KB)")
    try:
        from pypdf import PdfReader
        print(f"Pages: {len(PdfReader(str(PDF_OUT)).pages)}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
