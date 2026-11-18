"""
Download the primary sources and verify their integrity.

1. OGD/AISHE state GER 2021-22 (resource 7edbaeed-65e9-43ad-8f81-730187b70463)
   The OGD web page blocks direct file downloads (HTTP 403), so the resource is
   pulled through the official OGD API in pages of 10 records.
2. AISHE Final Report 2021-22 (PDF, 356 pp.) for the predictor tables.

Run:  python src/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

RESOURCE_ID = "7edbaeed-65e9-43ad-8f81-730187b70463"
# Public sample key published by OGD for API exploration; override with env var.
API_KEY = os.environ.get(
    "OGD_API_KEY", "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
)
OGD_PAGE = (
    "https://www.data.gov.in/resource/"
    "statesut-wise-gross-enrolment-ratio-ger-higher-education-based-2011-population-census"
)
AISHE_PDF = (
    "https://cdnbbsr.s3waas.gov.in/s392049debbe566ca5782a3045cf300a3c/"
    "uploads/2025/06/2025060466438560.pdf"
)


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (research replication script)",
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_ogd() -> None:
    out = RAW / "ogd_ger_2021_22.json"
    records, meta = [], None
    offset = 0
    while True:
        url = (
            f"https://api.data.gov.in/resource/{RESOURCE_ID}"
            f"?api-key={API_KEY}&format=json&limit=10&offset={offset}"
        )
        payload = json.loads(fetch(url))
        if meta is None:
            meta = {k: payload[k] for k in ("total", "title", "desc", "created_date",
                                            "updated_date", "source", "org_type") if k in payload}
        records.extend(payload["records"])
        offset += 10
        if offset >= int(payload["total"]):
            break
        time.sleep(0.4)
    assert len(records) == 36, f"expected 36 records, got {len(records)}"
    out.write_text(json.dumps({
        "resource_id": RESOURCE_ID,
        "resource_page": OGD_PAGE,
        "api_endpoint": f"https://api.data.gov.in/resource/{RESOURCE_ID}",
        "retrieved": time.strftime("%Y-%m-%d"),
        "metadata": meta,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"OGD: {len(records)} records -> {out.relative_to(ROOT)}")


def download_pdf() -> None:
    out = RAW / "aishe_2021_22.pdf"
    if out.exists() and out.stat().st_size > 5_000_000:
        print(f"AISHE PDF already present ({out.stat().st_size/1e6:.1f} MB)")
        return
    data = fetch(AISHE_PDF, timeout=300)
    assert data[:4] == b"%PDF", "downloaded file is not a PDF"
    out.write_bytes(data)
    print(f"AISHE PDF: {len(data)/1e6:.1f} MB -> {out.relative_to(ROOT)}")


def checksums() -> None:
    lines = ["# SHA-256 checksums of raw inputs", ""]
    for p in sorted(RAW.glob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.name}  ({p.stat().st_size} bytes)")
    (RAW / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    download_ogd()
    download_pdf()
    checksums()
