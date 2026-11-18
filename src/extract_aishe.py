"""
Extract state/UT-level higher-education indicators for 2017-18 .. 2021-22
from the AISHE 2021-22 Final Report (PDF, 356 pages).

Tables used (Appendix A):
  Table 4    p.72  : colleges registered, college density (per lakh 18-23 pop),
                     average enrolment per college (2021-22)
  Table 42   p.236 : college density, 5-year series
  Table 42   p.235 : average enrolment per college, 5-year series
  Table 47   p.276-279 : GER (all categories), 5-year series
  Table 48   p.280 : GPI (all categories), 5-year series
  Table 49   p.283-286 : PTR (all institutions, regular mode), 5-year series

The parser is deliberately strict: every state row must yield the expected
number of numeric tokens, otherwise the script fails loudly.  A validation
report is printed and written to data/processed/extraction_validation.txt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "aishe_2021_22.pdf"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]

# ----------------------------------------------------------------------------
# Canonical state/UT names and the spellings used in the AISHE PDF
# ----------------------------------------------------------------------------
CANON = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam",
    "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala",
    "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Dadra and Nagar Haveli and Daman and Diu", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]

ALIASES = {
    "a & n islands": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "chhatisgarh": "Chhattisgarh",
    "chhattisgarh": "Chhattisgarh",
    "uttrakhand": "Uttarakhand",
    "uttarakhand": "Uttarakhand",
    "lakshdweep": "Lakshadweep",
    "lakshadweep": "Lakshadweep",
    "the dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "the dadra and nagar haveli & daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "odisha": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
}


def canon_name(raw: str) -> str | None:
    s = re.sub(r"\s+", " ", raw).strip().strip(".")
    key = s.lower()
    if key in ALIASES:
        return ALIASES[key]
    if s in CANON:
        return s
    return None


def page_text(reader: PdfReader, page: int) -> str:
    return reader.pages[page - 1].extract_text() or ""


def numbers(line: str) -> list[str]:
    """Numeric tokens in a line (keeps minus sign; ignores table refs)."""
    return re.findall(r"-?\d+(?:\.\d+)?", line.replace("\u2013", "-"))


# ----------------------------------------------------------------------------
# Individual table parsers
# ----------------------------------------------------------------------------
def parse_table_47(reader: PdfReader) -> pd.DataFrame:
    """GER, all categories (male, female, both) - state blocks of 5 year rows."""
    rows = []
    for page in range(276, 280):
        lines = [l.strip() for l in page_text(reader, page).split("\n") if l.strip()]
        current = None
        for line in lines:
            name = canon_name(line)
            if name:
                current = name
                continue
            m = re.match(r"^(20\d\d-\d\d)\s+(.*)$", line)
            if m and current:
                vals = numbers(m.group(2))
                # Some states report only All-category (A&N, Ladakh, ...): 3 numbers.
                if len(vals) < 3:
                    raise ValueError(f"Table 47: short row {current} {line!r}")
                rows.append({
                    "state": current, "year": m.group(1),
                    "ger_male": float(vals[0]), "ger_female": float(vals[1]),
                    "ger_total": float(vals[2]),
                })
    df = pd.DataFrame(rows)
    _expect(df, 178, "Table 47 (GER)")
    return df


def parse_table_48(reader: PdfReader) -> pd.DataFrame:
    """GPI, all categories - one row per state with five values."""
    rows = []
    for page in [280]:
        for line in page_text(reader, page).split("\n"):
            line = line.strip()
            m = re.match(r"^(\d{1,2})\s+(.+?)\s+([\d.\s]+)$", line)
            if not m:
                continue
            name = canon_name(m.group(2))
            if not name:
                continue
            vals = numbers(m.group(3))
            if len(vals) == 5:
                rows.append({"state": name, **dict(zip(YEARS, map(float, vals)))})
            elif len(vals) == 3:
                # Ladakh: reported only for 2019-20 onwards
                rows.append({"state": name, **dict(zip(YEARS[-3:], map(float, vals)))})
    df = pd.DataFrame(rows)
    _expect(df, 36, "Table 48 (GPI)")
    return df.melt(id_vars="state", var_name="year", value_name="gpi")


def parse_table_42_density(reader: PdfReader) -> pd.DataFrame:
    """College density per lakh 18-23 population."""
    rows = []
    for line in page_text(reader, 236).split("\n"):
        line = line.strip()
        m = re.match(r"^(\d{1,2})\s+(.+?)\s+([\d.\s]+)$", line)
        if not m:
            continue
        name = canon_name(m.group(2))
        if not name:
            continue
        vals = numbers(m.group(3))
        if len(vals) != 5:
            # Ladakh: reported only for 2019-20 onwards
            if name == "Ladakh" and len(vals) == 3:
                vals = [float("nan")] * 2 + vals
            else:
                raise ValueError(f"Table 42 density: {name} -> {vals}")
        rows.append({"state": name, **dict(zip(YEARS, map(float, vals)))})
    df = pd.DataFrame(rows)
    _expect(df, 35, "Table 42 (college density)")  # Lakshadweep not reported
    return df.melt(id_vars="state", var_name="year", value_name="college_density")


def parse_table_42_enrolment(reader: PdfReader) -> pd.DataFrame:
    """Average enrolment per college."""
    rows = []
    for line in page_text(reader, 235).split("\n"):
        line = line.strip()
        m = re.match(r"^(\d{1,2})\s+(.+?)\s+([\d.\s]+)$", line)
        if not m:
            continue
        name = canon_name(m.group(2))
        if not name:
            continue
        vals = numbers(m.group(3))
        if len(vals) != 5:
            if name == "Ladakh" and len(vals) == 3:
                vals = [float("nan")] * 2 + vals
            else:
                raise ValueError(f"Table 42 enrolment: {name} -> {vals}")
        rows.append({"state": name, **dict(zip(YEARS, map(float, vals)))})
    df = pd.DataFrame(rows)
    _expect(df, 35, "Table 42 (average enrolment)")
    return df.melt(id_vars="state", var_name="year", value_name="avg_enrolment_per_college")


def parse_table_49(reader: PdfReader) -> pd.DataFrame:
    """PTR, all institutions, regular mode (second value in each year row)."""
    rows = []
    for page in range(283, 287):
        lines = [l.strip() for l in page_text(reader, page).split("\n") if l.strip()]
        current = None
        for line in lines:
            name = canon_name(line)
            if name:
                current = name
                continue
            m = re.match(r"^(20\d\d-\d\d)\s+(.*)$", line)
            if m and current:
                vals = numbers(m.group(2))
                if len(vals) < 2:
                    raise ValueError(f"Table 49: short row {current} {line!r}")
                rows.append({
                    "state": current, "year": m.group(1),
                    # col 1 = All Institutions, Regular & Distance; col 2 = Regular
                    "ptr": float(vals[1]),
                })
    df = pd.DataFrame(rows)
    _expect(df, 178, "Table 49 (PTR)")
    return df


def parse_table_4(reader: PdfReader) -> pd.DataFrame:
    """2021-22 snapshot: number of colleges, density, average enrolment."""
    rows = []
    for line in page_text(reader, 72).split("\n"):
        line = line.strip()
        m = re.match(r"^(\d{1,2})\s+(.+?)\s+([\d.\s]+)$", line)
        if not m:
            continue
        name = canon_name(m.group(2))
        if not name:
            continue
        vals = numbers(m.group(3))
        if len(vals) == 3:
            rows.append({
                "state": name,
                "colleges": float(vals[0]),
                "college_density_t4": float(vals[1]),
                "avg_enrolment_t4": float(vals[2]),
            })
    df = pd.DataFrame(rows)
    _expect(df, 35, "Table 4 (snapshot)")  # Lakshadweep not reported
    return df


def parse_india_ger(reader: PdfReader) -> float | None:
    """All-India GER (all categories), 2021-22, from Table 19 (p. 156)."""
    for line in page_text(reader, 156).split("\n"):
        if line.strip().lower().startswith("all india"):
            vals = numbers(line)
            if len(vals) >= 3:
                return float(vals[2])
    return None


def _expect(df: pd.DataFrame, n: int, label: str) -> None:
    if len(df) != n:
        raise AssertionError(f"{label}: expected {n} rows, got {len(df)}")
    key = "state" if "year" not in df.columns else ["state", "year"]
    if df.duplicated(key).any():
        dup = df.loc[df.duplicated(key), key].drop_duplicates()
        raise AssertionError(f"{label}: duplicate keys {dup.to_dict('records')[:5]}")


def main() -> None:
    reader = PdfReader(PDF)
    ger = parse_table_47(reader)
    gpi = parse_table_48(reader)
    density = parse_table_42_density(reader)
    enrol = parse_table_42_enrolment(reader)
    ptr = parse_table_49(reader)
    snap = parse_table_4(reader)

    merged = (ger.merge(gpi, on=["state", "year"], how="outer")
                 .merge(density, on=["state", "year"], how="outer")
                 .merge(enrol, on=["state", "year"], how="outer")
                 .merge(ptr, on=["state", "year"], how="outer"))
    merged = merged.sort_values(["state", "year"]).reset_index(drop=True)
    merged.to_csv(OUT / "aishe_panel_raw.csv", index=False)
    snap.to_csv(OUT / "aishe_snapshot_2021_22_raw.csv", index=False)

    # ---------------- validation against published point estimates ----------
    lines = ["AISHE extraction validation", "=" * 60, ""]
    lines.append(f"panel rows: {len(merged)} (expected {36*5})")
    lines.append(f"states: {merged['state'].nunique()}")
    missing = merged[merged.isna().any(axis=1)]
    lines.append(f"rows with missing values: {len(missing)}")
    if len(missing):
        lines.append(missing.to_string(index=False))

    snap21 = merged[merged.year == "2021-22"].set_index("state")
    s4 = snap.set_index("state")
    common = snap21.index.intersection(s4.index)
    d = (snap21.loc[common, "college_density"] - s4.loc[common, "college_density_t4"]).abs()
    e = (snap21.loc[common, "avg_enrolment_per_college"] - s4.loc[common, "avg_enrolment_t4"]).abs()
    lines.append(f"\nTable 42 vs Table 4: max |density diff| = {d.max():.3f}")
    lines.append(f"Table 42 vs Table 4: max |enrolment diff| = {e.max():.3f}")
    if d.max() > 0 or e.max() > 0:
        lines.append(d[d > 0].to_string())
        lines.append(e[e > 0].to_string())

    # external check: 2021-22 GER against the OGD release
    ogd_path = ROOT / "data" / "raw" / "ogd_ger_2021_22.json"
    if ogd_path.exists():
        records = json.loads(ogd_path.read_text(encoding="utf-8"))["records"]
        ogd = pd.DataFrame(records)
        ogd["state"] = ogd["states_uts"].map(canon_name)
        ogd = ogd.dropna(subset=["state"]).set_index("state")["ger_2021_22"]
        common_ogd = snap21.index.intersection(ogd.index)
        diff_ogd = (snap21.loc[common_ogd, "ger_total"] - ogd.loc[common_ogd]).abs()
        lines.append(f"GER 2021-22 vs OGD: max abs diff = {diff_ogd.max():.3f} "
                     f"(n = {len(common_ogd)})")
        if diff_ogd.max() > 0:
            lines.append(diff_ogd[diff_ogd > 0].to_string())

    # internal consistency: GPI == female GER / male GER
    chk = snap21.copy()
    chk["gpi_implied"] = chk["ger_female"] / chk["ger_male"]
    chk["gpi_diff"] = (chk["gpi"] - chk["gpi_implied"]).abs()
    lines.append(f"\nGPI vs female/male GER: max abs diff = {chk['gpi_diff'].max():.3f}")
    gpi_bad = chk.loc[chk["gpi_diff"] > 0.02,
                      ["ger_male", "ger_female", "gpi", "gpi_implied"]]
    if len(gpi_bad):
        lines.append(gpi_bad.to_string())

    lines.append("\nHeadline checks (2021-22):")
    checks = {
        "All-India GER (Table 19)": parse_india_ger(reader),
        "Karnataka college density": snap21.loc["Karnataka", "college_density"],
        "Bihar PTR": snap21.loc["Bihar", "ptr"],
        "Kerala GPI": snap21.loc["Kerala", "gpi"],
    }
    for k, v in checks.items():
        lines.append(f"  {k}: {v}")

    report = "\n".join(lines)
    (OUT / "extraction_validation.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

