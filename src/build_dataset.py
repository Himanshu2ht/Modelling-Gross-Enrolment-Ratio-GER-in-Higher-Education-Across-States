"""
Build analysis-ready datasets from the extracted AISHE panel and the OGD target.

Outputs
-------
data/processed/ger_state_2021_22.csv : 36 states/UTs, 2021-22 cross-section
data/processed/ger_panel_2017_22.csv  : 36 states/UTs x 5 years (unbalanced)
data/processed/data_dictionary.md     : variable definitions and provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

UTS = {
    "Andaman and Nicobar Islands", "Chandigarh", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
    "Dadra and Nagar Haveli and Daman and Diu",
}

REGION = {
    "Andaman and Nicobar Islands": "East",
    "Andhra Pradesh": "South", "Arunachal Pradesh": "Northeast", "Assam": "Northeast",
    "Bihar": "East", "Chandigarh": "North", "Chhattisgarh": "Central", "Delhi": "North",
    "Goa": "West", "Gujarat": "West", "Haryana": "North", "Himachal Pradesh": "North",
    "Jammu and Kashmir": "North", "Jharkhand": "East", "Karnataka": "South",
    "Kerala": "South", "Ladakh": "North", "Lakshadweep": "South",
    "Madhya Pradesh": "Central", "Maharashtra": "West", "Manipur": "Northeast",
    "Meghalaya": "Northeast", "Mizoram": "Northeast", "Nagaland": "Northeast",
    "Odisha": "East", "Puducherry": "South", "Punjab": "North", "Rajasthan": "North",
    "Sikkim": "Northeast", "Tamil Nadu": "South", "Telangana": "South",
    "Dadra and Nagar Haveli and Daman and Diu": "West", "Tripura": "Northeast",
    "Uttar Pradesh": "Central", "Uttarakhand": "North", "West Bengal": "East",
}


def load_ogd() -> pd.DataFrame:
    """Official OGD/AISHE GER target for 2021-22 (36 records)."""
    payload = json.loads((RAW / "ogd_ger_2021_22.json").read_text(encoding="utf-8"))
    rec = pd.DataFrame(payload["records"])
    rec = rec.rename(columns={
        "_sl__no_": "sl_no", "states_uts": "state", "ger_2021_22": "ger_ogd",
    })
    rec["state"] = rec["state"].str.strip()
    rec["ger_ogd"] = rec["ger_ogd"].astype(float)
    assert len(rec) == 36, len(rec)
    return rec[["state", "ger_ogd"]]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_ut"] = df["state"].isin(UTS).astype(int)
    df["region"] = df["state"].map(REGION)
    assert df["region"].notna().all()
    df["log_college_density"] = np.log(df["college_density"])
    df["gender_gap_abs"] = (df["gpi"] - 1).abs()
    return df


def main() -> None:
    ger_ogd = load_ogd()
    panel = pd.read_csv(PROC / "aishe_panel_raw.csv")

    # ---- validate the PDF extraction against the OGD target -----------------
    chk = panel[panel.year == "2021-22"].merge(ger_ogd, on="state", how="outer")
    assert len(chk) == 36, len(chk)
    diff = (chk["ger_total"] - chk["ger_ogd"]).abs()
    assert diff.max() < 1e-9, chk.loc[diff > 1e-9, ["state", "ger_total", "ger_ogd"]]
    print(f"Target cross-check: 36/36 state GER values match OGD exactly (max diff {diff.max():.2e})")

    panel = panel.merge(ger_ogd, on="state", how="left")
    # OGD is the authoritative target for 2021-22; other years come from Table 47
    panel["ger"] = np.where(panel["year"] == "2021-22", panel["ger_ogd"], panel["ger_total"])
    panel = add_derived(panel)
    panel["year_num"] = panel["year"].str.slice(0, 4).astype(int)
    panel = panel.drop(columns=["ger_ogd", "ger_total"])

    # ---- 2021-22 cross-section ---------------------------------------------
    cs = panel[panel.year == "2021-22"].copy().reset_index(drop=True)
    cs = cs.sort_values("state").reset_index(drop=True)

    # Number of colleges from Table 4 (validated against Table 42 density)
    snap = pd.read_csv(PROC / "aishe_snapshot_2021_22_raw.csv").set_index("state")
    cs["colleges"] = cs["state"].map(snap["colleges"])
    common = cs.dropna(subset=["college_density"]).set_index("state")
    assert (common["college_density"] - snap.loc[common.index, "college_density_t4"]).abs().max() == 0
    assert (common["avg_enrolment_per_college"] - snap.loc[common.index, "avg_enrolment_t4"]).abs().max() == 0

    # Four-year momentum in the covariates (2017-18 -> 2021-22)
    base = panel[panel.year == "2017-18"].set_index("state")
    for col in ["college_density", "ptr", "gpi"]:
        cs[f"delta_{col}_4y"] = cs["state"].map(
            cs.set_index("state")[col] - base[col]
        )
        cs[f"cagr_{col}_4y"] = cs["state"].map(
            (cs.set_index("state")[col] / base[col]) ** (1 / 4) - 1
        )

    cs = cs[[
        "state", "region", "is_ut", "ger", "ger_male", "ger_female",
        "gpi", "college_density", "ptr", "avg_enrolment_per_college",
        "log_college_density", "gender_gap_abs", "colleges",
        "delta_college_density_4y", "delta_ptr_4y", "delta_gpi_4y",
        "cagr_college_density_4y", "cagr_ptr_4y", "cagr_gpi_4y",
    ]]
    cs.to_csv(PROC / "ger_state_2021_22.csv", index=False)

    panel = panel[[
        "state", "year", "year_num", "region", "is_ut", "ger", "ger_male", "ger_female",
        "gpi", "college_density", "ptr", "avg_enrolment_per_college",
        "log_college_density", "gender_gap_abs",
    ]].sort_values(["state", "year"]).reset_index(drop=True)
    panel.to_csv(PROC / "ger_panel_2017_22.csv", index=False)

    print(f"cross-section: {cs.shape}, complete cases: {cs.dropna().shape[0]}")
    print(f"panel: {panel.shape}, complete cases: {panel.dropna().shape[0]}")
    print("\nMissing in cross-section:")
    print(cs.isna().sum()[cs.isna().sum() > 0].to_string())
    print("\nGER summary:", cs["ger"].describe().round(2).to_dict())
    print("\nCorrelations with GER:")
    print(cs[["ger", "college_density", "ptr", "gpi", "avg_enrolment_per_college"]]
          .corr()["ger"].round(3).to_string())

    (PROC / "data_dictionary.md").write_text(DATA_DICT, encoding="utf-8")


DATA_DICT = """# Data dictionary

Unit of observation: State / Union Territory of India.
Primary year: 2021-22 (AISHE Final Report 2021-22; OGD release).

| Variable | Definition | Source |
|---|---|---|
| `state` | State/UT (36 units) | AISHE / OGD |
| `region` | Descriptive region (North, Central, East, Northeast, West, South) | Author classification |
| `is_ut` | 1 if Union Territory | Author classification |
| `ger` | Gross Enrolment Ratio in higher education, 18-23 years, all categories (%) | OGD resource `7edbaeed-...`; AISHE Table 19 |
| `ger_male`, `ger_female` | Gender-specific GER (%) | AISHE Table 19 |
| `gpi` | Gender Parity Index = female GER / male GER (all categories) | AISHE Table 20 / 48 |
| `college_density` | Colleges registered with AISHE per lakh population aged 18-23 | AISHE Table 4 / 42 |
| `ptr` | Pupil-teacher ratio, all institutions, regular mode | AISHE Table 49 |
| `avg_enrolment_per_college` | Average enrolment per college (incl. estimated enrolment) | AISHE Table 4 / 42 |
| `colleges` | Number of colleges registered with AISHE, 2021-22 | AISHE Table 4 |
| `log_college_density` | ln(college_density); derived but not used in the reported models | Derived |
| `gender_gap_abs` | abs(GPI - 1), descriptive only | Derived |
| `delta_*_4y` | Change 2017-18 to 2021-22 | Derived from AISHE Tables 42/48/49 |
| `cagr_*_4y` | Compound annual growth rate 2017-18 to 2021-22 | Derived |

Notes
- Lakshadweep does not report college counts/density in AISHE 2021-22, so it has
  missing predictors and is excluded from the 35-unit UT-inclusive sample; the
  primary estimation sample is the 28 States.
- Ladakh was created as a UT in October 2019 and has only three years of
  college-density/PTR data.
- All India values are excluded from the state-level modelling file.
- GER for 2021-22 is provisional in the OGD release.
- AISHE 2021-22 uses population projections based on the 2011 Census.
"""


if __name__ == "__main__":
    main()
