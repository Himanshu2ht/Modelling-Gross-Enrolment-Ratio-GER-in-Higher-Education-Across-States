"""Shared configuration for the GER modelling project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = ROOT / "report"

for p in (PROC, FIG, TAB, REP):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = 50.0  # NEP 2020 / SDG-4 benchmark for 2035

FEATURES = ["college_density", "ptr", "gpi"]
FEATURE_LABELS = {
    "college_density": "Colleges per lakh population (18–23)",
    "ptr": "Pupil–teacher ratio",
    "gpi": "Gender Parity Index",
    "avg_enrolment_per_college": "Average enrolment per college",
}
SHORT_LABELS = {
    "college_density": "College density",
    "ptr": "PTR",
    "gpi": "GPI",
    "avg_enrolment_per_college": "Avg. enrolment/college",
}

REGION_COLORS = {
    "North": "#0072B2",
    "Central": "#D55E00",
    "East": "#009E73",
    "Northeast": "#CC79A7",
    "West": "#E69F00",
    "South": "#56B4E9",
}

# States used for estimation (28). Union Territories are analysed separately.
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]

UNION_TERRITORIES = [
    "Andaman and Nicobar Islands", "Chandigarh", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
    "Dadra and Nagar Haveli and Daman and Diu",
]


def load_cross_section():
    import pandas as pd
    return pd.read_csv(PROC / "ger_state_2021_22.csv")


def load_panel():
    import pandas as pd
    return pd.read_csv(PROC / "ger_panel_2017_22.csv")


def plotting_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })
