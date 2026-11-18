"""Exploratory analysis: distributions, correlations and five-year trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import (FIG, FEATURES, PROC, REGION_COLORS, TAB, TARGET,
                    load_cross_section, load_panel, plotting_style)


def main() -> None:
    plotting_style()
    cs = load_cross_section()
    panel = load_panel()
    est = cs[cs["is_ut"] == 0].dropna(subset=FEATURES).copy()

    # ------------------------------------------------------------------ tables
    desc_cols = ["ger", "ger_male", "ger_female", "college_density", "ptr", "gpi",
                 "avg_enrolment_per_college", "colleges"]

    def describe(df: pd.DataFrame) -> pd.DataFrame:
        out = df[desc_cols].describe().T
        out["missing"] = df[desc_cols].isna().sum()
        out["cv_pct"] = 100 * out["std"] / out["mean"]
        return out

    # Table 1 in the report is the 28-State estimation sample; the all-unit
    # version is kept alongside it so the UT context remains reproducible.
    describe(est).round(2).to_csv(TAB / "table01_summary_statistics.csv")
    describe(cs).round(2).to_csv(TAB / "table01b_summary_statistics_all_units.csv")

    corr = est[["ger"] + FEATURES + ["avg_enrolment_per_college"]].corr()
    corr.round(3).to_csv(TAB / "table02_correlations.csv")
    spear = est[["ger"] + FEATURES].corr(method="spearman").round(3)
    spear.to_csv(TAB / "table02b_spearman.csv")

    # ---------------------------------------------------------------- figure 1
    d = cs.sort_values("ger").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    y = np.arange(len(d))
    colors = [REGION_COLORS[r] for r in d["region"]]
    ax.hlines(y, 0, d["ger"], color=colors, alpha=0.45, linewidth=1.2)
    ax.scatter(d["ger"], y, s=np.where(d["is_ut"] == 1, 26, 40),
               color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} *" if u == 1 else s for s, u in zip(d["state"], d["is_ut"])],
                       fontsize=7.2)
    ax.axvline(TARGET, color="#B2182B", linestyle="--", linewidth=1.2)
    ax.text(TARGET + 1, 1.5, "NEP 2035 target (50%)", color="#B2182B", fontsize=8)
    for yy, v in zip(y, d["ger"]):
        ax.text(v + 0.8, yy, f"{v:.1f}", va="center", fontsize=6.4, color="#444444")
    ax.set_xlabel("Gross Enrolment Ratio, higher education (%, 18–23 years)")
    ax.set_title("State/UT GER in 2021-22\n(* = Union Territory; colour = region)")
    ax.set_xlim(0, 72)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig01_ger_ranked.png")
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2
    specs = [
        ("college_density", "Colleges per lakh population (18–23)"),
        ("ptr", "Pupil–teacher ratio"),
        ("gpi", "Gender Parity Index"),
        ("avg_enrolment_per_college", "Average enrolment per college"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    for ax, (col, lab) in zip(axes.ravel(), specs):
        sub = cs.dropna(subset=[col, "ger"])
        is_est = sub["state"].isin(est["state"])
        ax.scatter(sub.loc[is_est, col], sub.loc[is_est, "ger"], s=34,
                   c=[REGION_COLORS[r] for r in sub.loc[is_est, "region"]],
                   edgecolor="white", linewidth=0.5, zorder=3, label="States (estimation)")
        ax.scatter(sub.loc[~is_est, col], sub.loc[~is_est, "ger"], s=34,
                   facecolor="white", edgecolor="#777777", linewidth=0.9, zorder=2,
                   label="Union Territories")
        # fit on the 28 estimation states only
        x = est[col].values
        yv = est["ger"].values
        b = np.polyfit(x, yv, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, np.polyval(b, xx), color="#333333", linewidth=1.1, zorder=1)
        r = np.corrcoef(x, yv)[0, 1]
        ax.set_title(f"{lab}\nr = {r:+.2f} (states)")
        ax.set_xlabel(lab)
        ax.set_ylabel("GER (%)")
        if col == "college_density":
            offsets = {
                "Bihar": (4, -10), "Karnataka": (-10, 8), "Tamil Nadu": (-34, 8),
                "Kerala": (-12, 8), "Uttar Pradesh": (-8, -16), "Delhi": (5, 3),
                "Chandigarh": (5, 3),
            }
            for _, row in sub.iterrows():
                if row["state"] in offsets:
                    ax.annotate(row["state"].replace(" and ", " & "), (row[col], row["ger"]),
                                textcoords="offset points", xytext=offsets[row["state"]],
                                fontsize=6.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("GER and its supply-side correlates", fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    fig.savefig(FIG / "fig02_correlates.png")
    plt.close(fig)

    # ---------------------------------------------------------------- figure 3
    cm = est[["ger"] + FEATURES + ["avg_enrolment_per_college"]].corr()
    labels = ["GER", "College density", "PTR", "GPI", "Avg. enrolment"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color="white" if abs(v) > 0.55 else "#222222")
    ax.set_title("Pearson correlations, 28 states (2021-22)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.78, label="r")
    fig.savefig(FIG / "fig03_correlation_heatmap.png")
    plt.close(fig)

    # ---------------------------------------------------------------- figure 4
    p = panel[panel["state"].isin(est["state"])].dropna(subset=["ger"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for state, g in p.groupby("state"):
        g = g.sort_values("year_num")
        ax.plot(g["year_num"], g["ger"], color="#9E9E9E", linewidth=0.55, alpha=0.55)
    mean = p.groupby("year_num")["ger"].mean()
    ax.plot(mean.index, mean.values, color="#B2182B", linewidth=2.2, marker="o",
            label="Unweighted mean of 28 states")
    for state in ["Bihar", "Karnataka", "Tamil Nadu", "Kerala", "Uttar Pradesh"]:
        g = p[p.state == state].sort_values("year_num")
        if len(g):
            ax.plot(g["year_num"], g["ger"], linewidth=1.4, marker="o", markersize=3,
                    label=state)
    ax.axhline(TARGET, color="#B2182B", linestyle=":", linewidth=1)
    ax.text(2017.05, TARGET + 1, "50% NEP target", color="#B2182B", fontsize=7)
    # Sikkim's 2019-20 value is published but is an isolated reporting anomaly.
    sik = p[(p["state"] == "Sikkim") & (p["year"] == "2019-20")]
    if len(sik):
        ax.annotate("Sikkim 2019-20 = 66.9\n(reporting anomaly)",
                    xy=(float(sik["year_num"].iloc[0]), float(sik["ger"].iloc[0])),
                    xytext=(16, -12), textcoords="offset points",
                    fontsize=6.3, color="#333333", ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777"))
    ax.set_xticks(sorted(p["year_num"].unique()))
    ax.set_xticklabels(["2017-18", "2018-19", "2019-20", "2020-21", "2021-22"])
    ax.set_ylabel("GER (%)")
    ax.set_title("GER trajectories of 28 states, 2017-18 to 2021-22")
    ax.legend(ncol=3, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False)
    fig.savefig(FIG / "fig04_trajectories.png")
    plt.close(fig)

    # ------------------------------------------------------------- text output
    print("States in estimation sample:", len(est))
    print("GER mean (28 states):", round(est.ger.mean(), 2), "sd:", round(est.ger.std(), 2))
    print("GER range:", est.ger.min(), "-", est.ger.max())
    print("States below 30% GER:", (est.ger < 30).sum())
    print("States at/above 50%:", (est.ger >= 50).sum())
    print("\nPearson correlations (28 states):")
    print(corr["ger"].round(3).to_string())
    print("\nSpearman correlations (28 states):")
    print(spear["ger"].round(3).to_string())
    print("\nVIF-ready correlation matrix:\n", est[FEATURES].corr().round(2).to_string())
    print("\nMissing cells in analysis variables (all 36 units):")
    print(cs[["ger"] + FEATURES].isna().sum().to_string())

    # normality / outliers for the text
    for c in FEATURES:
        z = (est[c] - est[c].mean()) / est[c].std()
        print(f"extreme {c}:", est.loc[z.abs() > 2.5, "state"].tolist())


if __name__ == "__main__":
    main()
