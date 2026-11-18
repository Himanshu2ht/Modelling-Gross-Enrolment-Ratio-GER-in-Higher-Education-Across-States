"""
Project state GER to 2035-36 under alternative policy scenarios.

The projection is deliberately *change-based*: it is anchored to each state's
observed GER in 2021-22 and adds the change implied by (i) the OLS cross-state
coefficients, (ii) the state's projected change in college density, PTR and GPI,
and (iii) a common annual trend estimated from the 2017-22 state panel with year
fixed effects.  This avoids using the cross-sectional intercept to forecast the
*level* of GER, which would embed the 2021-22 cross-section permanently.

Scenarios
---------
S1  Damped            half the 2017-18 to 2021-22 annual density change,
                      half the common annual trend, PTR and GPI unchanged.
S2  Recent trend      full recent annual density change, full common trend,
                      PTR and GPI unchanged.  Reference scenario.
S3  NEP reform        full density change, PTR converges to min(2021 PTR, 20),
                      GPI converges to 1.05 where below, full common trend.

Uncertainty is a state-resampling bootstrap (2,000 draws): the OLS coefficients
are re-estimated on resampled states, the common trend is drawn from its
sampling distribution, and a state-level residual is added from N(0, sigma).
The same draws are used across scenarios (common random numbers).

Outputs
-------
outputs/tables/table06_projections_2035.csv
outputs/tables/table06b_projections_uts.csv
outputs/tables/table08_required_density.csv
outputs/tables/table09_scenario_summary.csv
outputs/tables/table11_state_appendix.csv
outputs/projection_results.json
outputs/figures/fig12_projection_dumbbell.png
outputs/figures/fig13_probability_reaching_50.png
outputs/figures/fig14_scenario_distributions.png
outputs/figures/fig15_required_density.png
"""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D

from config import (FEATURES, FIG, OUT, RANDOM_STATE, REGION_COLORS, TAB,
                    TARGET, load_cross_section, plotting_style)

warnings.filterwarnings("ignore")

YEARS = 14.0          # 2021-22 (base) to 2035-36
DENSITY_CAP = 100.0   # colleges per lakh; beyond observed range (max 66)
N_BOOT = 2000
PTR_REFORM_TARGET = 20.0
GPI_REFORM_TARGET = 1.05

SCENARIOS = {
    "damped": {
        "label": "S1 Damped",
        "density_mult": 0.5,
        "trend_mult": 0.5,
        "ptr_reform": False,
        "gpi_reform": False,
    },
    "recent": {
        "label": "S2 Recent trend",
        "density_mult": 1.0,
        "trend_mult": 1.0,
        "ptr_reform": False,
        "gpi_reform": False,
    },
    "reform": {
        "label": "S3 NEP reform",
        "density_mult": 1.0,
        "trend_mult": 1.0,
        "ptr_reform": True,
        "gpi_reform": True,
    },
}
REFERENCE = "recent"
STATUS_COLORS = {
    "Achieved": "#0072B2",
    "On track": "#009E73",
    "At risk": "#E69F00",
    "Off track": "#9E9E9E",
}


# ---------------------------------------------------------------------------
# parameters and scenario paths
# ---------------------------------------------------------------------------
def load_parameters() -> dict:
    res = json.loads((OUT / "model_results.json").read_text(encoding="utf-8"))
    coef = res["ols"]["coef"]
    return {
        "b_density": float(coef["college_density"]),
        "b_ptr": float(coef["ptr"]),
        "b_gpi": float(coef["gpi"]),
        "trend": float(res["panel"]["annual_common_trend"]),
        "trend_se": float(res["panel"]["annual_common_trend_se"]),
        "resid_sd": float(res["ols"]["resid_sd"]),
        "ols_r2": float(res["ols"]["r2"]),
    }


def scenario_paths(df: pd.DataFrame, key: str) -> dict:
    """End-of-horizon values of the three predictors for one scenario."""
    cfg = SCENARIOS[key]
    annual_density = df["delta_college_density_4y"] / 4.0
    density = df["college_density"] + cfg["density_mult"] * annual_density * YEARS
    density = density.clip(lower=1.0, upper=DENSITY_CAP)

    ptr = df["ptr"].astype(float).copy()
    gpi = df["gpi"].astype(float).copy()
    if cfg["ptr_reform"]:
        ptr = np.minimum(ptr, PTR_REFORM_TARGET)
    if cfg["gpi_reform"]:
        gpi = np.where(gpi < GPI_REFORM_TARGET, GPI_REFORM_TARGET, gpi)

    return {"density": density.to_numpy(), "ptr": np.asarray(ptr, dtype=float),
            "gpi": np.asarray(gpi, dtype=float), "trend_mult": cfg["trend_mult"]}


def project(ger0, base, path, params) -> np.ndarray:
    """Change-based projection (no intercept)."""
    return (ger0
            + params["b_density"] * (path["density"] - base["density"])
            + params["b_ptr"] * (path["ptr"] - base["ptr"])
            + params["b_gpi"] * (path["gpi"] - base["gpi"])
            + params["trend"] * path["trend_mult"] * YEARS)


def classify(ger2021, prob):
    if ger2021 >= TARGET:
        return "Achieved"
    if prob >= 0.5:
        return "On track"
    if prob >= 0.15:
        return "At risk"
    return "Off track"


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def bootstrap(est: pd.DataFrame, base: dict, paths: dict, params: dict) -> dict:
    """Return projected GER draws, shape (N_BOOT, n_states), per scenario."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = sm.add_constant(est[FEATURES], has_constant="add")
    y = est["ger"].to_numpy(dtype=float)
    n = len(est)
    ger0 = y.copy()

    draws = {k: np.empty((N_BOOT, n)) for k in SCENARIOS}
    coef_draws = np.empty((N_BOOT, X.shape[1]))
    trend_draws = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx].to_numpy(dtype=float)
        yb = y[idx]
        coef_b = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        coef_draws[b] = coef_b
        bd, bp, bg = coef_b[1], coef_b[2], coef_b[3]
        trend_b = rng.normal(params["trend"], params["trend_se"])
        trend_draws[b] = trend_b
        eps = rng.normal(0.0, params["resid_sd"], n)

        for key, path in paths.items():
            delta = (bd * (path["density"] - base["density"])
                     + bp * (path["ptr"] - base["ptr"])
                     + bg * (path["gpi"] - base["gpi"])
                     + trend_b * path["trend_mult"] * YEARS)
            draws[key][b] = ger0 + delta + eps

    return {"draws": draws, "coef_draws": coef_draws,
            "trend_draws": trend_draws, "coef_names": list(X.columns)}


def project_from_draws(df: pd.DataFrame, paths: dict, coef_draws: np.ndarray,
                       trend_draws: np.ndarray, resid_sd: float,
                       seed: int = RANDOM_STATE + 1) -> dict:
    """Apply the estimation-sample coefficient/trend draws to another set of units."""
    rng = np.random.default_rng(seed)
    n = len(df)
    base = {"density": df["college_density"].to_numpy(),
            "ptr": df["ptr"].to_numpy(dtype=float),
            "gpi": df["gpi"].to_numpy(dtype=float)}
    ger0 = df["ger"].to_numpy(dtype=float)
    draws = {k: np.empty((len(coef_draws), n)) for k in SCENARIOS}
    for b in range(len(coef_draws)):
        bd, bp, bg = coef_draws[b, 1], coef_draws[b, 2], coef_draws[b, 3]
        eps = rng.normal(0.0, resid_sd, n)
        for key, path in paths.items():
            delta = (bd * (path["density"] - base["density"])
                     + bp * (path["ptr"] - base["ptr"])
                     + bg * (path["gpi"] - base["gpi"])
                     + trend_draws[b] * path["trend_mult"] * YEARS)
            draws[key][b] = ger0 + delta + eps
    return draws


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def state_table(df: pd.DataFrame, paths: dict, boot: dict, params: dict,
                primary: bool) -> pd.DataFrame:
    base = {"density": df["college_density"].to_numpy(),
            "ptr": df["ptr"].to_numpy(dtype=float),
            "gpi": df["gpi"].to_numpy(dtype=float)}
    ger0 = df["ger"].to_numpy(dtype=float)

    out = pd.DataFrame({
        "state": df["state"].to_numpy(),
        "region": df["region"].to_numpy(),
        "in_estimation_sample": 1 if primary else 0,
        "ger_2021": ger0,
        "college_density_2021": base["density"],
        "ptr_2021": base["ptr"],
        "gpi_2021": base["gpi"],
        "colleges_2021": df["colleges_2021"].to_numpy(),
        "avg_enrolment_per_college_2021": df["avg_enrolment_per_college"].to_numpy(),
        "observed_annual_density_change": (df["delta_college_density_4y"] / 4.0).to_numpy(),
    })

    for key, path in paths.items():
        mean = project(ger0, base, path, params)
        d = boot["draws"][key]
        out[f"density_2035_{key}"] = path["density"]
        out[f"ger_2035_{key}"] = mean
        out[f"prob_reach_50_{key}"] = (d >= TARGET).mean(axis=0)
        out[f"ger_2035_{key}_p05"] = np.percentile(d, 5, axis=0)
        out[f"ger_2035_{key}_p95"] = np.percentile(d, 95, axis=0)
        out[f"status_{key}"] = [classify(g, p) for g, p in
                                zip(out["ger_2021"], out[f"prob_reach_50_{key}"])]

    out["status_reference"] = out[f"status_{REFERENCE}"]
    out["status_damped"] = out["status_damped"]
    out["status_reform"] = out["status_reform"]
    return out.sort_values("ger_2021", ascending=False).reset_index(drop=True)


def required_density(tab: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Density needed by 2035-36 under S2 to reach 50%, holding PTR/GPI fixed."""
    ref = "recent"
    trend_gain = params["trend"] * SCENARIOS[ref]["trend_mult"] * YEARS
    # density effect needed after the common trend
    needed = TARGET - (tab["ger_2021"] + trend_gain)
    d_req = tab["college_density_2021"] + needed / params["b_density"]
    d_req = d_req.clip(lower=0.0)

    out = pd.DataFrame({
        "state": tab["state"],
        "region": tab["region"],
        "ger_2021": tab["ger_2021"],
        "college_density_2021": tab["college_density_2021"],
        "density_2035_recent": tab["density_2035_recent"],
        "required_density_model": d_req,
        "density_gap_model": (d_req - tab["density_2035_recent"]).clip(lower=0.0),
        "required_annual_density_change": (d_req - tab["college_density_2021"]) / YEARS,
        "observed_annual_density_change": tab["observed_annual_density_change"],
        "implied_population_18_23_lakh": np.nan,
        "additional_colleges_model": np.nan,
        "reachable_by_density_alone": d_req <= DENSITY_CAP,
    })
    # density is colleges per lakh of the 18-23 population
    out["implied_population_18_23_lakh"] = (tab["colleges_2021"]
                                            / tab["college_density_2021"])
    out["additional_colleges_model"] = (out["density_gap_model"]
                                         * out["implied_population_18_23_lakh"]).round(0)
    return out.sort_values("density_gap_model", ascending=False).reset_index(drop=True)


def appendix_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Per-state appendix: observed GER, LOOCV prediction and 2035 projection."""
    diag = pd.read_csv(TAB / "table05_diagnostics.csv")
    keep = diag[["state", "loo_pred", "loo_error", "cooks_d"]]
    out = tab[["state", "ger_2021", "college_density_2021", "density_2035_recent",
               "ger_2035_recent", "ger_2035_recent_p05", "ger_2035_recent_p95",
               "prob_reach_50_recent", "status_reference"]].merge(
                   keep, on="state", how="left")
    out = out.rename(columns={
        "ger_2021": "ger_observed_2021",
        "density_2035_recent": "density_2035_s2",
        "ger_2035_recent": "ger_projected_2035_s2",
        "ger_2035_recent_p05": "ger_2035_p05",
        "ger_2035_recent_p95": "ger_2035_p95",
        "prob_reach_50_recent": "prob_reach_50_s2",
        "status_reference": "status_s2",
    })
    out = out[["state", "ger_observed_2021", "loo_pred", "loo_error", "cooks_d",
               "college_density_2021", "density_2035_s2", "ger_projected_2035_s2",
               "ger_2035_p05", "ger_2035_p95", "prob_reach_50_s2", "status_s2"]]
    out = out.sort_values("ger_observed_2021", ascending=False).reset_index(drop=True)
    out.round(3).to_csv(TAB / "table11_state_appendix.csv", index=False)
    return out


def scenario_summary(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, cfg in SCENARIOS.items():
        vals = tab[f"ger_2035_{key}"]
        probs = tab[f"prob_reach_50_{key}"]
        status = tab[f"status_{key}"]
        rows.append({
            "scenario": cfg["label"],
            "key": key,
            "n_states": len(tab),
            "n_point_above_50": int((vals >= TARGET).sum()),
            "n_prob_above_50": int((probs >= 0.5).sum()),
            "mean_projected_ger": float(vals.mean()),
            "median_projected_ger": float(vals.median()),
            "min_projected_ger": float(vals.min()),
            "max_projected_ger": float(vals.max()),
            "n_achieved": int((status == "Achieved").sum()),
            "n_on_track": int((status == "On track").sum()),
            "n_at_risk": int((status == "At risk").sum()),
            "n_off_track": int((status == "Off track").sum()),
            "states_point_above_50": "; ".join(tab.loc[vals >= TARGET, "state"].tolist()),
            "states_prob_above_50": "; ".join(tab.loc[probs >= 0.5, "state"].tolist()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_dumbbell(tab: pd.DataFrame) -> None:
    d = tab.sort_values("ger_2021", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 8.0))
    y = np.arange(len(d))
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.plot([r["ger_2021"], r[f"ger_2035_{REFERENCE}"]], [yy, yy],
                color="#B0B0B0", linewidth=1.1, zorder=1)
        ax.plot([r[f"ger_2035_{REFERENCE}_p05"], r[f"ger_2035_{REFERENCE}_p95"]],
                [yy, yy], color="#D9D9D9", linewidth=3.2, alpha=0.55, zorder=0)
    ax.scatter(d["ger_2021"], y, s=30, c="#0072B2", edgecolor="white",
               linewidth=0.4, zorder=3, label="2021-22")
    ax.scatter(d[f"ger_2035_{REFERENCE}"], y, s=38,
               c=[STATUS_COLORS[s] for s in d["status_reference"]],
               edgecolor="white", linewidth=0.5, zorder=4,
               label="2035-36, reference (S2)")
    ax.axvline(TARGET, color="#B2182B", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(TARGET + 0.8, len(d) - 0.2, "50% target", color="#B2182B",
            fontsize=7.5, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(d["state"], fontsize=7.2)
    ax.set_xlabel("GER, higher education (%)")
    ax.set_xlim(0, max(80, d[f"ger_2035_{REFERENCE}_p95"].max() + 4))
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("GER in 2021-22 and projected 2035-36 under recent trends\n"
                 "(bar = 5th–95th percentile of the bootstrap projection)")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#0072B2",
                      markeredgecolor="white", markersize=6, label="2021-22"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="#009E73",
                      markeredgecolor="white", markersize=7, label="Projected 2035-36"),
               Line2D([0], [0], color="#B2182B", linestyle="--", label="50% NEP target")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_projection_dumbbell.png")
    plt.close(fig)


def fig_probability(tab: pd.DataFrame) -> None:
    d = tab.sort_values(f"prob_reach_50_{REFERENCE}", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, 7.6))
    y = np.arange(len(d))
    ax.axvline(0.5, color="#B2182B", linestyle="--", linewidth=1.1)
    ax.axvline(0.15, color="#999999", linestyle=":", linewidth=0.9)
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.plot([r["prob_reach_50_damped"], r["prob_reach_50_reform"]], [yy, yy],
                color="#D0D0D0", linewidth=1.2, zorder=1)
    for key, color, marker, label in [
        ("damped", "#9E9E9E", "o", "S1 damped"),
        ("recent", "#0072B2", "o", "S2 recent trend"),
        ("reform", "#009E73", "^", "S3 NEP reform"),
    ]:
        ax.scatter(d[f"prob_reach_50_{key}"], y, s=24 if key != "recent" else 34,
                   c=color, marker=marker, edgecolor="white", linewidth=0.4,
                   zorder=3, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(d["state"], fontsize=7.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Probability of reaching 50% GER by 2035-36")
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Probability of reaching the 50% NEP target\n"
                 "(bootstrap with coefficient, trend and residual uncertainty)")
    ax.legend(fontsize=7, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "fig13_probability_reaching_50.png")
    plt.close(fig)


def fig_scenarios(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    data = [tab["ger_2021"].to_numpy()] + [tab[f"ger_2035_{k}"].to_numpy()
                                           for k in SCENARIOS]
    labels = ["2021-22", "S1\ndamped", "S2\nrecent", "S3\nreform"]
    colors = ["#0072B2", "#9E9E9E", "#0072B2", "#009E73"]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops={"color": "#333333", "linewidth": 1.2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.22)
        patch.set_edgecolor(c)
    rng = np.random.default_rng(RANDOM_STATE)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=12,
                   c=colors[i - 1], alpha=0.75, edgecolor="white", linewidth=0.3,
                   zorder=3)
    ax.axhline(TARGET, color="#B2182B", linestyle="--", linewidth=1.1)
    ax.text(0.55, TARGET + 1.2, "50% target", color="#B2182B", fontsize=7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("GER (%)")
    ax.set_title("(a) Distribution of state projections")

    ax = axes[1]
    x = np.arange(len(SCENARIOS))
    n_point = [int((tab[f"ger_2035_{k}"] >= TARGET).sum()) for k in SCENARIOS]
    n_prob = [int((tab[f"prob_reach_50_{k}"] >= 0.5).sum()) for k in SCENARIOS]
    w = 0.36
    ax.bar(x - w / 2, n_point, w, color="#0072B2", label="Point estimate ≥ 50%")
    ax.bar(x + w / 2, n_prob, w, color="#E69F00", label="P(≥50%) ≥ 0.5")
    for xx, a, b in zip(x, n_point, n_prob):
        ax.text(xx - w / 2, a + 0.2, str(a), ha="center", fontsize=7)
        ax.text(xx + w / 2, b + 0.2, str(b), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["S1\ndamped", "S2\nrecent", "S3\nreform"], fontsize=7.5)
    ax.set_ylabel("Number of states (of 28)")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("(b) States reaching 50%")
    fig.suptitle("Projected GER in 2035-36 under three scenarios", fontweight="bold",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig14_scenario_distributions.png")
    plt.close(fig)


def fig_required_density(req: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    sc = ax.scatter(req["college_density_2021"], req["required_density_model"],
                    s=np.clip(req["additional_colleges_model"] / 120, 16, 180),
                    c=req["density_gap_model"], cmap="OrRd", edgecolor="#555555",
                    linewidth=0.5, zorder=3)
    lim = max(req["college_density_2021"].max(), req["required_density_model"].max()) + 5
    ax.plot([0, lim], [0, lim], color="#555555", linestyle="--", linewidth=1,
            label="required = current")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("College density in 2021-22")
    ax.set_ylabel("Density required by 2035-36 (S2)")
    ax.set_title("(a) Model-implied density gap")
    label_offsets = {
        "Nagaland": (-44, 7), "Chhattisgarh": (5, 8), "Assam": (5, 5),
        "Gujarat": (5, 6), "Uttar Pradesh": (5, -12), "Bihar": (5, 6),
        "Jharkhand": (-20, -26), "Tripura": (10, -18),
    }
    for _, r in req.head(8).iterrows():
        arrow = (dict(arrowstyle="-", lw=0.5, color="#666666", shrinkA=0, shrinkB=2)
                 if r["state"] == "Jharkhand" else None)
        ax.annotate(r["state"], (r["college_density_2021"], r["required_density_model"]),
                    textcoords="offset points",
                    xytext=label_offsets.get(r["state"], (5, 4)),
                    arrowprops=arrow, fontsize=6)
    ax.legend(fontsize=7)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Gap (colleges per lakh)")

    ax = axes[1]
    d = req[req["additional_colleges_model"] > 0].nlargest(12, "additional_colleges_model").iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y, d["additional_colleges_model"] / 1000, color="#D55E00", alpha=0.85, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(d["state"], fontsize=7)
    ax.set_xlabel("Additional colleges needed (thousands)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Indicative college requirement")
    for yy, v in zip(y, d["additional_colleges_model"] / 1000):
        ax.text(v + 0.15, yy, f"{v:.1f}", va="center", fontsize=6.5)
    fig.suptitle("College density needed to reach 50% GER by 2035-36\n"
                 "(reference scenario; model-implied, 18-23 population base)", fontweight="bold",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig15_required_density.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    plotting_style()
    params = load_parameters()
    cs = load_cross_section()

    est = cs[(cs["is_ut"] == 0) & cs[FEATURES].notna().all(axis=1)
             & cs["delta_college_density_4y"].notna()].copy().reset_index(drop=True)
    est["colleges_2021"] = est["colleges"]
    est["observed_annual_density_change"] = est["delta_college_density_4y"] / 4.0

    base = {"density": est["college_density"].to_numpy(),
            "ptr": est["ptr"].to_numpy(dtype=float),
            "gpi": est["gpi"].to_numpy(dtype=float)}
    paths = {k: scenario_paths(est, k) for k in SCENARIOS}

    boot = bootstrap(est, base, paths, params)
    tab = state_table(est, paths, boot, params, primary=True)
    req = required_density(tab, params)
    summ = scenario_summary(tab)
    app = appendix_table(tab)

    # population-weighted aggregates (weights = implied 18-23 population)
    w = req.set_index("state")["implied_population_18_23_lakh"]
    tw = tab.set_index("state").join(w)
    weighted = {
        "total_population_18_23_lakh": float(tw["implied_population_18_23_lakh"].sum()),
        "ger_2021": float(np.average(tw["ger_2021"], weights=tw["implied_population_18_23_lakh"])),
        "ger_2035_damped": float(np.average(tw["ger_2035_damped"], weights=tw["implied_population_18_23_lakh"])),
        "ger_2035_recent": float(np.average(tw["ger_2035_recent"], weights=tw["implied_population_18_23_lakh"])),
        "ger_2035_reform": float(np.average(tw["ger_2035_reform"], weights=tw["implied_population_18_23_lakh"])),
        "share_population_below_50_recent": float(
            tw.loc[tw["ger_2035_recent"] < TARGET, "implied_population_18_23_lakh"].sum()
            / tw["implied_population_18_23_lakh"].sum()),
    }

    # UT appendix: model-based, outside the estimation sample
    ut = cs[(cs["is_ut"] == 1) & cs[FEATURES].notna().all(axis=1)
            & cs["delta_college_density_4y"].notna()].copy().reset_index(drop=True)
    ut["colleges_2021"] = ut["colleges"]
    ut_paths = {k: scenario_paths(ut, k) for k in SCENARIOS}
    ut_boot = {"draws": project_from_draws(ut, ut_paths, boot["coef_draws"],
                                           boot["trend_draws"], params["resid_sd"])}
    ut_tab = state_table(ut, ut_paths, ut_boot, params, primary=False) if len(ut) else pd.DataFrame()
    if len(ut_tab):
        ut_tab["note"] = "Outside estimation sample; indicative only"

    # save
    tab.round(3).to_csv(TAB / "table06_projections_2035.csv", index=False)
    if len(ut_tab):
        ut_tab.round(3).to_csv(TAB / "table06b_projections_uts.csv", index=False)
    req.round(3).to_csv(TAB / "table08_required_density.csv", index=False)
    summ.round(3).to_csv(TAB / "table09_scenario_summary.csv", index=False)
    results = {
        "parameters": params,
        "horizon_years": YEARS,
        "density_cap": DENSITY_CAP,
        "n_bootstrap": N_BOOT,
        "scenarios": {k: v["label"] for k, v in SCENARIOS.items()},
        "reference_scenario": REFERENCE,
        "summary": summ.to_dict(orient="records"),
        "weighted": weighted,
        "states_on_track_reference": tab.loc[tab["status_reference"] == "On track",
                                            "state"].tolist(),
        "states_on_track_damped": tab.loc[tab["status_damped"] == "On track",
                                          "state"].tolist(),
        "states_reform": tab.loc[tab["status_reform"] == "On track",
                                 "state"].tolist(),
        "required_density_top_gaps": req.head(10)[
            ["state", "required_density_model", "density_gap_model",
             "additional_colleges_model"]].to_dict(orient="records"),
    }
    (OUT / "projection_results.json").write_text(json.dumps(results, indent=2,
                                                            default=float),
                                                 encoding="utf-8")

    fig_dumbbell(tab)
    fig_probability(tab)
    fig_scenarios(tab)
    fig_required_density(req)

    print("Projection parameters:", json.dumps(params, indent=2))
    print("\nScenario summary:")
    print(summ[["scenario", "n_point_above_50", "n_prob_above_50",
                "mean_projected_ger", "median_projected_ger"]].round(2).to_string(index=False))
    print("\nStates projected above 50% (point estimate):")
    for key in SCENARIOS:
        names = tab.loc[tab[f"ger_2035_{key}"] >= TARGET, "state"].tolist()
        print(f"  {SCENARIOS[key]['label']}: {len(names)} -> {', '.join(names)}")
    print("\nStatus under the reference scenario:")
    print(tab["status_reference"].value_counts().to_string())
    print("\nPopulation-weighted GER (implied 18-23 population):")
    print(json.dumps({k: round(v, 2) for k, v in weighted.items()}, indent=2))
    print("\nLargest density gaps:")
    print(req.head(8)[["state", "college_density_2021", "required_density_model",
                       "density_gap_model", "additional_colleges_model"]].round(2).to_string(index=False))
    print("\nSaved projection tables, JSON and figures 12-15.")


if __name__ == "__main__":
    main()
