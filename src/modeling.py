"""
Model comparison, inference and residual diagnostics.

Primary question: how much of the cross-state variation in GER (2021-22) is
associated with college density, pupil-teacher ratio and gender parity?

Estimation sample: 28 States (Union Territories are analysed as a robustness
check because their tertiary catchments cross state boundaries and three UTs
are high-leverage outliers).

Outputs
-------
outputs/tables/table03_model_comparison.csv
outputs/tables/table04_coefficients.csv
outputs/tables/table05_diagnostics.csv
outputs/tables/table07_sensitivity.csv
outputs/tables/table10_hyperparameter_search.csv
outputs/model_results.json
outputs/figures/fig05..fig11
"""

from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, LeaveOneOut,
                                     cross_val_predict)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (FEATURES, FEATURE_LABELS, FIG, RANDOM_STATE, SHORT_LABELS,
                    TAB, TARGET, load_cross_section, load_panel, plotting_style)

warnings.filterwarnings("ignore")
ALPHAS = np.logspace(-3, 3, 61)
N_REPEATS = 20


# ---------------------------------------------------------------------------
# model definitions
# ---------------------------------------------------------------------------
def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def poly_ridge(degree: int = 2):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=ALPHAS),
    )


KNN_GRID = {
    "kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 15],
    "kneighborsregressor__weights": ["uniform", "distance"],
}
TREE_GRID = {
    "max_depth": [2, 3, 4],
    "min_samples_leaf": [3, 5],
}


def knn_tuned():
    """k-NN with k and the weighting scheme tuned inside each training fold."""
    return GridSearchCV(
        make_pipeline(StandardScaler(), KNeighborsRegressor()),
        KNN_GRID, cv=LeaveOneOut(), scoring="neg_root_mean_squared_error",
    )


def tree_tuned():
    """Shallow regression tree with depth and leaf size tuned inside each fold."""
    return GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        TREE_GRID, cv=LeaveOneOut(), scoring="neg_root_mean_squared_error",
    )


MODELS = {
    "Baseline (mean)": (DummyRegressor(strategy="mean"), FEATURES),
    "OLS (3 indicators)": (LinearRegression(), FEATURES),
    "Ridge (3 indicators)": (ridge(), FEATURES),
    "Polynomial deg 2 + ridge": (poly_ridge(), FEATURES),
    "k-NN (tuned, 3 indicators)": (knn_tuned(), FEATURES),
    "Regression tree (tuned)": (tree_tuned(), FEATURES),
    "OLS (+ college size)": (LinearRegression(), FEATURES + ["avg_enrolment_per_college"]),
    "Ridge (+ college size)": (ridge(), FEATURES + ["avg_enrolment_per_college"]),
}
COLORS = {
    "Baseline (mean)": "#9E9E9E",
    "OLS (3 indicators)": "#0072B2",
    "Ridge (3 indicators)": "#B2182B",
    "Polynomial deg 2 + ridge": "#CC79A7",
    "k-NN (tuned, 3 indicators)": "#56B4E9",
    "Regression tree (tuned)": "#8C564B",
    "OLS (+ college size)": "#009E73",
    "Ridge (+ college size)": "#E69F00",
}


def metrics(y, pred) -> dict:
    return {
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": mean_absolute_error(y, pred),
    }


def repeated_cv(model, X, y, repeats=N_REPEATS) -> dict:
    rows = []
    for r in range(repeats):
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + r)
        pred = cross_val_predict(model, X, y, cv=cv)
        rows.append(metrics(y, pred))
    df = pd.DataFrame(rows)
    return {f"{k}_{s}": v for k in ("r2", "rmse", "mae")
            for s, v in (("mean", df[k].mean()), ("sd", df[k].std()))}


def in_sample_stats(model, X, y) -> dict:
    model.fit(X, y)
    pred = model.predict(X)
    out = {"insample_r2": r2_score(y, pred),
           "insample_rmse": float(np.sqrt(mean_squared_error(y, pred)))}
    p = X.shape[1]
    n = len(y)
    out["adj_r2"] = 1 - (1 - out["insample_r2"]) * (n - 1) / (n - p - 1)
    return out


def compare_models(cs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    est = cs[cs["is_ut"] == 0].dropna(subset=FEATURES + ["avg_enrolment_per_college"]).copy()
    y = est["ger"].values
    rows = []
    loo_preds = {}
    for name, (model, feats) in MODELS.items():
        X = est[feats].values
        loo = cross_val_predict(model, X, y, cv=LeaveOneOut())
        loo_preds[name] = loo
        m = metrics(y, loo)
        cv = repeated_cv(model, X, y)
        ins = in_sample_stats(model, X, y)
        rows.append({"model": name, "n": len(y), "k": len(feats),
                     "loo_r2": m["r2"], "loo_rmse": m["rmse"], "loo_mae": m["mae"],
                     **cv, **ins})
    tab = pd.DataFrame(rows)
    # AIC / BIC for the two OLS specifications
    tab["aic"] = np.nan
    tab["bic"] = np.nan
    for name in ["OLS (3 indicators)", "OLS (+ college size)"]:
        _, feats = MODELS[name]
        fit = sm.OLS(y, sm.add_constant(est[feats])).fit()
        tab.loc[tab.model == name, "aic"] = fit.aic
        tab.loc[tab.model == name, "bic"] = fit.bic
    tab = tab.sort_values("loo_rmse").reset_index(drop=True)
    tab.to_csv(TAB / "table03_model_comparison.csv", index=False)
    return tab, loo_preds


def tuning_search(est: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """Document the hyper-parameter grids and the values selected.

    The metrics in Table 3 are produced with *nested* tuning: the search below
    runs inside every outer leave-one-out fold.  This table records the search
    itself, the values selected on the full estimation sample, the internal
    selection score, and the nested LOOCV score that is actually reported.
    """
    X = est[FEATURES].values
    y = est["ger"].values
    nested = dict(zip(comp["model"], comp["loo_rmse"]))
    rows = []

    # Ridge: RidgeCV selects alpha by efficient leave-one-out generalised CV.
    r = ridge().fit(X, y)
    rows.append({
        "model": "Ridge (3 indicators)",
        "grid_searched": f"alpha in {len(ALPHAS)} log-spaced values, "
                         f"{ALPHAS.min():.0e} to {ALPHAS.max():.0e}",
        "selection_rule": "RidgeCV generalised cross-validation",
        "selected": f"alpha={r.named_steps['ridgecv'].alpha_:.3f}",
        "internal_cv_rmse": float(metrics(y, cross_val_predict(r, X, y, cv=LeaveOneOut()))["rmse"]),
        "nested_loo_rmse": nested["Ridge (3 indicators)"],
        "notes": "Scaling fitted inside the pipeline on each training fold.",
    })

    # Polynomial: degree chosen by LOOCV; alpha by RidgeCV inside each fit.
    deg_rows = []
    for degree in (1, 2, 3):
        pred = cross_val_predict(poly_ridge(degree), X, y, cv=LeaveOneOut())
        deg_rows.append((degree, float(metrics(y, pred)["rmse"])))
    best_degree, best_rmse = min(deg_rows, key=lambda t: t[1])
    poly_full = poly_ridge(best_degree).fit(X, y)
    detail = "; ".join(f"degree {d} = {v:.3f}" for d, v in deg_rows)
    rows.append({
        "model": "Polynomial + ridge",
        "grid_searched": "degree in {1, 2, 3}; alpha in 61 log-spaced values",
        "selection_rule": "LOOCV RMSE for degree; RidgeCV for alpha",
        "selected": f"degree={best_degree}, "
                    f"alpha={poly_full.named_steps['ridgecv'].alpha_:.3f}",
        "internal_cv_rmse": best_rmse,
        "nested_loo_rmse": nested["Polynomial deg 2 + ridge"],
        "notes": detail,
    })

    # k-NN and tree: grid search with LOOCV inside each training fold.
    for label, search, grid in [
        ("k-NN (tuned, 3 indicators)", knn_tuned(), KNN_GRID),
        ("Regression tree (tuned)", tree_tuned(), TREE_GRID),
    ]:
        search.fit(X, y)
        params = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in
                           sorted(search.best_params_.items()))
        rows.append({
            "model": label,
            "grid_searched": "; ".join(f"{k.split('__')[-1]} in {v}" for k, v in grid.items()),
            "selection_rule": "GridSearchCV with LOOCV, RMSE",
            "selected": params,
            "internal_cv_rmse": float(-search.best_score_),
            "nested_loo_rmse": nested[label],
            "notes": "Search repeated inside every outer fold; nested score is the "
                     "honest one because selection cannot see the held-out state.",
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "table10_hyperparameter_search.csv", index=False)
    return tab


def fit_inference(est: pd.DataFrame):
    """OLS with HC3 errors, bootstrap CIs and VIF for the 3-indicator model."""
    X = sm.add_constant(est[FEATURES])
    y = est["ger"]
    fit = sm.OLS(y, X).fit(cov_type="HC3")

    rng = np.random.default_rng(RANDOM_STATE)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(est), len(est))
        b = sm.OLS(y.iloc[idx], sm.add_constant(est[FEATURES].iloc[idx])).fit()
        boot.append(b.params.values)
    boot = pd.DataFrame(boot, columns=X.columns)

    # standardized coefficients (comparable effect sizes)
    z = (est[FEATURES] - est[FEATURES].mean()) / est[FEATURES].std()
    zy = (y - y.mean()) / y.std()
    zfit = sm.OLS(zy, sm.add_constant(z)).fit()
    zboot = []
    for _ in range(2000):
        idx = rng.integers(0, len(est), len(est))
        zb = sm.OLS(zy.iloc[idx], sm.add_constant(z.iloc[idx])).fit()
        zboot.append(zb.params.values)
    zboot = pd.DataFrame(zboot, columns=zfit.params.index)

    rows = []
    for c in FEATURES:
        rows.append({
            "term": SHORT_LABELS[c],
            "coef_raw": fit.params[c],
            "se_hc3": fit.bse[c],
            "p_value": fit.pvalues[c],
            "ci_low": boot[c].quantile(0.025),
            "ci_high": boot[c].quantile(0.975),
            "coef_std": zfit.params[c],
            "std_ci_low": zboot[c].quantile(0.025),
            "std_ci_high": zboot[c].quantile(0.975),
            "vif": variance_inflation_factor(X.values, X.columns.get_loc(c)),
        })
    coef = pd.DataFrame(rows)
    coef.to_csv(TAB / "table04_coefficients.csv", index=False)
    return fit, coef, boot


def diagnostics(est: pd.DataFrame, fit, loo_pred=None) -> tuple[pd.DataFrame, dict]:
    infl = fit.get_influence()
    res = pd.DataFrame({
        "state": est["state"].values,
        "ger": est["ger"].values,
        "fitted": fit.fittedvalues.values,
        "residual": fit.resid.values,
        "std_residual": infl.resid_studentized_internal,
        "leverage": infl.hat_matrix_diag,
        "cooks_d": infl.cooks_distance[0],
        "dffits": infl.dffits[0],
    })
    if loo_pred is not None:
        res["loo_pred"] = np.asarray(loo_pred, dtype=float)
        res["loo_error"] = res["ger"] - res["loo_pred"]
        res["loo_abs_error"] = res["loo_error"].abs()
    res["abs_std_residual"] = res["std_residual"].abs()
    res = res.sort_values("cooks_d", ascending=False).reset_index(drop=True)
    res.to_csv(TAB / "table05_diagnostics.csv", index=False)

    bp = het_breuschpagan(fit.resid, fit.model.exog)
    reset = linear_reset(fit, power=2, test_type="fitted")
    shp = stats.shapiro(fit.resid)
    out = {
        "shapiro_W": float(shp.statistic), "shapiro_p": float(shp.pvalue),
        "breusch_pagan_LM": float(bp[0]), "breusch_pagan_p": float(bp[1]),
        "reset_F": float(reset.statistic), "reset_p": float(reset.pvalue),
        "condition_number": float(np.linalg.cond(fit.model.exog)),
        "max_cooks_d": float(res["cooks_d"].max()),
        "cooks_threshold_4_over_n": 4 / len(res),
        "n_influential": int((res["cooks_d"] > 4 / len(res)).sum()),
        "n_outliers_2sd": int((res["abs_std_residual"] > 2).sum()),
        "residual_sd": float(np.sqrt(fit.mse_resid)),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
    }
    return res, out


def sensitivity(cs: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """UT-inclusive regressions, panel with year effects, and first differences."""
    rows = []
    allu = cs.dropna(subset=FEATURES).copy()
    for label, feats in [("All 35 units, core indicators", FEATURES),
                         ("All 35 units, core + UT dummy", FEATURES + ["is_ut"])]:
        X = allu[feats].values
        y = allu["ger"].values
        loo = cross_val_predict(ridge(), X, y, cv=LeaveOneOut())
        m = metrics(y, loo)
        rows.append({"specification": label, "n": len(allu),
                     "loo_r2": m["r2"], "loo_rmse": m["rmse"], "loo_mae": m["mae"]})

    p = panel[panel["is_ut"] == 0].dropna(subset=["ger"] + FEATURES).copy()
    p = p.sort_values(["state", "year_num"])
    # pooled with year fixed effects, cluster-robust SEs
    Xy = pd.get_dummies(p["year"], drop_first=True, dtype=float)
    Xp = pd.concat([p[FEATURES].reset_index(drop=True), Xy.reset_index(drop=True)], axis=1)
    Xp = sm.add_constant(Xp)
    fit_y = sm.OLS(p["ger"].reset_index(drop=True), Xp).fit(
        cov_type="cluster", cov_kwds={"groups": p["state"].reset_index(drop=True)})
    year_cols = [c for c in Xp.columns if c in set(p["year"])]
    time_fe = {c: float(fit_y.params[c]) for c in year_cols}
    annual_trend = float(fit_y.params["2021-22"]) / 4.0 if "2021-22" in year_cols else np.nan
    annual_trend_se = float(fit_y.bse["2021-22"]) / 4.0 if "2021-22" in year_cols else np.nan
    rows.append({"specification": "Panel, year FE (2017-22)", "n": len(p),
                 "loo_r2": np.nan, "loo_rmse": np.nan, "loo_mae": np.nan,
                 "note": f"R2={fit_y.rsquared:.3f}; density b={fit_y.params['college_density']:.3f}, "
                         f"PTR b={fit_y.params['ptr']:.3f}, GPI b={fit_y.params['gpi']:.3f}"})

    # Sikkim's 2019-20 GER (66.9) is published but breaks the series by 27 points
    # in one year and 18.8 the next; it is treated as a reporting anomaly and the
    # year-FE model is re-estimated without the State as a robustness check.
    ps = p[p["state"] != "Sikkim"]
    Xys = pd.concat([ps[FEATURES].reset_index(drop=True),
                     pd.get_dummies(ps["year"], drop_first=True, dtype=float).reset_index(drop=True)],
                    axis=1)
    fit_ys = sm.OLS(ps["ger"].reset_index(drop=True), sm.add_constant(Xys)).fit(
        cov_type="cluster", cov_kwds={"groups": ps["state"].reset_index(drop=True)})
    trend_ys = float(fit_ys.params["2021-22"]) / 4.0
    trend_ys_se = float(fit_ys.bse["2021-22"]) / 4.0
    rows.append({"specification": "Panel, year FE excluding Sikkim", "n": len(ps),
                 "loo_r2": np.nan, "loo_rmse": np.nan, "loo_mae": np.nan,
                 "note": f"R2={fit_ys.rsquared:.3f}; density b={fit_ys.params['college_density']:.3f}, "
                         f"PTR b={fit_ys.params['ptr']:.3f}, GPI b={fit_ys.params['gpi']:.3f}; "
                         f"trend={trend_ys:.3f} (SE {trend_ys_se:.3f})"})

    # state + year FE
    Xs = pd.concat([Xp, pd.get_dummies(p["state"], drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
    fit_s = sm.OLS(p["ger"].reset_index(drop=True), Xs).fit(
        cov_type="cluster", cov_kwds={"groups": p["state"].reset_index(drop=True)})
    rows.append({"specification": "Panel, state + year FE", "n": len(p),
                 "loo_r2": np.nan, "loo_rmse": np.nan, "loo_mae": np.nan,
                 "note": f"R2={fit_s.rsquared:.3f}; density b={fit_s.params['college_density']:.3f}, "
                         f"PTR b={fit_s.params['ptr']:.3f}, GPI b={fit_s.params['gpi']:.3f}"})

    # first differences
    d = p.copy()
    d["d_ger"] = d.groupby("state")["ger"].diff()
    for c in FEATURES:
        d[f"d_{c}"] = d.groupby("state")[c].diff()
    d = d.dropna(subset=["d_ger"] + [f"d_{c}" for c in FEATURES])
    Xd = d[[f"d_{c}" for c in FEATURES]]
    fit_d = sm.OLS(d["d_ger"], Xd).fit(cov_type="cluster", cov_kwds={"groups": d["state"]})
    rows.append({"specification": "First differences (within-state)", "n": len(d),
                 "loo_r2": np.nan, "loo_rmse": np.nan, "loo_mae": np.nan,
                 "note": f"R2={fit_d.rsquared:.3f}; d_density b={fit_d.params['d_college_density']:.3f}, "
                         f"d_PTR b={fit_d.params['d_ptr']:.3f}, d_GPI b={fit_d.params['d_gpi']:.3f}"})

    # leave-out influence checks
    est28 = cs[cs["is_ut"] == 0].dropna(subset=FEATURES).copy()
    for drops in (["Karnataka"], ["Karnataka", "Bihar"]):
        sub = est28[~est28["state"].isin(drops)]
        Xo = sm.add_constant(sub[FEATURES])
        fo = sm.OLS(sub["ger"], Xo).fit(cov_type="HC3")
        loo = cross_val_predict(ridge(), sub[FEATURES].values, sub["ger"].values,
                                cv=LeaveOneOut())
        mo = metrics(sub["ger"].values, loo)
        rows.append({
            "specification": "OLS excluding " + " + ".join(drops), "n": len(sub),
            "loo_r2": mo["r2"], "loo_rmse": mo["rmse"], "loo_mae": mo["mae"],
            "note": f"R2={fo.rsquared:.3f}; density b={fo.params['college_density']:.3f} "
                    f"(p={fo.pvalues['college_density']:.3f}), PTR b={fo.params['ptr']:.3f} "
                    f"(p={fo.pvalues['ptr']:.3f}), GPI b={fo.params['gpi']:.3f} "
                    f"(p={fo.pvalues['gpi']:.3f})",
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "table07_sensitivity.csv", index=False)
    extra = {
        "year_fe": time_fe,
        "annual_common_trend": annual_trend,
        "annual_common_trend_se": annual_trend_se,
        "panel_density": float(fit_y.params["college_density"]),
        "panel_ptr": float(fit_y.params["ptr"]),
        "panel_gpi": float(fit_y.params["gpi"]),
        "panel_r2": float(fit_y.rsquared),
        "state_fe_density": float(fit_s.params["college_density"]),
        "state_fe_ptr": float(fit_s.params["ptr"]),
        "state_fe_gpi": float(fit_s.params["gpi"]),
        "fd_density": float(fit_d.params["d_college_density"]),
        "fd_ptr": float(fit_d.params["d_ptr"]),
        "fd_gpi": float(fit_d.params["d_gpi"]),
        "fd_r2": float(fit_d.rsquared),
        "fd_n": int(len(d)),
    }
    return tab, extra


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_model_comparison(tab: pd.DataFrame) -> None:
    d = tab.sort_values("loo_rmse", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))
    y = np.arange(len(d))
    for ax, metric, xlabel in [(axes[0], "rmse", "LOOCV RMSE (GER points)"),
                               (axes[1], "r2", "LOOCV R²")]:
        vals = d[f"loo_{metric}"]
        ax.barh(y, vals, color=[COLORS[m] for m in d.model], alpha=0.9, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(d.model, fontsize=7.2)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        span = max(vals) - min(vals)
        for yy, v in zip(y, vals):
            ax.text(v + (0.02 * span if metric == "r2" else 0.03 * span), yy,
                    f"{v:.2f}", va="center", fontsize=6.8)
        ax.set_xlim(min(0, min(vals)) - 0.05 * span if metric == "r2" else 0,
                    max(vals) + 0.14 * span)
    axes[0].set_title("(a) Error")
    axes[1].set_title("(b) Explanatory power")
    fig.suptitle("Model comparison, 28 states (bars = leave-one-out cross-validation)",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_model_comparison.png")
    plt.close(fig)


def fig_actual_predicted(est: pd.DataFrame, loo_preds: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), sharex=True, sharey=True)
    offsets = {"Karnataka": (4, -8), "Tamil Nadu": (-10, 14), "Nagaland": (4, 3),
               "Assam": (4, -2)}
    for ax, name in zip(axes, ["Ridge (3 indicators)", "Polynomial deg 2 + ridge"]):
        p = loo_preds[name]
        ax.scatter(est["ger"], p, s=30, c="#0072B2", edgecolor="white", linewidth=0.5)
        lo, hi = 8, 52
        ax.plot([lo, hi], [lo, hi], color="#B2182B", linestyle="--", linewidth=1)
        if name == "Ridge (3 indicators)":
            for _, row in est.assign(pred=p).iterrows():
                if row["state"] in offsets:
                    ax.annotate(row["state"], (row["ger"], row["pred"]),
                                textcoords="offset points", xytext=offsets[row["state"]],
                                fontsize=6)
        m = metrics(est["ger"], p)
        ax.set_title(f"{name}\nLOOCV R²={m['r2']:.2f}, RMSE={m['rmse']:.1f}")
        ax.set_xlabel("Observed GER (%)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel("Leave-one-out prediction (%)")
    fig.suptitle("Out-of-sample predictions: linear versus polynomial specification",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_actual_vs_predicted.png")
    plt.close(fig)


def fig_residuals(est: pd.DataFrame, fit) -> None:
    res = fit.resid
    fitted = fit.fittedvalues
    std = res / res.std()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
    ax = axes[0, 0]
    ax.scatter(fitted, res, s=26, c="#0072B2", edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Fitted GER (%)")
    ax.set_ylabel("Residual (percentage points)")
    ax.set_title("(a) Residuals vs fitted")
    ax = axes[0, 1]
    osm, osr = stats.probplot(std, dist="norm", fit=False)
    ax.scatter(osm, osr, s=22, c="#0072B2", edgecolor="white", linewidth=0.4)
    sl, ic, rr = stats.probplot(std, dist="norm", fit=True)[1]
    ax.plot(osm, sl * osm + ic, color="#B2182B", linewidth=1)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardised residuals")
    ax.set_title(f"(b) Normal Q–Q (Shapiro p = {stats.shapiro(res).pvalue:.3f})")
    ax = axes[1, 0]
    ax.scatter(fitted, np.sqrt(np.abs(std)), s=26, c="#0072B2",
               edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Fitted GER (%)")
    ax.set_ylabel("√|standardised residual|")
    ax.set_title("(c) Scale–location")
    ax = axes[1, 1]
    ax.hist(res, bins=9, color="#0072B2", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Residual (percentage points)")
    ax.set_ylabel("Count")
    ax.set_title("(d) Residual distribution")
    fig.suptitle("Residual diagnostics for the OLS 3-indicator model", fontweight="bold",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_residual_diagnostics.png")
    plt.close(fig)


def fig_coefficients(coef: pd.DataFrame) -> None:
    d = coef.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    y = np.arange(len(d))
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.errorbar(d["coef_std"], y,
                xerr=[d["coef_std"] - d["std_ci_low"], d["std_ci_high"] - d["coef_std"]],
                fmt="o", color="#B2182B", ecolor="#B2182B", elinewidth=1.1,
                capsize=3, markersize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["term"])
    ax.set_xlabel("Standardised coefficient (SD of GER per SD of predictor)")
    ax.set_title("Standardised associations with GER, 28 states\n(95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    for yy, v in zip(y, d["coef_std"]):
        ax.text(v + 0.02, yy + 0.16, f"{v:+.2f}", fontsize=7, color="#333333")
    fig.savefig(FIG / "fig08_coefficients.png")
    plt.close(fig)


def fig_partial_effects(est: pd.DataFrame, fit) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    for ax, c in zip(axes, FEATURES):
        others = [f for f in FEATURES if f != c]
        grid = np.linspace(est[c].min(), est[c].max(), 80)
        at = est[others].mean()
        offset = fit.params["const"] + sum(fit.params[o] * at[o] for o in others)
        pred = offset + fit.params[c] * grid
        ax.plot(grid, pred, color="#B2182B", linewidth=1.6)
        # component-plus-residual: residual + b_c*x_c + mean of the other terms
        pr = fit.resid + fit.params[c] * est[c] + offset
        ax.scatter(est[c], pr, s=24, c="#0072B2", alpha=0.75,
                   edgecolor="white", linewidth=0.4)
        ax.set_xlabel(FEATURE_LABELS[c])
        ax.set_title(SHORT_LABELS[c])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("GER (%) / partial residual")
    fig.suptitle("Partial effects at mean values of the other indicators",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig09_partial_effects.png")
    plt.close(fig)


def fig_influence(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    d = res.sort_values("cooks_d")
    ax = axes[0]
    ax.barh(np.arange(len(d)), d["cooks_d"], color="#0072B2", height=0.7)
    ax.axvline(4 / len(d), color="#B2182B", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["state"], fontsize=6.5)
    ax.set_xlabel("Cook's distance")
    ax.set_title("(a) Influence")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.scatter(res["leverage"], res["std_residual"], s=28, c="#0072B2",
               edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(2, color="#B2182B", linestyle=":", linewidth=0.9)
    ax.axhline(-2, color="#B2182B", linestyle=":", linewidth=0.9)
    for _, row in res.iterrows():
        if row["cooks_d"] > 4 / len(res) or abs(row["std_residual"]) > 2:
            ax.annotate(row["state"], (row["leverage"], row["std_residual"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Studentised residual")
    ax.set_title("(b) Leverage and outliers")
    fig.tight_layout()
    fig.savefig(FIG / "fig10_influence.png")
    plt.close(fig)


def fig_poly_instability(est: pd.DataFrame) -> None:
    """Show why the polynomial is not used for projection."""
    ols = LinearRegression().fit(est[FEATURES], est["ger"])
    poly = poly_ridge().fit(est[FEATURES], est["ger"])
    grid = np.linspace(5, 120, 200)
    Xg = pd.DataFrame({"college_density": grid,
                       "ptr": est["ptr"].mean(), "gpi": est["gpi"].mean()})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(est["college_density"], est["ger"], s=30, c="#0072B2",
               edgecolor="white", linewidth=0.5, zorder=3, label="States (2021-22)")
    ax.plot(grid, ols.predict(Xg), color="#B2182B", linewidth=1.7, label="Linear (used)")
    ax.plot(grid, poly.predict(Xg), color="#CC79A7", linewidth=1.7, linestyle="--",
            label="Polynomial deg 2")
    ax.axvline(est["college_density"].max(), color="#555555", linestyle=":", linewidth=1)
    ax.text(est["college_density"].max() + 1, 15, "observed maximum", fontsize=7, color="#555555")
    ax.axhline(TARGET, color="#B2182B", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Colleges per lakh population (18–23)")
    ax.set_ylabel("Predicted GER (%)")
    ax.set_title("Out-of-range behaviour of the polynomial specification\n(PTR and GPI held at sample means)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(FIG / "fig11_polynomial_instability.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    plotting_style()
    cs = load_cross_section()
    panel = load_panel()
    est = cs[cs["is_ut"] == 0].dropna(subset=FEATURES).copy().reset_index(drop=True)

    comp, loo_preds = compare_models(cs)
    print("Model comparison (28 states):")
    print(comp[["model", "loo_r2", "loo_rmse", "loo_mae", "rmse_mean", "rmse_sd"]]
          .round(3).to_string(index=False))

    tune = tuning_search(est, comp)
    print("\nHyper-parameter search:")
    print(tune[["model", "selected", "internal_cv_rmse", "nested_loo_rmse"]]
          .round(3).to_string(index=False))

    fit, coef, boot = fit_inference(est)
    print("\nOLS coefficients (HC3):")
    print(coef[["term", "coef_raw", "se_hc3", "p_value", "coef_std", "vif"]].round(3).to_string(index=False))

    res, diag = diagnostics(est, fit, loo_preds["OLS (3 indicators)"])
    print("\nDiagnostics:", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                        for k, v in diag.items()}, indent=2))
    print("\nMost influential states:")
    print(res.head(6)[["state", "residual", "std_residual", "cooks_d", "leverage"]].round(2).to_string(index=False))

    sens, extra = sensitivity(cs, panel)
    print("\nSensitivity:")
    print(sens[["specification", "n", "loo_r2", "note"]].to_string(index=False))
    print("\nAnnual common trend from panel:", round(extra["annual_common_trend"], 3), "points/year")

    # ridge final model: alpha and coefficients on the raw scale
    r = ridge().fit(est[FEATURES], est["ger"])
    rcv = r.named_steps["ridgecv"]
    scaler = r.named_steps["standardscaler"]
    ridge_raw = pd.Series(rcv.coef_ / scaler.scale_, index=FEATURES)
    ridge_intercept = float(rcv.intercept_ - (scaler.mean_ / scaler.scale_ * rcv.coef_).sum())
    print("\nRidge alpha:", rcv.alpha_)
    print("Ridge coefficients (raw scale):", ridge_raw.round(3).to_dict(), "intercept", round(ridge_intercept, 2))

    fig_model_comparison(comp)
    fig_actual_predicted(est, loo_preds)
    fig_residuals(est, fit)
    fig_coefficients(coef)
    fig_partial_effects(est, fit)
    fig_influence(res)
    fig_poly_instability(est)

    results = {
        "n_estimation": int(len(est)),
        "n_all_units": int(len(cs)),
        "ols": {
            "intercept": float(fit.params["const"]),
            "coef": {c: float(fit.params[c]) for c in FEATURES},
            "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "resid_sd": float(np.sqrt(fit.mse_resid)),
        },
        "ridge": {
            "alpha": float(rcv.alpha_),
            "intercept": ridge_intercept,
            "coef": {c: float(ridge_raw[c]) for c in FEATURES},
        },
        "diagnostics": diag,
        "panel": extra,
        "model_comparison": comp.to_dict(orient="records"),
        "tuning": tune.to_dict(orient="records"),
        "bootstrap_coef": {
            c: {"low": float(boot[c].quantile(0.025)),
                "high": float(boot[c].quantile(0.975))} for c in FEATURES
        },
    }
    (FIG.parent / "model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved model_results.json, table10 and figures 05-11.")


if __name__ == "__main__":
    main()
