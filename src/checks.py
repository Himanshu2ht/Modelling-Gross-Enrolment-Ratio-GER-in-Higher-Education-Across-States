"""
Supplementary checks requested by the reviewer.

1. Skewness and log-transform test (Table 12).  The paper uses college density,
   PTR and GPI on their natural scale.  This script records the skewness of each
   predictor, refits OLS with log transforms, and compares in-sample fit,
   leave-one-out error, AIC and residual diagnostics.  The raw scale is kept only
   if the log specifications do not improve out-of-sample prediction.
2. Missingness table (Table 13).  Documents every unit-year that is absent from
   the panel and the treatment applied, so that structural non-reports are not
   mistaken for random gaps.

Run:  python src/checks.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import FEATURES, FIG, TAB, load_cross_section, load_panel

TOL = 1e-12


def loocv_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Leave-one-out predictions from OLS, refitting on each training set."""
    n = len(y)
    pred = np.empty(n, dtype=float)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        fit = sm.OLS(y[train], sm.add_constant(X[train])).fit()
        x_new = np.concatenate([[1.0], X[i]])
        pred[i] = float(fit.predict(x_new[None, :])[0])
    return pred, {
        "loo_r2": float(r2_score(y, pred)),
        "loo_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "loo_mae": float(mean_absolute_error(y, pred)),
    }


def fit_spec(est: pd.DataFrame, name: str, cols: list[str]) -> dict:
    y = est["ger"].to_numpy(dtype=float)
    X = est[cols].to_numpy(dtype=float)
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    pred, loo = loocv_ols(y, X)
    resid = y - pred
    bp = sm.stats.diagnostic.het_breuschpagan(fit.resid, fit.model.exog)
    sw = stats.shapiro(fit.resid)
    # Coefficient on the density term (first predictor in every specification).
    return {
        "specification": name,
        "predictors": " + ".join(cols),
        "n": int(len(y)),
        "r2_in_sample": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        **loo,
        "resid_skew": float(stats.skew(resid)),
        "shapiro_p": float(sw.pvalue),
        "breusch_pagan_p": float(bp[1]),
        "density_coef": float(fit.params[1]),
        "density_se": float(fit.bse[1]),
        "density_p": float(fit.pvalues[1]),
        "max_abs_predictor_skew": float(
            np.max(np.abs([stats.skew(est[c]) for c in cols]))
        ),
    }


def log_transform_table(cs: pd.DataFrame) -> pd.DataFrame:
    est = cs[(cs["is_ut"] == 0) & cs[FEATURES].notna().all(axis=1)].copy()
    est["log_density"] = np.log(est["college_density"])
    est["log_ptr"] = np.log(est["ptr"])
    est["log_gpi"] = np.log(est["gpi"])

    specs = [
        ("Raw scale (reported)", ["college_density", "ptr", "gpi"]),
        ("Log density", ["log_density", "ptr", "gpi"]),
        ("Log density and log PTR", ["log_density", "log_ptr", "gpi"]),
        ("Log all three predictors", ["log_density", "log_ptr", "log_gpi"]),
    ]
    rows = [fit_spec(est, name, cols) for name, cols in specs]
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "table12_log_transform.csv", index=False)

    skew = pd.DataFrame({
        "variable": ["college_density", "ptr", "gpi", "ger"],
        "raw_skew": [float(stats.skew(est[c])) for c in
                     ["college_density", "ptr", "gpi", "ger"]],
        "log_skew": [float(stats.skew(np.log(est[c]))) for c in
                     ["college_density", "ptr", "gpi", "ger"]],
    })
    skew.to_csv(TAB / "table12b_skewness.csv", index=False)

    payload = {
        "note": "LOOCV refits OLS on the 27 training States at every step.",
        "specifications": tab.to_dict(orient="records"),
        "skewness": skew.to_dict(orient="records"),
    }
    (FIG.parent / "log_transform_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return tab


def missingness_table(cs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    core = ["ger", "college_density", "ptr", "gpi"]
    miss = panel[panel[core + ["avg_enrolment_per_college"]].isna().any(axis=1)].copy()
    rows = []
    for (state,), grp in miss.groupby(["state"]):
        years = ", ".join(sorted(grp["year"].astype(str).unique()))
        cols = [c for c in core + ["avg_enrolment_per_college"]
                if grp[c].isna().any()]
        if state == "Ladakh":
            reason = ("Ladakh was constituted as a Union Territory on 31 October "
                      "2019; AISHE has no separate pre-2019 rows.")
            treatment = ("Two unit-years excluded from the panel; no imputation "
                         "because the gap is structural.")
        elif state == "Lakshadweep":
            reason = ("AISHE does not report college counts (or density and "
                      "average enrolment per college) for Lakshadweep.")
            treatment = ("Excluded from the multivariate models and from the "
                         "density-based projection; GER, PTR and GPI are retained "
                         "for description only.")
        else:
            reason = "Not reported in the source table."
            treatment = "Excluded from the affected specification."
        rows.append({
            "unit": state,
            "years": years,
            "variables_missing": ", ".join(cols),
            "reason": reason,
            "treatment": treatment,
        })

    est = cs[(cs["is_ut"] == 0) & cs[FEATURES].notna().all(axis=1)]
    rows.append({
        "unit": "28-State estimation sample",
        "years": "2021-22",
        "variables_missing": "none",
        "reason": "Complete cases on GER, college density, PTR and GPI.",
        "treatment": f"n = {len(est)}; no imputation.",
    })
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "table13_missingness.csv", index=False)
    return tab


def main() -> None:
    cs = load_cross_section()
    panel = load_panel()

    logtab = log_transform_table(cs)
    print("Log-transform comparison")
    print(logtab[["specification", "r2_in_sample", "loo_r2", "loo_rmse",
                  "aic", "density_p", "max_abs_predictor_skew"]]
          .round(3).to_string(index=False))

    misstab = missingness_table(cs, panel)
    print("\nMissingness")
    print(misstab[["unit", "years", "variables_missing"]].to_string(index=False))
    print("\nSaved table12, table12b and table13.")


if __name__ == "__main__":
    main()
