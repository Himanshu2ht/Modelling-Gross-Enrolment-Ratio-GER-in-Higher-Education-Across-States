"""
Reproduction checks for the GER modelling project.

This script does not re-estimate anything.  It reads the artefacts that the
pipeline wrote and asserts that the headline numbers quoted in the report are
present, internally consistent and reproducible from the stored outputs.  Run it
after ``run_all.py`` (or after ``modeling.py`` and ``projection.py``).

Exit code 0 means every check passed.  Exit code 1 means at least one check
failed, and the failing check is printed with the observed and expected values.

Usage
-----
    python src/verify_outputs.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from config import FEATURES, FIG, PROC, TAB, TARGET, load_cross_section

TOL = 5e-3
failures: list[str] = []
checks = 0


def close(name: str, observed: float, expected: float, tol: float = TOL) -> None:
    global checks
    checks += 1
    if abs(float(observed) - float(expected)) <= tol:
        print(f"  PASS  {name}: {float(observed):.4f}")
    else:
        failures.append(f"{name}: observed {observed!r}, expected {expected!r}")
        print(f"  FAIL  {name}: observed {float(observed):.4f}, expected {expected:.4f}")


def truth(name: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  PASS  {name}{(' — ' + detail) if detail else ''}")
    else:
        failures.append(f"{name}{(' — ' + detail) if detail else ''}")
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def main() -> int:
    print("1. Input data")
    cs = load_cross_section()
    est = cs[(cs["is_ut"] == 0) & cs[FEATURES].notna().all(axis=1)]
    truth("36 units in the cross-section", len(cs) == 36, f"n={len(cs)}")
    truth("28 States in the estimation sample", len(est) == 28, f"n={len(est)}")
    truth("no missing predictors in the estimation sample",
          est[FEATURES].notna().all().all())
    truth("GER values lie inside 0–100",
          bool(((cs["ger"] > 0) & (cs["ger"] < 100)).all()))

    print("\n2. Estimation results (model_results.json)")
    res = json.loads((FIG.parent / "model_results.json").read_text(encoding="utf-8"))
    close("OLS R²", res["ols"]["r2"], 0.5427)
    close("adjusted R²", res["ols"]["adj_r2"], 0.4855)
    close("residual SD", res["ols"]["resid_sd"], 6.3148)
    close("college-density coefficient", res["ols"]["coef"]["college_density"], 0.3528)
    close("PTR coefficient", res["ols"]["coef"]["ptr"], -0.1894)
    close("GPI coefficient", res["ols"]["coef"]["gpi"], 0.5079)
    close("ridge alpha", res["ridge"]["alpha"], 7.9433)

    comp = pd.DataFrame(res["model_comparison"]).set_index("model")
    close("OLS LOOCV R²", comp.loc["OLS (3 indicators)", "loo_r2"], 0.3696)
    close("OLS LOOCV RMSE", comp.loc["OLS (3 indicators)", "loo_rmse"], 6.8644)
    close("baseline LOOCV RMSE", comp.loc["Baseline (mean)", "loo_rmse"], 8.9655)
    close("polynomial LOOCV RMSE",
          comp.loc["Polynomial deg 2 + ridge", "loo_rmse"], 5.9573)
    truth("at least eight model rows are reported", len(comp) >= 8, f"n={len(comp)}")
    truth("every learned model beats the mean baseline",
          bool((comp.drop(index="Baseline (mean)")["loo_rmse"]
                < comp.loc["Baseline (mean)", "loo_rmse"]).all()))

    print("\n3. Diagnostics and panel checks")
    diag = res["diagnostics"]
    truth("Shapiro–Wilk does not reject normality", diag["shapiro_p"] > 0.05,
          f"p={diag['shapiro_p']:.3f}")
    truth("Breusch–Pagan does not reject homoskedasticity",
          diag["breusch_pagan_p"] > 0.05, f"p={diag['breusch_pagan_p']:.3f}")
    truth("Ramsey RESET does not reject linearity", diag["reset_p"] > 0.05,
          f"p={diag['reset_p']:.3f}")
    truth("no studentised residual above |2|", diag["n_outliers_2sd"] == 0)
    close("annual common trend", res["panel"]["annual_common_trend"], 0.8663)
    close("trend standard error", res["panel"]["annual_common_trend_se"], 0.1739)
    close("panel density coefficient", res["panel"]["panel_density"], 0.3546)

    sens = pd.read_csv(TAB / "table07_sensitivity.csv").set_index("specification")
    truth("Sikkim-excluded panel row is reported",
          "Panel, year FE excluding Sikkim" in sens.index)
    truth("Sikkim exclusion leaves the common trend close to 0.85",
          "trend=0.845" in str(sens.loc["Panel, year FE excluding Sikkim", "note"]))

    print("\n4. Projections (projection_results.json)")
    proj = json.loads((FIG.parent / "projection_results.json").read_text(encoding="utf-8"))
    summary = {s["key"]: s for s in proj["summary"]}
    truth("reference scenario crosses 13 States",
          summary["recent"]["n_point_above_50"] == 13,
          f"n={summary['recent']['n_point_above_50']}")
    truth("damped scenario crosses 3 States",
          summary["damped"]["n_point_above_50"] == 3,
          f"n={summary['damped']['n_point_above_50']}")
    truth("reform scenario crosses 13 States",
          summary["reform"]["n_point_above_50"] == 13,
          f"n={summary['reform']['n_point_above_50']}")
    close("population-weighted GER 2021", proj["weighted"]["ger_2021"], 28.0368)
    close("population-weighted GER 2035 (S2)",
          proj["weighted"]["ger_2035_recent"], 43.5118)
    close("population share below 50% (S2)",
          proj["weighted"]["share_population_below_50_recent"], 0.6864)

    print("\n5. Output inventory")
    tables = sorted(p.name for p in TAB.glob("table*.csv"))
    figures = sorted(p.name for p in FIG.glob("fig*.png"))
    truth("11 analysis tables are present", len(tables) >= 11, f"n={len(tables)}")
    truth("15 figures are present", len(figures) == 15, f"n={len(figures)}")
    truth("hyper-parameter table exists",
          (TAB / "table10_hyperparameter_search.csv").exists())
    truth("per-State appendix table exists",
          (TAB / "table11_state_appendix.csv").exists())
    app = pd.read_csv(TAB / "table11_state_appendix.csv")
    truth("appendix covers all 28 States", len(app) == 28, f"n={len(app)}")
    truth("appendix has no missing LOOCV predictions",
          app["loo_pred"].notna().all())

    print("\n6. Data provenance")
    validation = (PROC / "extraction_validation.txt").read_text(encoding="utf-8")
    truth("OGD–AISHE GER cross-check is exact",
          "max abs diff = 0.000 (n = 36)" in validation)
    truth("Table 42 vs Table 4 density check is exact",
          "max |density diff| = 0.000" in validation)
    truth("raw-file checksums are recorded",
          (PROC.parent / "raw" / "SHA256SUMS.txt").exists())

    print("\n7. Report inputs")
    report = (FIG.parent.parent / "report" / "report.md")
    truth("journal-format report source exists", report.exists())
    if report.exists():
        text = report.read_text(encoding="utf-8")
        truth("report contains no unresolved template tokens",
              "{{" not in text)
        truth("report quotes the 50% target", "50%" in text)
        for fig in figures:
            truth(f"figure referenced: {fig}", fig in text)
        truth("report declares AI-tool use", "Use of AI tools" in text)

    print(f"\n{checks - len(failures)}/{checks} checks passed.")
    if failures:
        print("\nFailures:")
        for f in failures:
            print("  -", f)
        return 1
    print("All reproducibility checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
