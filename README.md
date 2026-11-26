# Modelling State-Level Gross Enrolment Ratio in Indian Higher Education

**Problem 18 — regression analysis and 2035 projection**

This repository contains a complete, reproducible analysis of the state-level
Gross Enrolment Ratio (GER) in Indian higher education in 2021-22, and a
transparent projection of which States can reach the NEP 2020 target of **50%
GER by 2035**.  The deliverables include the analysis code, official data
provenance, validation checks, all tables and figures, and a submission-ready
journal-format report (PDF and Markdown).

---

## 1. Research question

> How is the state-level GER in higher education associated with college
> density, the pupil–teacher ratio (PTR) and gender parity, and which States are
> on track to reach 50% GER by 2035-36?

The analysis is deliberately **state-level and policy-oriented**.  It does not
claim causal identification from a 28-state cross-section; it combines:

1. a cross-sectional regression with three policy-relevant indicators;
2. a 2017-18 to 2021-22 panel to estimate a common time trend;
3. a change-based projection anchored to each State's actual 2021-22 GER;
4. a bootstrap that carries coefficient, trend and residual uncertainty into
   the 2035 projections.

---

## 2. Key findings at a glance

| Finding | Result |
|---|---|
| Estimation sample | 28 States (8 Union Territories analysed separately) |
| Mean GER, 2021-22 | 30.4% (range 16.9–47.0%; 14 States below 30%) |
| OLS fit | R² = 0.543, adjusted R² = 0.486, residual SD = 6.31 points |
| Out-of-sample fit | LOOCV R² = 0.370, RMSE = 6.86, MAE = 5.73 |
| Best LOOCV predictor | Polynomial degree 2 + ridge: R² = 0.525, RMSE = 5.96 |
| Tuned non-parametric models | k-NN RMSE 7.59, tree RMSE 7.39 (nested LOOCV) — worse than OLS despite in-sample R² of 0.66 and 0.70 |
| College density | +0.353 GER points per additional college per lakh (p = 0.029; 95% bootstrap CI 0.112–0.656) |
| Pupil–teacher ratio | −0.189 (p = 0.211); sign flips in panel/first-difference models — treat as endogenous |
| Gender Parity Index | +0.508 (p = 0.959); no independent association conditional on the other two |
| Diagnostic checks | Shapiro p = 0.561, Breusch–Pagan p = 0.384, RESET p = 0.885 |
| Reference projection (S2) | 13 of 28 States reach 50% by 2035-36 |
| Damped projection (S1) | 3 of 28 States reach 50% |
| Reform projection (S3) | 13 of 28 States; PTR/GPI reforms add ~1.2 points on average but do not change the count |
| Population-weighted GER, 2035 (S2) | 43.5% — below the 50% target; 69% of the 18–23 population lives in States projected below 50% |

**Reference-scenario States reaching 50%:** Tamil Nadu, Himachal Pradesh,
Uttarakhand, Kerala, Telangana, Sikkim, Arunachal Pradesh, Karnataka, Goa,
Manipur, Maharashtra, Mizoram and Rajasthan.

**Largest model-implied density gaps:** Nagaland, Bihar, Assam, Jharkhand,
Tripura, Chhattisgarh, Uttar Pradesh and Gujarat.  These are extrapolations of a
linear association, not costed plans; see the limitations below.

---

## 3. Data

### 3.1 Target

- **Source:** Open Government Data Platform India (OGD), resource
  `7edbaeed-65e9-43ad-8f81-730187b70463`
  ([resource page](https://www.data.gov.in/resource/statesut-wise-gross-enrolment-ratio-ger-higher-education-based-2011-population-census)).
- **Licence and access date:** Government Open Data Licence--India (GODL-India);
  retrieved on 25 October 2026.
- **Title:** *States/UT-wise Gross Enrolment Ratio (GER) in Higher Education
  Based on 2011 Population Census Projection for 18–23 Age Group during
  2021-22.*
- **Coverage:** 36 States/UTs; 36 records retrieved through the official OGD API.
- **Note:** the OGD CSV link returns HTTP 403; the API is used instead and the
  raw JSON is stored in `data/raw/ogd_ger_2021_22.json`.

### 3.2 Predictors

- **Source:** Ministry of Education, *All India Survey on Higher Education
  (AISHE) 2021-22, Final Report* (356 pp.).
- **Tables used:** Table 4 (p. 72), Table 42 (pp. 235–236), Table 47
  (pp. 276–279), Table 48 (p. 280), Table 49 (pp. 283–286).
- **Variables:**
  - `college_density`: colleges registered with AISHE per lakh population aged
    18–23;
  - `ptr`: pupil–teacher ratio, all institutions, regular mode;
  - `gpi`: female GER / male GER;
  - `avg_enrolment_per_college`: used only in a robustness model because it
    shares enrolment mechanically with GER.
- **Panel:** 2017-18 to 2021-22, 36 units × 5 years = 180 rows.

### 3.3 Validation

`src/extract_aishe.py` writes `data/processed/extraction_validation.txt` and
checks:

- 180 panel rows and 36 units;
- Table 42 vs Table 4: maximum absolute difference = **0.000** for both college
  density and average enrolment per college;
- GPI vs female/male GER: maximum absolute difference = **0.005**;
- GER 2021-22 matches the OGD target exactly (maximum difference = 0);
- known headline values (Karnataka density 66, Bihar PTR 64, Kerala GPI 1.44).

Raw-file SHA-256 checksums are in `data/raw/SHA256SUMS.txt`.  The AISHE PDF
hash is `c4400273cb2d574c793390a900528c825b036667cb0977bbdb8c40f0276d5f15`.

### 3.4 Sample

The primary estimation sample is the **28 States**.  The eight Union
Territories are excluded from estimation because:

- their tertiary catchments are city-scale and cross State boundaries;
- three UTs (Chandigarh, Delhi, Ladakh) are high-leverage outliers;
- including all 35 units with complete predictors reduces LOOCV R² from 0.37 to
  0.16.

UTs are nevertheless reported in `outputs/tables/table06b_projections_uts.csv`
as an indicative appendix.  Lakshadweep has no college-density/enrolment data;
Ladakh has only three panel years.

---

## 4. Methods

### 4.1 Models compared

| Model | Purpose |
|---|---|
| Baseline (mean) | Reference for R²/RMSE |
| OLS, 3 indicators | Interpretable inference; primary model |
| Ridge, 3 indicators | Regularisation / coefficient-stability check |
| Polynomial degree 2 + ridge | Nonlinearity check and prediction benchmark |
| k-NN, tuned | Non-parametric benchmark with nested tuning |
| Regression tree, tuned | Interpretable non-parametric benchmark |
| OLS + average enrolment per college | Robustness only (mechanical share with GER) |
| Ridge + average enrolment per college | Robustness only |

- Feature scaling (`StandardScaler`) is applied inside the ridge and polynomial
  pipelines, and is fitted on the training fold only.
- Model selection uses **leave-one-out cross-validation (LOOCV)** and **20
  repeats of shuffled 5-fold CV**; metrics are R², RMSE and MAE.
- Hyper-parameters for k-NN and the tree are chosen by `GridSearchCV` **inside
  each outer fold**, so the reported LOOCV score is not contaminated by tuning.
  `table10_hyperparameter_search.csv` reports both the internal search score
  (optimistic) and the nested score (honest).
- OLS inference uses HC3 heteroskedasticity-robust standard errors and 2,000
  bootstrap resamples; standardised coefficients and VIFs are reported.
- Diagnostics: Shapiro–Wilk, Breusch–Pagan, Ramsey RESET, Cook's distance,
  leverage, studentised residuals and condition number.
- Panel checks: year fixed effects, State + year fixed effects, first
  differences, State-clustered standard errors.
- Influence checks: drop Karnataka; drop Karnataka + Bihar; UT-inclusive
  sensitivity.

### 4.2 Projection

The projection is **change-based**, not level-based:

```
GER_s,2035 = GER_s,2021
           + b_density × (D_s,2035 − D_s,2021)
           + b_PTR     × (P_s,2035 − P_s,2021)
           + b_GPI     × (G_s,2035 − G_s,2021)
           + trend × h × m
```

where `h = 14` years (2021-22 to 2035-36), `m` is the scenario trend multiplier,
and the coefficients are the OLS estimates.  Anchoring to the observed 2021-22
GER avoids using the cross-sectional intercept to forecast the level of GER.

| Scenario | Density change | PTR | GPI | Common trend |
|---|---|---|---|---|
| **S1 Damped** | 0.5 × recent annual change | constant | constant | 0.5 × |
| **S2 Recent trend** (reference) | 1.0 × recent annual change | constant | constant | 1.0 × |
| **S3 NEP reform** | 1.0 × recent annual change | converges to min(2021, 20) | converges to 1.05 if below | 1.0 × |

- Recent annual density change = 2017-18 to 2021-22 change / 4.
- Density is capped at 100 colleges per lakh (observed maximum = 66).
- The common trend is **0.866 GER points per year** (SE 0.174), estimated from
  the 2017-22 panel with year fixed effects.
- Uncertainty: 2,000 state-resampling bootstrap draws; OLS re-estimated on each
  resample; trend drawn from N(0.866, 0.174); a State-level residual drawn from
  N(0, 6.315); common random numbers across scenarios.
- Classification: **Achieved** (2021 GER ≥ 50), **On track** (P(≥50) ≥ 0.5),
  **At risk** (0.15 ≤ P < 0.5), **Off track** (P < 0.15).
- Required density: solve the S2 equation for the density that yields 50%,
  holding PTR/GPI at 2021 values; the gap is reported with an indicative
  additional-college calculation based on the implied 18–23 population
  (`colleges / density`).

### 4.3 Why not use the polynomial for projection?

The degree-2 polynomial has the best LOOCV metrics (R² = 0.525, RMSE = 5.96),
but beyond about 45–66 colleges per lakh it turns downward and becomes
non-monotone (`outputs/figures/fig11_polynomial_instability.png`).  Using it to
project 2035 would imply that further expansion eventually *reduces* GER.  The
linear model is retained for projection because it is monotone and
interpretable; the polynomial result is reported as a predictive benchmark and
a caution against extrapolating flexible functional forms.

---

## 5. Project structure

```
ml_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # official PDF, OGD JSON, checksums
│   └── processed/               # analysis CSVs, data dictionary, validation
├── src/
│   ├── config.py                # paths, feature lists, State/UT lists, style
│   ├── download_data.py         # OGD API + AISHE PDF download and checksums
│   ├── extract_aishe.py         # PDF table extraction and validation
│   ├── build_dataset.py         # merge target and predictors, derive variables
│   ├── eda.py                   # tables 01-02b, figures 01-04
│   ├── checks.py                # tables 12-13 (log transform, skewness, missingness)
│   ├── modeling.py              # tables 03-05, 07, 10, figures 05-11
│   ├── projection.py            # tables 06, 06b, 08, 09, 11, figures 12-15
│   ├── verify_outputs.py        # asserts the numbers quoted in the paper
│   └── run_all.py               # end-to-end runner
├── outputs/
│   ├── tables/                  # all CSV tables
│   ├── figures/                 # all PNG figures (300 dpi)
│   ├── model_results.json       # fitted coefficients, diagnostics, panel, tuning
│   ├── log_transform_results.json # skewness and log-transform specification tests
│   └── projection_results.json  # scenario definitions, summaries, weights
├── paper/
│   ├── paper.tex                # Overleaf-ready LaTeX manuscript
│   ├── references.bib           # BibTeX bibliography with verified DOIs
│   ├── sections/                # modular LaTeX sections
│   ├── figures/                 # publication figures
│   └── paper.pdf                # compiled article PDF
└── report/
    ├── report_template.md       # initial Markdown template
    ├── report_meta.json         # author, college, roll number, e-mail, supervisor
    ├── build_report.py          # Markdown -> HTML -> PDF
    ├── report.md                # generated Markdown report
    ├── report.html              # generated print-ready HTML
    └── GER_Report.pdf           # initial submission PDF
```

`scratch/` holds working downloads and notes from data exploration; nothing in
the pipeline reads from it.

---

## 6. Quick start

The project was developed on Windows with a local Python at `.py\python.exe`.
The code itself is cross-platform; use the section for your operating system.

### Windows

```powershell
# 1. Install dependencies (once)
.\.py\python.exe -m pip install -r requirements.txt

# 2. Run everything (downloads the two official sources)
.\.py\python.exe src\run_all.py

# 3. Or skip the downloads if data/raw is already populated
.\.py\python.exe src\run_all.py --skip-download

# 4. Run individual steps
.\.py\python.exe src\eda.py
.\.py\python.exe src\modeling.py
.\.py\python.exe src\projection.py

# 5. Check the reported numbers, then rebuild the report
.\.py\python.exe src\verify_outputs.py
.\.py\python.exe report\build_report.py
```

### Linux / macOS

```bash
# 1. Create and activate a virtual environment (once)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (once)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Run everything (downloads the two official sources)
python src/run_all.py

# 4. Or skip the downloads if data/raw is already populated
python src/run_all.py --skip-download

# 5. Run individual steps
python src/eda.py
python src/modeling.py
python src/projection.py

# 6. Check the reported numbers, then rebuild the report
python src/verify_outputs.py
python report/build_report.py
```

`build_report.py` looks for `google-chrome`, `chromium` or `chromium-browser`
on Linux and for Edge/Chrome in their standard Windows locations.  If no browser
is found it still writes `report/report.md` and `report/report.html`, which can
be printed to PDF manually.

On a headless Linux server, set the Matplotlib backend before running:

```bash
export MPLBACKEND=Agg
python src/run_all.py --skip-download
```

A convenience wrapper is also included:

```bash
chmod +x run.sh
./run.sh                 # full run
./run.sh --skip-download # use existing raw files
```

The pipeline is deterministic (`RANDOM_STATE = 42`).  Re-running
`modeling.py` and `projection.py` reproduces the reported numbers; the
bootstrap uses 2,000 draws.

---

## 7. Output inventory

### Tables

| File | Contents |
|---|---|
| `table01_summary_statistics.csv` | Descriptive statistics, 28-State estimation sample |
| `table01b_summary_statistics_all_units.csv` | Descriptive statistics, all 36 units |
| `table02_correlations.csv` | Pearson correlations, 28 States |
| `table02b_spearman.csv` | Spearman correlations, 28 States |
| `table03_model_comparison.csv` | LOOCV, repeated CV and in-sample metrics |
| `table04_coefficients.csv` | OLS coefficients, HC3 SEs, bootstrap CIs, standardised coefficients, VIF |
| `table05_diagnostics.csv` | Residuals, leverage, Cook's D, DFFITS |
| `table06_projections_2035.csv` | State projections, probabilities, intervals, status |
| `table06b_projections_uts.csv` | Indicative UT appendix |
| `table07_sensitivity.csv` | UT-inclusive, panel, first-difference and leave-out checks |
| `table08_required_density.csv` | Required density, gap, annual change, indicative colleges |
| `table09_scenario_summary.csv` | Scenario-level counts and distributions |
| `table10_hyperparameter_search.csv` | Tuning grids, selected values, internal vs nested CV error |
| `table11_state_appendix.csv` | Per-State LOOCV prediction and S2 projection with intervals |
| `table12_log_transform.csv` | Raw vs log-transformed OLS comparison |
| `table12b_skewness.csv` | Raw and log skewness of predictors and GER |
| `table13_missingness.csv` | Panel and estimation-sample missingness audit |

### Figures

| File | Contents |
|---|---|
| `fig01_ger_ranked.png` | Ranked GER, all 36 units |
| `fig02_correlates.png` | GER vs the four candidate correlates |
| `fig03_correlation_heatmap.png` | Correlation matrix |
| `fig04_trajectories.png` | 2017-18 to 2021-22 State trajectories |
| `fig05_model_comparison.png` | LOOCV RMSE and R² by model |
| `fig06_actual_vs_predicted.png` | Linear vs polynomial out-of-sample predictions |
| `fig07_residual_diagnostics.png` | Residuals, Q–Q, scale–location, histogram |
| `fig08_coefficients.png` | Standardised coefficients with bootstrap CIs |
| `fig09_partial_effects.png` | Component-plus-residual plots |
| `fig10_influence.png` | Cook's distance and leverage |
| `fig11_polynomial_instability.png` | Why the polynomial is not used for projection |
| `fig12_projection_dumbbell.png` | 2021 → 2035 projections with intervals |
| `fig13_probability_reaching_50.png` | Probability of reaching 50% by scenario |
| `fig14_scenario_distributions.png` | Scenario distributions and counts |
| `fig15_required_density.png` | Required density and indicative college requirement |

---

## 8. Reproducibility and validation notes

- The OGD API key in `download_data.py` is a public sample key; set
  `OGD_API_KEY` to your own key if the sample key is rate-limited.
- The OGD resource page blocks direct CSV downloads (HTTP 403); the script uses
  the official API with pagination (`limit=10`, `offset`).
- AISHE extraction is validated against Table 4 and the OGD target; see
  `data/processed/extraction_validation.txt`.
- `src/verify_outputs.py` re-reads the generated artefacts and asserts the
  headline numbers quoted in the report (sample size, coefficients,
  diagnostics, scenario counts, output inventory, log-transform checks,
  provenance).  A failing check means a reproduction failure, not a typo.
- Sikkim's published 2019-20 GER (66.9%) breaks its own series by 27 points in
  one year.  It is retained and annotated in Figure 4, and the year
  fixed-effects model is also estimated without Sikkim (density 0.416, trend
  0.845, SE 0.161) to show that the reported trend does not depend on it.
- All India rows are excluded from the State-level files; `India` is not in the
  36-unit canonical list.
- `college_density` is per lakh of the **18–23 population**, not total
  population.  This matters for interpreting the implied population and the
  additional-college calculation.
- The 2011 Census population projection is the denominator used by AISHE and
  OGD; the analysis inherits that base.

---

## 9. Limitations

1. **Cross-sectional association is not causation.**  The density coefficient
   is conditional on PTR and GPI; it does not identify the effect of building a
   college.
2. **Small n.**  With 28 States and three predictors, the model has limited
   power; the GPI coefficient is essentially unidentified.
3. **PTR endogeneity.**  PTR is mechanically related to enrolment and teacher
   supply; its sign flips in panel and first-difference models.  It should not
   be read as a causal lever.
4. **Omitted variables.**  Income, urbanisation, secondary-school completion,
   private provision, distance education and State capacity are not observed.
   The model explains about 54% of cross-State variance in-sample and 37%
   out-of-sample.
5. **Extrapolation.**  Density projections beyond the observed range (max 66)
   are capped at 100; the required-density and additional-college figures are
   model-implied illustrations, not costed plans.
6. **Trend assumption.**  The common 0.866-point annual trend is an average
   from a short, COVID-affected panel; it may not continue for 14 years.
7. **UT exclusion.**  UTs are excluded from estimation for methodological
   reasons; their appendix projections are indicative only.
8. **Data revisions.**  GER 2021-22 is provisional in the OGD release; AISHE
   figures may be revised.  Reporting practice can also break a series abruptly,
   as Sikkim's 2019-20 value shows; such breaks are annotated and tested rather
   than smoothed away.

---

## 10. Suggested citation

> Jha, H. K. (2026). *Modelling State-Level Gross Enrolment Ratio in Indian
> Higher Education: Correlates, Validation and a 2035 Projection*. Project
> report, Department of Computer Science, Ramanujan College, University of
> Delhi.

---

## 11. Licence and attribution

The code is released for academic and policy use.  The underlying data remain
subject to the terms of the Ministry of Education (AISHE) and the Open
Government Data Platform India (GODL-India).  Please cite the original sources
when reusing the data.
