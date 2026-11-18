# Data dictionary

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
