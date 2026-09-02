# Home-Credit
An interactive risk-scoring dashboard for the Home Credit Default Risk dataset, built with a proper FastAPI backend and a Streamlit frontend as two separate services. The backend owns the trained model and all inference logic; the frontend is a pure UI layer that talks to it over HTTP.

Live app: https://home-credit-default-risk-7fax.onrender.com API docs: https://home-credit-backend.onrender.com/docs

# What it does
Pick a real sampled applicant from the test set, then edit the fields a person could actually declare on a loan application: income, credit amount, annuity, goods price, family size, and days employed. Everything else the model uses — external bureau scores, prior loan history, payment behavior — stays fixed to that applicant's real data, because those are things a bureau reports, not things an applicant types in.

I went with this split on purpose. An earlier version of the project let you edit every single feature, including the aggregated bureau fields, and it just felt dishonest — those numbers represent years of financial history sitting in a credit bureau's systems, not something you'd know off the top of your head. Letting someone drag a slider for it misrepresents how the model actually works.
# Backend
inference.py — loads the model, feature list, and cached applicant data; recomputes derived features; runs predictions and SHAP explanations

model_store.py — loads everything once at startup and keeps it in memory, so nothing gets re-read from disk per request

schemas.py — Pydantic models for request/response validation

predict.py — /predict/ router: applicant ID + field overrides in, probability + risk band + top-10 SHAP values out

application.py — /applicants/ router: lists valid applicant IDs, returns one applicant's declarable fields

timer.py — middleware that stamps every response with an X-Response-Time header

main.py — entrypoint, wires up startup, CORS, middleware, and both routers
# The model
Trained on a merge of seven tables from the Home Credit dataset — application, bureau, previous applications, installments, credit card balances, POS cash balances — aggregated to one row per applicant, fed into an XGBoost classifier tuned with Optuna over 100 trials. That whole pipeline lives in script/ and only runs once, locally, to produce model.pkl and features.pkl.

For serving, I ran the same merge once more over the Kaggle test set and cached the result to test_merged.parquet. Doing the full seven-table merge on every request would mean grouping tens of millions of rows live, which is slow and would probably blow through memory on a free-tier deploy. Precomputing it once means the backend just loads one small file at startup and responds fast.

When you edit a field and hit predict, the backend recomputes the ratio features the model actually depends on — credit-to-income, annuity-to-income, employment years, that kind of thing — before running inference, so the prediction reflects what you actually typed in.
# Risk sensitivity experiment
I wanted to know which of the six editable fields actually move the prediction, so I ran a quick test on one sampled applicant. Baseline default probability: 28.44%.

Round 1: set income to 300,000, credit to 250,000, annuity to 15,000, goods price to 240,000, family members to 1, days employed to -5,000 (about 13.7 years). This replaced an earlier set of values that didn't make much sense together (goods price way out of line with credit, annuity way out of line with both). Result: 24.85%. The SHAP chart showed annuity and employment years shift into favorable territory, while EXT_SOURCE_3 and POS_COUNT stayed locked and unfavorable, as expected since they're tied to the applicant's actual bureau history.

Round 2: pushed annuity down to 8,000 and days employed to -8,000 (~22 years), left everything else alone since round 1 showed it wasn't doing much. Result: 18.89%.

Where I stopped: by round 2, annuity's effect on the chart was visibly smaller than it had been in round 1 — diminishing returns. EXT_SOURCE_3 alone was carrying about as much weight as everything else in the chart combined, and it's not something the dashboard can touch. No point chasing the number further once the pattern's clear.

Only two of the six editable fields actually mattered for this applicant. The other four barely moved anything once the first round confirmed that. I think that's a good sign, not a limitation — it means you can't just fudge your income on a form and talk the model down to near-zero risk. The prediction stays anchored to real financial history.
