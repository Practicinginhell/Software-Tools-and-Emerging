# MapleFreight Logistics: Delivery Delay Prediction and Slack Alerts

Predict which in-transit shipments will miss the promised delivery window, and post the
high-risk shipments to Slack (`#dispatch-alerts`) while dispatch still has time to re-route or
warn the customer.

| | |
|---|---|
| Notebook | [`MapleFreight_Delivery_Delay_Prediction_c0955748.ipynb`](MapleFreight_Delivery_Delay_Prediction_c0955748.ipynb) (runs top to bottom, outputs saved) |
| Dataset | [`data/maplefreight_delivery_delay_dataset.csv`](data/maplefreight_delivery_delay_dataset.csv) |
| Slack screenshot | [`screenshots/slack_alert.png`](screenshots/slack_alert.png) |
| Final model | Class-weighted logistic regression (C = 0.03) |
| Alert threshold | **0.52** risk score: recall 0.73, precision 0.46 on the held-out test set |

## Problem

MapleFreight moves about 8,000 shipments a week across Ontario, Quebec and the Prairies. On-time
performance has fallen from 94% to 88%, and today a delay is only discovered when the customer
calls. By then the service credit is owed, a re-delivery has to be expedited, and the account
manager has heard of the delay before dispatch. The goal is to flag likely-late shipments **at
dispatch time** and push alerts to Slack automatically.

## Dataset

`maplefreight_delivery_delay_dataset.csv` has 6,035 shipments and 23 columns: route, carrier,
service level, goods, weather forecast, traffic, driver and vehicle details. The target is
`delivered_late` (1 = missed the promised window), and 20.6% of shipments are late.

The raw export needed cleaning:

| Issue | Fix |
|---|---|
| `service_level` spelled `standard` / `express` / `economy` in 120 rows | normalised to title case |
| 40 `weight_kg` values between 170,900 and 2,999,300 (recorded in **grams**) | divided by 1,000 |
| 32 exact duplicate rows plus 3 IDs repeated with a different `service_level` | deduplicated to 6,000 unique shipments |
| Missing `fuel_cost_cad` (240), `driver_experience_years` (210), `weather_condition` (150), `traffic_index` (121) | median imputation inside the pipeline, and missing weather becomes a separate `Unknown` level |
| **`actual_transit_hours` is target leakage.** Late ⇔ actual > about 1.1 × scheduled hours, and the value is only known after delivery | excluded from all models |

## How to run

```bash
cd MapleFreight_Delivery_Delay_Prediction
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Slack: create #dispatch-alerts → create a Slack app → enable Incoming Webhooks →
# add a webhook for the channel, then:
cp .env.example .env          # paste the webhook URL as SLACK_WEBHOOK_URL
                              # .env is git-ignored and must never be committed

jupyter nbconvert --to notebook --execute --inplace MapleFreight_Delivery_Delay_Prediction_c0955748.ipynb
# or open the notebook in Jupyter / VS Code and choose "Restart & Run All"
```

Without a `.env` the notebook still runs end to end. The notebook prints the alert text and reports
that the message was not sent. The webhook URL is never printed: if the request fails, only the exception
type is shown, because `requests` error messages include the full URL.

## Approach

1. **Data understanding and cleaning.** Profiling, the quality fixes above, range checks, and the
   leakage test.
2. **EDA.** Late rate by every category and numeric quintile, a carrier × weather heatmap, and
   correlations. A plain-language guide follows every chart in the notebook, explaining how to read
   the chart and what stands out.
3. **Feature engineering.** `severe_weather`, `pickup_delay_hours`, `sched_hours_per_100km`,
   `stops_per_100km`, `is_winter`, `novice_driver`, `pickup_delay_share`. An ablation shows the new
   features add no measurable PR-AUC over the raw columns, and the notebook reports the result.
4. **Modelling.** Stratified 80/20 split. All selection uses 5-fold CV on the training set, and the
   test set plays no part in any model or threshold choice.
   - Floor: majority-class dummy (79% accuracy, 0 recall).
   - **Baseline:** unweighted logistic regression on raw features at threshold 0.5.
   - Candidates: class-weighted logistic regression, random forest, and XGBoost
     (`scale_pos_weight`), with a small grid search for logistic regression and XGBoost.
   - Imbalance is handled with class weights plus a recall-driven threshold (no SMOTE, so no
     synthetic shipments).
5. **Explainability.** Coefficients and odds ratios, permutation importance, a SHAP beeswarm, and a
   SHAP waterfall for the top alert.
6. **Slack.** The held-out test shipments stand in for the live in-transit batch. The shipments are scored
   and filtered at the threshold, and one phone-friendly digest is posted with the count, the number of
   critical shipments, and the top 5 (ID, destination, carrier, risk score).

## Results

**Cross-validation (training set, threshold 0.5)**

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Majority class | 0.206 | 0.500 | 0.00 | 0.00 | 0.00 |
| **Baseline: LogReg, unweighted** | 0.560 | 0.811 | 0.63 | 0.33 | 0.43 |
| LogReg, class-weighted | 0.559 | 0.810 | 0.42 | 0.73 | 0.53 |
| LogReg, class-weighted + engineered | 0.558 | 0.810 | 0.42 | 0.72 | 0.53 |
| Random forest, class-weighted | 0.516 | 0.789 | 0.56 | 0.37 | 0.45 |
| XGBoost, `scale_pos_weight` | 0.532 | 0.792 | 0.44 | 0.65 | 0.53 |

After tuning, logistic regression reaches 0.561 (C = 0.03) and XGBoost 0.556. The best XGBoost uses
depth-1 trees, an additive model, consistent with the finding that the risk factors simply add up.

**Held-out test set (1,200 shipments, 247 late)**

| Model | Threshold | Precision | Recall | F1 | PR-AUC | Late caught | False alarms |
|---|---|---|---|---|---|---|---|
| Baseline: LogReg, unweighted | 0.50 | 0.70 | 0.35 | 0.47 | 0.600 | 87 | 38 |
| **Final: LogReg, class-weighted** | **0.52** | 0.46 | **0.73** | **0.56** | 0.608 | **181** | 216 |

The confusion matrices for both models are in §12 of the notebook. Almost all of the gain comes
from handling the imbalance rather than from a better ranking: every model family tops out around
0.81 to 0.83 ROC-AUC with dispatch-time features.

## Chosen threshold: 0.52, and why

The model's risk score is used to rank shipments, and the cut-off for an alert was chosen on
**out-of-fold training predictions**, so the test set played no part in the choice.

- **Recall first.** A missed late shipment costs a service credit, an expedited re-delivery and an
  escalation. A false alarm is assumed to cost a dispatcher only a few minutes (the dataset has no cost
  figures). The rule is: catch **at least 70% of late shipments** (a business target, not a result of
  the data), then pick the **highest** threshold that meets the target. The highest qualifying
  threshold keeps precision and alert volume as good as possible. That gives 0.52: out-of-fold recall 0.71, precision 0.44. On the test
  set, recall is 0.73 and precision 0.46, or about 1 real late shipment for every 2.2 alerts.
- **Why not lower.** Moving from 0.50 to 0.40 would catch about 170 more late shipments a week but add
  about 940 alerts. That is **5.6 alerts per extra catch** compared with 2.4 at 0.50. Precision would
  fall to 0.36, where dispatch would likely start ignoring the channel.
- **Why not higher.** From 0.50 to 0.60, recall drops from 0.73 to 0.60, about 200 fewer late shipments
  caught each week.
- The F1-optimal threshold is 0.55. F1 differs by under 0.01 (0.541 at 0.52 against 0.545), so
  prioritising recall costs very little.
- About a third of shipments are flagged (about 2,700 alerts a week), so Slack gets **one digest per run** rather
  than one message per shipment. Scores **≥ 0.80** are labelled *critical* (precision about 0.67)
  for triage.

## Key findings: what drives late delivery

| Driver | Evidence |
|---|---|
| **Forecast weather** | Storm 46% late, snow 34%, clear 14%. The top permutation-importance feature. |
| **Carrier** | Contract owner-operators 36% late vs in-house MapleFreight Fleet 13% (odds ratio 1.8 vs 0.47). The gap persists in every weather type, and an owner-operator in snow or a storm is late more than half the time. |
| **Traffic and route congestion** | Late rate rises from 12% to 32% across traffic quintiles and from 16% to 27% across route congestion quintiles. |
| **Trip length and stops** | Longest-distance quintile 34% late vs 13% for the shortest. 0 stops 12% vs 5 stops 29%. |
| **Pickup delay** | 13% late when picked up early vs 28% for the latest pickups. |
| **Service level** | Economy 27% late, Express 15%. |
| **Carrier's prior 30-day late count** and **driver experience** | Prior late count: 15% late with none and 31% with four (a small dip at five, on few shipments). Experienced drivers are protective (26% → 14%). |
| **No signal** | Customer tier (Gold = Bronze = 21%), city, weight, fuel cost. |

## Recommendations

1. **Run the model in the dispatch loop.** Score in-transit shipments every 30 minutes and post the
   digest at threshold 0.52. About 1,200 of roughly 1,650 weekly late shipments get flagged before
   the window closes, compared with about 580 under the baseline. Handle critical (≥ 0.80)
   shipments first.
2. **Weather playbook.** On snow and storm days, pad promised windows, move pickups earlier, and keep
   owner-operators off severe-weather lanes.
3. **Carrier management.** Put on-time SLAs into owner-operator contracts and use
   `prior_late_deliveries_30d` as a live scorecard.
4. **Make the priority tier mean something.** Gold accounts get no better on-time performance than
   Bronze today. Send Gold and Express loads to the fleet and experienced drivers first.
5. **Fix pickups.** A pickup more than 30 minutes late should trigger a re-score.
6. **Quote realistic transit times.** Build traffic, stops and congestion into
   `scheduled_transit_hours`.
7. **Driver pairing.** Give novice drivers low-risk lanes.
8. **Data and model.** Enforce kg units, a service-level vocabulary and unique IDs at the source.
   Add in-transit signals (GPS progress, live traffic). Log alert outcomes and retrain monthly.

## Repository hygiene

- `.env` holds the webhook and is listed in `.gitignore` (here and at the repository root).
  `.env.example` is the committed template.
- The notebook prints only whether a webhook is configured, never the URL.

## AI usage

**Tool used:** Sonnet 5.5

**Where the AI output was wrong:**

1. **The SHAP waterfall showed standardised values instead of real units.** The first version
   labelled the top alert as `2.041 = severe_weather` and `2.784 = distance_km`. The values are z-scores
   from the `StandardScaler`, not a binary flag of 1 or a distance of 1,512 km, so a dispatcher
   reading the chart would have been misled. The fix was to attach unscaled (imputed) values as the
   explanation's `display_data`.
2. **Unsupported and stale claims in the written text.** The AI added an explanation the data does not
   support (Economy shipments "get bumped when capacity is tight") and quoted numbers that did not
   match the executed notebook: baseline precision "about 0.6" (actually 0.70), snow "33% late"
   (34%), a SHAP waterfall average of -0.337 (the notebook shows -0.471), and "about 30
   Unknown-weather shipments per carrier" (the real range is 12 to 56). Each claim had to be
   recomputed against the notebook and corrected.

**Also checked.** One point was verified on purpose rather than trusted: `requests` puts the full URL in connection-error
messages, so the Slack error handler prints only the exception type and the webhook cannot leak into
the saved notebook output.
