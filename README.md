# Telco Customer Churn — Prediction and Intervention Tiers

**Who is about to leave, and what should we do about each of them?**

Capstone project — ALY 6140, M.P.S. Analytics, Northeastern University.

A companion analysis to
[**customer-retention-causal-analysis**](https://github.com/janemeimei/customer-retention-causal-analysis),
which takes the same dataset and asks a different question: not *who will churn*, but *who will
stay because of what we do*. Prediction ranks risk. It does not tell you whom to treat. That
repository shows why the distinction matters — 19.3% of this same customer base is made **more**
likely to leave by the intervention meant to retain them.

---

## Result

| | |
|---|---|
| Best model | NeuralNetTorch (bagged), selected by AutoGluon from a 20+ model leaderboard |
| **ROC-AUC** | **0.841** on held-out test data |
| Train / test | 8,278 / 1,409 after SMOTE balancing |
| Explainability | SHAP values on the final model |

Model selection was delegated to AutoGluon rather than hand-tuned. The leaderboard is in the
notebook — neural nets, CatBoost, and weighted ensembles land within 0.006 AUC of each other,
which is a more useful finding than any single winner. **At that margin, model choice is not
where the value is.** Feature quality and what you do with the score are.

---

## What actually drives churn

| Finding | Churn rate | Business action |
|---|---|---|
| Month-to-month contracts | **42.7%** | Incentivize longer commitments |
| Electronic check payment | **45.3%** | Encourage auto-pay enrollment |
| New customers (<12 months) with high charges | highest risk segment | First-year loyalty pricing |
| **ServiceBundleCount** | **#1 predictor overall** | Cross-sell for stickiness |
| Fiber optic vs DSL | **2× higher churn** | Investigate service quality and pricing |

The payment-method finding is the one worth pausing on. Electronic check is not causing churn —
it marks customers who never committed to an automatic relationship with the company. The action
is not to discourage the payment method; it is to understand what auto-pay enrollment represents.

---

## From probability to policy

A binary churn/no-churn output forces one intervention for everyone. The probability score
supports tiering instead:

| Risk | Probability | Intervention |
|---|---|---|
| Critical | 80%+ | Personal outreach from retention team, discount and contract incentives |
| High | 60–80% | Proactive email campaigns, loyalty rewards, upgrade offers |
| Medium | 40–60% | Monitoring — check-ins, satisfaction surveys, value reinforcement |
| Low | <40% | Standard service |

**This is where prediction stops being useful on its own.** The tiers assume intervention
intensity should scale with risk. That assumption is testable — the companion causal analysis
tests it, and finds it does not hold uniformly.

---

## Method

```
EDA                  distribution and correlation analysis, churn rate by segment
Feature engineering  ServiceBundleCount and derived tenure / charge features
Class imbalance      SMOTE on the training split only
Scaling              StandardScaler, parameters retained for inverse transformation
Model selection      AutoGluon TabularPredictor, ROC-AUC as evaluation metric
Explainability       SHAP
```

---

## Limitations

- **SMOTE was applied to the training split only**, but synthetic minority samples still risk
  teaching the model boundaries that do not exist in the real distribution.
- **No time dimension.** The dataset is a cross-sectional snapshot; there is no way to observe
  behaviour trending toward churn. The notebook identifies time-series features as the highest
  priority enhancement.
- **Correlational.** Every finding here is an association. "Month-to-month contracts have 42.7%
  churn" does not mean converting customers to annual contracts would reduce their churn to the
  annual-contract rate — those customers differ in ways contract type only reflects.
- **Public benchmark dataset**, widely used in coursework. The results are not novel; the value
  is in the treatment of the problem.

---

## Data and stack

IBM Telco Customer Churn — 7,043 customers; demographics, contract terms, service configuration,
billing, and churn outcome.

Python · Pandas · NumPy · scikit-learn · AutoGluon · SHAP · imbalanced-learn · Matplotlib · Seaborn

---

**Jane (Jingjie) Mei** · M.P.S. Analytics, Northeastern University
[LinkedIn](https://linkedin.com/in/janemei-analytics) · [Tableau](https://public.tableau.com/app/profile/jane.mei)
