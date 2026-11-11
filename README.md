# RouteMind – Late-Delivery Risk Engine and Agent Demo

ML engine for the 5023CMD Artificial Intelligence CW2 group project.
Application area: **Agent-driven Logistics and Dynamic Supply Chains**.

RouteMind is an autonomous logistics agent. Its core ML component is a **Random Forest classifier** that predicts whether an order will be delivered late. The agent combines that risk score with Bayesian reasoning and fuzzy safety rules to choose an action (continue, monitor, reroute or escalate), and asks a human before any high-risk action.

## What is real and what is simulated

| Part | Status |
|---|---|
| Order data, late-delivery labels, the trained models and all evaluation results | **Real** (DataCo dataset) |
| Truck evidence ("idle > 60 min"), stock days, the routing API and its failures | **Simulated**, and labelled `SIMULATED` in every output. The dataset has no GPS, sensor or stock data. |

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis** (Constante, Silva and Pereira, 2019).

- Kaggle: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
- Mendeley Data: https://data.mendeley.com/datasets/8gx2fvg2k6/5

The CSV is **not stored in this repository** (about 96 MB). Download `DataCoSupplyChainDataset.csv` and place it in `data/`.

- 180,519 order items, 53 columns; label: `Late_delivery_risk` (1 = late, 0 = on time)
- 172,765 orders are used after removing cancelled shipments

## Setup

Requires Python 3.10 or newer (developed on Python 3.14, Windows).

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If activation fails with "running scripts is disabled", run this once and activate again:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run order

Run the scripts from the project folder, in this order. Every script reads its settings from `config.yaml` and writes a timestamped log to `logs/`.

| Step | Command | What it does | Uses the test set? |
|---|---|---|---|
| 1 | `python explore_data.py` | Dataset size, label balance, leakage check, shipping-mode quirk | No |
| 2 | `python tune.py` | `max_depth` validation curve (5-fold CV) | No |
| 3 | `python train.py` | Trains the Random Forest (main) and MLP (comparison); saves both | No |
| 4 | `python select_threshold.py` | Threshold analysis, precision–recall and calibration curves on out-of-fold training predictions | No |
| 5 | `python evaluate.py` | **Final evaluation, run once** after all settings are frozen | **Yes** |
| 6 | `python predict.py` | Scores the orders in `sample_orders.json` with the saved model | No |
| 7 | `python agent_demo.py --scripted` | Runs the agent test cases and demo scenarios with scripted answers | No |

To try the agent interactively (you type the approval yourself):

```bash
python agent_demo.py --only A
```

## Project structure

| Path | Purpose |
|---|---|
| `config.yaml` | Every setting: features, leakage columns, split, model settings, threshold, uncertainty band, agent numbers |
| `features.py` | Shared data preparation: leakage guard, date features, preprocessing pipeline, the single train/test split |
| `explore_data.py` | Data exploration and leakage evidence |
| `tune.py` | Random Forest `max_depth` validation curve |
| `train.py` | Training with 5-fold cross-validation |
| `select_threshold.py` | Decision-threshold and calibration analysis (training data only) |
| `evaluate.py` | Final test-set evaluation |
| `predict.py` | `RiskPredictor`: scores one order and decides whether a human must review it |
| `agent_demo.py` | Agent loop: risk score → Bayesian update → fuzzy rules → reasoning card → approval → tool call |
| `sample_orders.json` | Example orders, including one real DataCo order and four edge cases |
| `agent_scenarios.json` | Agent test cases A–F (with expected values) and demo scenarios G–L |
| `logs/` | Run logs, decision log, feedback log, incident log |
| `outputs/` | Charts and result tables |
| `outputs/preliminary_10nov/` | Outputs of an earlier test run, kept for disclosure (see below) |
| `models/` | Saved models (rebuilt by `train.py`, not committed) |

## Method summary

- **Leakage removed.** Four columns are only known after delivery (`Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`, `Order Status`) and are never model inputs. A guard in `features.py` stops the run if one appears.
- **Split.** Stratified 80/20 with `random_state: 42`: 138,212 training orders and 34,553 test orders.
- **Model choice.** `max_depth: 10` is the smallest depth where the validation score stops improving.
- **Threshold.** 0.50, chosen on training cross-validation because it sits on a stable plateau (0.38–0.70 give the same result). F1 and F2 were rejected because they select thresholds that flag 93–97% of orders.
- **Test discipline.** All choices were made on training data; the final evaluation was then run once.

## Final results (test set, 34,553 orders)

| Model | Recall (late) | Precision (late) | Accuracy |
|---|---|---|---|
| Baseline (flag every order) | 1.000 | 0.573 | 0.573 |
| **Random Forest, threshold 0.50 (operating)** | **0.590** | **0.893** | **0.725** |
| Random Forest, threshold 0.35 (comparison only) | 0.642 | 0.829 | 0.719 |
| MLP, threshold 0.50 | 0.600 | 0.875 | 0.722 |

Source: `logs/evaluate_final_20261110_234011.log`. Recall is always reported with precision, because flagging every order gives 100% recall.

## Disclosure

A preliminary test-set run at threshold 0.50 was made on 10 November (`logs/evaluate_20261110_162939.log`) before the threshold analysis. No setting was chosen from it: `max_depth`, the threshold and the uncertainty band were all chosen on training cross-validation. Its outputs are kept in `outputs/preliminary_10nov/`.

## Limitations

- About 92% of the model's feature importance rests on shipping mode and scheduled days, which carry the same information in this dataset.
- The model misses 41% of late orders at the operating threshold; these sit in the Standard Class and Same Day group, where it cannot separate late from on-time orders.
- Risk scores are slightly too low (mean predicted 0.519 against an actual late rate of 0.573 on training cross-validation).
- The data is historical and from one company, and the split is random rather than time-based.
- Reasoning-card "top features" are global importances, the same for every order.
- Truck, stock and routing-API behaviour are simulated.
