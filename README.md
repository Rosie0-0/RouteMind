# RouteMind – Late-Delivery Risk Engine

ML engine for the 5023CMD CW2 group project (Agent-driven Logistics and Dynamic Supply Chains).

RouteMind is an autonomous logistics agent. Its core ML component is a **Random Forest classifier** that predicts whether an order will be delivered late. The predicted risk drives the agent's next action: continue, reroute, reorder, switch supplier, or escalate to a human supervisor.

## Dataset
**DataCo Smart Supply Chain for Big Data Analysis** (Kaggle)
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Download `DataCoSupplyChainDataset.csv` and place it in `data/`. It is not stored in this repo.

## Setup
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure
| Path | Purpose |
|---|---|
| `config.yaml` | All data, split, model and evaluation settings |
| `train.py` | Train and evaluate the Random Forest, save the model |
| `predict.py` | Score a new order with the saved model |
| `agent_demo.py` | Agent loop: ML risk → Bayesian update → fuzzy rules → action |
| `logs/` | Timestamped training and agent logs |
| `outputs/` | Charts and evaluation results |
| `models/` | Saved model (rebuilt by `train.py`, not committed) |

## Run
```bash
python train.py
python predict.py
python agent_demo.py
```
