"""
AI-Based Autonomous Cloud Management for Retail Systems
=======================================================
A simple base project that demonstrates a MAPE-K style autonomous loop
(Monitor -> Analyze -> Plan -> Execute) for managing cloud servers of a
retail platform (web store / checkout / inventory APIs).

Components
----------
1. Retail traffic simulator   : hourly requests with daily/weekly patterns + flash sales
2. Demand forecaster (ML)     : RandomForest predicts next-hour requests
3. Anomaly detector (ML)      : IsolationForest flags unusual forecast errors (surprise spikes)
4. Autonomous scaler          : decides number of servers every hour
5. Baselines                  : static provisioning and reactive (threshold-style) scaling
6. Evaluation                 : cost, SLA violations, average utilisation + plots

Run:
    pip install numpy pandas scikit-learn matplotlib
    python autonomous_cloud_retail.py
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
DAYS = 28                    # total simulated days
TRAIN_DAYS = 14              # forecaster training window
VAL_DAYS = 7                 # used to learn "normal" forecast error
# remaining days = live run (autonomous control)

CAPACITY_PER_SERVER = 500    # requests/hour one server can handle
COST_PER_SERVER_HOUR = 0.50  # USD
TARGET_UTIL = 0.70           # desired utilisation
MIN_SERVERS, MAX_SERVERS = 2, 120
SCALE_DOWN_STEP = 3          # max servers removed per hour (avoids flapping)
SAFETY_MARGIN = 0.10         # extra headroom on top of forecast
ANOMALY_BOOST = 0.35         # extra capacity when anomaly detected
BASE_LATENCY_MS = 80
SLA_LATENCY_MS = 300         # SLA violated if latency exceeds this


# --------------------------------------------------------------------------
# 1. Retail traffic simulator
# --------------------------------------------------------------------------
def simulate_retail_traffic(days=DAYS, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    hour = idx.hour.values
    dow = idx.dayofweek.values

    # Daily shopping pattern: low at night, peaks at lunch and evening
    daily = (
        0.35
        + 0.45 * np.exp(-((hour - 13) ** 2) / 14)
        + 0.70 * np.exp(-((hour - 20) ** 2) / 8)
    )
    weekend = np.where(dow >= 5, 1.25, 1.0)
    base = 9000 * daily * weekend
    noise = rng.normal(1.0, 0.06, len(idx))
    demand = base * noise

    # Flash sales / surprise spikes (unpredictable events)
    n_spikes = days // 5
    for _ in range(n_spikes):
        start = rng.integers(24, len(idx) - 5)
        length = rng.integers(2, 5)
        demand[start:start + length] *= rng.uniform(1.8, 2.8)

    return pd.DataFrame({"requests": demand.astype(float)}, index=idx)


# --------------------------------------------------------------------------
# 2. Demand forecaster
# --------------------------------------------------------------------------
def make_features(series: np.ndarray, t: int, index: pd.DatetimeIndex):
    """Features for predicting demand at hour t using only data before t."""
    return [
        index[t].hour,
        index[t].dayofweek,
        series[t - 1],
        series[t - 2],
        series[t - 24],
        np.mean(series[t - 3:t]),
    ]


class DemandForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=SEED, n_jobs=-1
        )

    def fit(self, df: pd.DataFrame, start: int, end: int):
        X, y = [], []
        for t in range(max(start, 24), end):
            X.append(make_features(df["requests"].values, t, df.index))
            y.append(df["requests"].values[t])
        self.model.fit(np.array(X), np.array(y))

    def predict(self, df: pd.DataFrame, t: int) -> float:
        x = np.array(make_features(df["requests"].values, t, df.index)).reshape(1, -1)
        return float(self.model.predict(x)[0])


# --------------------------------------------------------------------------
# 3. Anomaly detector (on forecast error)
# --------------------------------------------------------------------------
class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.03, random_state=SEED)

    def fit(self, errors_ratio: np.ndarray):
        self.model.fit(errors_ratio.reshape(-1, 1))

    def is_anomaly(self, error_ratio: float) -> bool:
        # Only flag under-prediction (demand higher than expected) as dangerous
        return error_ratio > 0 and self.model.predict([[error_ratio]])[0] == -1


# --------------------------------------------------------------------------
# Cloud environment model
# --------------------------------------------------------------------------
def evaluate_hour(demand: float, servers: int):
    capacity = servers * CAPACITY_PER_SERVER
    util = demand / capacity
    latency = BASE_LATENCY_MS + 400 * min(util, 2.0) ** 4
    violated = latency > SLA_LATENCY_MS
    cost = servers * COST_PER_SERVER_HOUR
    return util, latency, violated, cost


def servers_needed(demand: float, util_target=TARGET_UTIL) -> int:
    n = int(np.ceil(demand / (CAPACITY_PER_SERVER * util_target)))
    return int(np.clip(n, MIN_SERVERS, MAX_SERVERS))


# --------------------------------------------------------------------------
# 4. Scaling strategies
# --------------------------------------------------------------------------
def run_static(df, start, train_end):
    n = servers_needed(df["requests"].values[:train_end].max(), util_target=0.85)
    return [n] * (len(df) - start)


def run_reactive(df, start):
    """Scales according to the PREVIOUS hour's demand (always one step late)."""
    return [servers_needed(df["requests"].values[t - 1]) for t in range(start, len(df))]


def run_autonomous(df, start, forecaster, detector):
    """MAPE-K loop executed every hour."""
    demand = df["requests"].values
    servers = servers_needed(demand[start - 1])
    plan, log = [], []
    boost = 0.0

    for t in range(start, len(df)):
        # MONITOR  : previous hour's actual demand vs. what we predicted
        # ANALYZE  : forecast next hour + detect anomaly
        pred = forecaster.predict(df, t)
        if t > start:
            err_ratio = (demand[t - 1] - prev_pred) / max(prev_pred, 1)
            boost = ANOMALY_BOOST if detector.is_anomaly(err_ratio) else 0.0
        # PLAN     : required servers with safety margin (+ anomaly boost)
        target = servers_needed(pred * (1 + SAFETY_MARGIN + boost))
        # EXECUTE  : scale up immediately, scale down gradually
        if target >= servers:
            servers = target
        else:
            servers = max(target, servers - SCALE_DOWN_STEP)

        plan.append(servers)
        log.append((df.index[t], pred, boost > 0, servers))
        prev_pred = pred

    return plan, log


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------
def summarize(name, df, start, plan):
    demand = df["requests"].values[start:]
    rows = [evaluate_hour(d, s) for d, s in zip(demand, plan)]
    util, lat, viol, cost = map(np.array, zip(*rows))
    return {
        "Strategy": name,
        "Total Cost ($)": round(cost.sum(), 2),
        "Avg Utilisation (%)": round(100 * util.mean(), 1),
        "SLA Violations (hrs)": int(viol.sum()),
        "SLA Violation (%)": round(100 * viol.mean(), 1),
        "Avg Latency (ms)": round(lat.mean(), 1),
    }


def main():
    print("Simulating retail traffic ...")
    df = simulate_retail_traffic()
    train_end = TRAIN_DAYS * 24
    val_end = (TRAIN_DAYS + VAL_DAYS) * 24
    start = val_end                                  # live phase starts here

    print("Training demand forecaster ...")
    forecaster = DemandForecaster()
    forecaster.fit(df, 24, train_end)

    print("Learning normal forecast behaviour for anomaly detection ...")
    ratios = []
    for t in range(train_end, val_end):
        p = forecaster.predict(df, t)
        ratios.append((df["requests"].values[t] - p) / max(p, 1))
    detector = AnomalyDetector()
    detector.fit(np.array(ratios))

    print("Running autonomous management loop ...\n")
    auto_plan, log = run_autonomous(df, start, forecaster, detector)
    static_plan = run_static(df, start, train_end)
    reactive_plan = run_reactive(df, start)

    results = pd.DataFrame([
        summarize("Static (peak provisioning)", df, start, static_plan),
        summarize("Reactive (threshold)", df, start, reactive_plan),
        summarize("AI Autonomous", df, start, auto_plan),
    ])
    print(results.to_string(index=False))
    results.to_csv("results.csv", index=False)

    # ---- Plots ----
    live_idx = df.index[start:]
    demand_live = df["requests"].values[start:]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(live_idx, demand_live, color="black", label="Actual demand (req/hr)")
    axes[0].plot(live_idx, [l[1] for l in log], "--", color="tab:blue", label="Forecast")
    anom = [l[0] for l in log if l[2]]
    if anom:
        axes[0].scatter(anom, df.loc[anom, "requests"], color="red", zorder=5,
                        label="Anomaly detected")
    axes[0].set_ylabel("Requests / hour")
    axes[0].set_title("Retail Demand vs AI Forecast")
    axes[0].legend()

    axes[1].step(live_idx, static_plan, label="Static", alpha=0.6)
    axes[1].step(live_idx, reactive_plan, label="Reactive", alpha=0.6)
    axes[1].step(live_idx, auto_plan, label="AI Autonomous", linewidth=2, color="green")
    axes[1].set_ylabel("Active servers")
    axes[1].set_title("Provisioned Servers by Strategy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cloud_management_results.png", dpi=130)
    print("\nSaved: results.csv, cloud_management_results.png")


if __name__ == "__main__":
    main()
