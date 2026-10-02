# AI-Based Autonomous Cloud Management for Retail Systems

A simple base project showing a MAPE-K (Monitor, Analyze, Plan, Execute) autonomous loop
that scales cloud servers for a retail platform using ML demand forecasting and anomaly detection.

## Run
    pip install -r requirements.txt
    python main.py

## Outputs
- results.csv                  : cost / utilisation / SLA comparison
- cloud_management_results.png : demand vs forecast and server counts

## Components
- Retail traffic simulator (daily/weekly patterns + flash sales)
- RandomForest demand forecaster
- IsolationForest anomaly detector
- Autonomous scaler vs static and reactive baselines
