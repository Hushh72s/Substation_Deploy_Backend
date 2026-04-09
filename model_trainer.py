"""
model_trainer.py
-----------------
Run this ONCE locally to generate model.pkl, then commit model.pkl to the repo.
Trains IsolationForest on 4200 synthetic substation readings.

Usage:
    python model_trainer.py
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

RANDOM_SEED = 42
N_NORMAL    = 4000
N_ANOMALY   = 200

np.random.seed(RANDOM_SEED)

# ── Normal data ─────────────────────────────────────────────────
normal_data = np.column_stack([
    np.random.normal(55,  12, N_NORMAL).clip(20, 85),   # temperature
    np.random.normal(1.2, 0.5, N_NORMAL).clip(0.1, 3.0), # vibration
    np.random.normal(225, 8, N_NORMAL).clip(210, 240),   # voltage
    np.random.normal(50,  12, N_NORMAL).clip(25, 75),    # humidity
])

# ── Anomaly data ────────────────────────────────────────────────
n = N_ANOMALY // 4
anomaly_data = np.vstack([
    np.column_stack([np.random.uniform(92, 130, n), np.random.uniform(0.1, 3.0, n), np.random.uniform(210, 240, n), np.random.uniform(25, 75, n)]),
    np.column_stack([np.random.uniform(20, 85, n),  np.random.uniform(4.0, 8.0, n), np.random.uniform(210, 240, n), np.random.uniform(25, 75, n)]),
    np.column_stack([np.random.uniform(20, 85, n),  np.random.uniform(0.1, 3.0, n), np.concatenate([np.random.uniform(180, 204, n//2), np.random.uniform(246, 260, n//2)]), np.random.uniform(25, 75, n)]),
    np.column_stack([np.random.uniform(20, 85, n),  np.random.uniform(0.1, 3.0, n), np.random.uniform(210, 240, n), np.random.uniform(87, 100, n)]),
])

training_data = np.vstack([normal_data, anomaly_data])

model = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_SEED, max_samples="auto")
print(f"Training IsolationForest on {len(training_data)} samples …")
model.fit(training_data)
joblib.dump(model, "model.pkl")
print("✅  model.pkl saved.")

normal_acc  = (model.predict(normal_data)  == 1).mean() * 100
anomaly_acc = (model.predict(anomaly_data) == -1).mean() * 100
print(f"   Normal accuracy : {normal_acc:.1f}%")
print(f"   Anomaly accuracy: {anomaly_acc:.1f}%")
