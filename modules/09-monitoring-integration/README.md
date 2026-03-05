# Module 09: Production Monitoring - Model Performance Tracking and Drift Detection

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 08 completed, understanding of model serving |

---

## Learning Objectives

By the end of this module, you will be able to:

- Monitor model performance metrics in production
- Detect data drift using statistical tests
- Detect concept drift (model performance degradation over time)
- Log monitoring results to MLflow for historical tracking
- Build automated alerts for drift and performance drops
- Create a monitoring dashboard with MLflow

---

## Concepts

### Why Monitor ML Models?

Models degrade over time because the real world changes:

| Scenario | Type of Drift | Example |
|---|---|---|
| Input data distribution changes | **Data drift** | Customer demographics shift after marketing campaign |
| Relationship between features and target changes | **Concept drift** | Fraud patterns evolve as criminals adapt |
| Model's predictions become less accurate | **Performance degradation** | Recommendation model gets stale |
| Feature pipeline breaks | **Data quality issue** | Upstream ETL starts sending nulls |

### Types of Drift

```
Data Drift:     P(X) changes        (features shift)
Concept Drift:  P(Y|X) changes      (relationship between features and target changes)
Prediction Drift: P(Y_hat) changes  (model's output distribution shifts)
```

### Monitoring Architecture

```
Production Traffic
       │
       v
  ┌──────────┐     ┌──────────────┐     ┌──────────────┐
  │  Model    │ --> │  Prediction  │ --> │  Monitor     │
  │  Server   │     │  Logger      │     │  (scheduled) │
  └──────────┘     └──────────────┘     └──────────────┘
                                               │
                                               v
                                        ┌──────────────┐
                                        │  MLflow      │
                                        │  (metrics)   │
                                        └──────────────┘
                                               │
                                               v
                                        ┌──────────────┐
                                        │  Alert       │
                                        │  (if drift)  │
                                        └──────────────┘
```

---

## Hands-On Lab

### Exercise 1: Build a Performance Monitor

**Goal:** Track model accuracy over time using a sliding window approach.

```python
# exercise1_performance_monitor.py
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("09-performance-monitoring")


class PerformanceMonitor:
    """Track model performance over time with sliding windows."""

    def __init__(self, model, baseline_accuracy: float, alert_threshold: float = 0.05):
        self.model = model
        self.baseline_accuracy = baseline_accuracy
        self.alert_threshold = alert_threshold
        self.history = []

    def evaluate_batch(self, X_batch, y_batch, batch_id: str):
        """Evaluate model on a new batch of data and log to MLflow."""
        y_pred = self.model.predict(X_batch)

        metrics = {
            "batch_accuracy": accuracy_score(y_batch, y_pred),
            "batch_precision": precision_score(y_batch, y_pred, average="weighted"),
            "batch_recall": recall_score(y_batch, y_pred, average="weighted"),
            "batch_f1": f1_score(y_batch, y_pred, average="weighted"),
            "batch_size": len(y_batch),
        }

        # Calculate degradation from baseline
        degradation = self.baseline_accuracy - metrics["batch_accuracy"]
        metrics["degradation_from_baseline"] = degradation
        metrics["alert_triggered"] = 1.0 if degradation > self.alert_threshold else 0.0

        self.history.append({
            "batch_id": batch_id,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        })

        return metrics

    def get_rolling_metrics(self, window_size: int = 5):
        """Calculate rolling average metrics over the last N batches."""
        if len(self.history) < window_size:
            return None

        recent = self.history[-window_size:]
        return {
            "rolling_accuracy": np.mean([h["batch_accuracy"] for h in recent]),
            "rolling_f1": np.mean([h["batch_f1"] for h in recent]),
            "rolling_degradation": np.mean([h["degradation_from_baseline"] for h in recent]),
            "alerts_in_window": sum(h["alert_triggered"] for h in recent),
        }


# Simulate production monitoring
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

monitor = PerformanceMonitor(model, baseline_accuracy, alert_threshold=0.05)

with mlflow.start_run(run_name="performance-monitoring"):
    mlflow.log_param("baseline_accuracy", round(baseline_accuracy, 4))
    mlflow.log_param("alert_threshold", 0.05)
    mlflow.log_param("n_batches", 20)

    # Simulate 20 batches of production data with increasing noise
    rng = np.random.RandomState(42)

    for batch_idx in range(20):
        # Simulate data that gradually drifts
        noise_level = batch_idx * 0.02
        X_batch = X_test + rng.normal(0, noise_level, X_test.shape)
        y_batch = y_test

        batch_id = f"batch_{batch_idx:03d}"
        metrics = monitor.evaluate_batch(X_batch, y_batch, batch_id)

        # Log each batch as a step
        mlflow.log_metrics({
            "accuracy": metrics["batch_accuracy"],
            "f1_score": metrics["batch_f1"],
            "degradation": metrics["degradation_from_baseline"],
            "alert": metrics["alert_triggered"],
        }, step=batch_idx)

        # Log rolling metrics after enough history
        rolling = monitor.get_rolling_metrics(window_size=5)
        if rolling:
            mlflow.log_metrics({
                "rolling_accuracy": rolling["rolling_accuracy"],
                "rolling_f1": rolling["rolling_f1"],
            }, step=batch_idx)

        status = "ALERT" if metrics["alert_triggered"] else "OK"
        print(f"Batch {batch_idx:2d}: accuracy={metrics['batch_accuracy']:.4f} "
              f"degradation={metrics['degradation_from_baseline']:+.4f} [{status}]")

    # Log final summary
    final_rolling = monitor.get_rolling_metrics(5)
    if final_rolling:
        mlflow.log_metrics({
            "final_rolling_accuracy": final_rolling["rolling_accuracy"],
            "total_alerts": final_rolling["alerts_in_window"],
        })

    # Save history as artifact
    history_df = pd.DataFrame(monitor.history)
    history_df.to_csv("monitoring_history.csv", index=False)
    mlflow.log_artifact("monitoring_history.csv")
```

### Exercise 2: Data Drift Detection

**Goal:** Detect when incoming data distribution shifts from the training distribution.

```python
# exercise2_data_drift.py
import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_wine

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("09-data-drift-detection")


class DataDriftDetector:
    """Detect data drift using statistical tests."""

    def __init__(self, reference_data: pd.DataFrame, feature_names: list):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, data: pd.DataFrame) -> dict:
        """Compute summary statistics for each feature."""
        return {
            col: {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "median": data[col].median(),
            }
            for col in self.feature_names
        }

    def detect_drift(self, current_data: pd.DataFrame, significance_level: float = 0.05) -> dict:
        """
        Run drift detection tests on each feature.

        Uses the Kolmogorov-Smirnov test to compare distributions.
        """
        results = {}

        for feature in self.feature_names:
            ref_values = self.reference_data[feature].values
            cur_values = current_data[feature].values

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)

            # Population Stability Index (PSI)
            psi = self._calculate_psi(ref_values, cur_values)

            # Mean shift
            ref_mean = ref_values.mean()
            cur_mean = cur_values.mean()
            ref_std = ref_values.std()
            mean_shift = abs(cur_mean - ref_mean) / ref_std if ref_std > 0 else 0

            drifted = ks_pvalue < significance_level

            results[feature] = {
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 6),
                "psi": round(psi, 4),
                "mean_shift_std": round(mean_shift, 4),
                "drifted": drifted,
                "ref_mean": round(ref_mean, 4),
                "cur_mean": round(cur_mean, 4),
            }

        return results

    @staticmethod
    def _calculate_psi(reference, current, n_bins=10):
        """Calculate Population Stability Index."""
        ref_min = min(reference.min(), current.min())
        ref_max = max(reference.max(), current.max())
        bins = np.linspace(ref_min, ref_max, n_bins + 1)

        ref_counts = np.histogram(reference, bins=bins)[0] + 1
        cur_counts = np.histogram(current, bins=bins)[0] + 1

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return psi


# Run drift detection
wine = load_wine()
feature_names = wine.feature_names

reference_df = pd.DataFrame(wine.data[:120], columns=feature_names)

# Simulate drifted data: shift some features
rng = np.random.RandomState(42)
drifted_data = wine.data[120:].copy()
drifted_data[:, 0] += 2.0     # Shift alcohol
drifted_data[:, 1] *= 1.5     # Scale malic_acid
drifted_data[:, 5] += rng.normal(0, 0.5, drifted_data.shape[0])  # Add noise to total_phenols
current_df = pd.DataFrame(drifted_data, columns=feature_names)

detector = DataDriftDetector(reference_df, feature_names)

with mlflow.start_run(run_name="drift-detection"):
    results = detector.detect_drift(current_df)

    n_drifted = sum(1 for r in results.values() if r["drifted"])
    total = len(results)

    mlflow.log_metric("n_features_drifted", n_drifted)
    mlflow.log_metric("drift_ratio", n_drifted / total)

    print(f"Drift Detection Results ({n_drifted}/{total} features drifted):\n")

    for feature, result in results.items():
        status = "DRIFTED" if result["drifted"] else "OK"
        mlflow.log_metric(f"ks_stat_{feature}", result["ks_statistic"])
        mlflow.log_metric(f"psi_{feature}", result["psi"])

        if result["drifted"]:
            print(f"  {feature}: KS={result['ks_statistic']:.4f}, "
                  f"p={result['ks_pvalue']:.6f}, PSI={result['psi']:.4f} [{status}]")

    # Save detailed results
    results_df = pd.DataFrame(results).T
    results_df.to_csv("drift_results.csv")
    mlflow.log_artifact("drift_results.csv")

    mlflow.set_tag("drift_detected", "true" if n_drifted > 0 else "false")
```

### Exercise 3: Automated Monitoring Pipeline

**Goal:** Build a scheduled monitoring job that checks both performance and drift.

```python
# exercise3_monitoring_pipeline.py
import json
import time
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")


class MonitoringPipeline:
    """Complete monitoring pipeline: performance + drift + alerting."""

    def __init__(
        self,
        model_name: str,
        reference_data: pd.DataFrame,
        reference_labels: np.ndarray,
        feature_names: list,
        accuracy_threshold: float = 0.85,
        drift_threshold: float = 3,  # number of drifted features to trigger alert
    ):
        self.model_name = model_name
        self.reference_data = reference_data
        self.reference_labels = reference_labels
        self.feature_names = feature_names
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold

    def run_check(self, current_data, current_labels, model, check_id: str):
        """Run a complete monitoring check."""
        report = {
            "check_id": check_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": self.model_name,
            "alerts": [],
        }

        # 1. Performance check
        predictions = model.predict(current_data)
        accuracy = accuracy_score(current_labels, predictions)
        report["accuracy"] = accuracy

        if accuracy < self.accuracy_threshold:
            report["alerts"].append({
                "type": "performance_degradation",
                "severity": "high",
                "message": f"Accuracy dropped to {accuracy:.4f} (threshold: {self.accuracy_threshold})",
            })

        # 2. Drift check
        current_df = pd.DataFrame(current_data, columns=self.feature_names)
        n_drifted = 0

        for i, feature in enumerate(self.feature_names):
            ks_stat, ks_p = stats.ks_2samp(
                self.reference_data[feature].values,
                current_df[feature].values,
            )
            if ks_p < 0.05:
                n_drifted += 1

        report["features_drifted"] = n_drifted

        if n_drifted >= self.drift_threshold:
            report["alerts"].append({
                "type": "data_drift",
                "severity": "medium",
                "message": f"{n_drifted} features show significant drift",
            })

        # 3. Prediction distribution check
        ref_pred = model.predict(self.reference_data.values)
        ks_pred, ks_pred_p = stats.ks_2samp(ref_pred.astype(float), predictions.astype(float))
        report["prediction_drift_pvalue"] = ks_pred_p

        if ks_pred_p < 0.05:
            report["alerts"].append({
                "type": "prediction_drift",
                "severity": "medium",
                "message": f"Prediction distribution shifted (p={ks_pred_p:.6f})",
            })

        report["status"] = "alert" if report["alerts"] else "healthy"
        return report


# Run the pipeline
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

reference_df = pd.DataFrame(X_test[:30], columns=wine.feature_names)
reference_labels = y_test[:30]

pipeline = MonitoringPipeline(
    model_name="wine-classifier",
    reference_data=reference_df,
    reference_labels=reference_labels,
    feature_names=list(wine.feature_names),
    accuracy_threshold=0.85,
    drift_threshold=3,
)

mlflow.set_experiment("09-monitoring-pipeline")

with mlflow.start_run(run_name="monitoring-pipeline"):
    rng = np.random.RandomState(42)

    for check_idx in range(10):
        # Simulate increasingly drifted data
        noise = check_idx * 0.03
        batch_data = X_test[30:] + rng.normal(0, noise, X_test[30:].shape)
        batch_labels = y_test[30:]

        report = pipeline.run_check(batch_data, batch_labels, model, f"check_{check_idx:03d}")

        mlflow.log_metrics({
            "check_accuracy": report["accuracy"],
            "check_features_drifted": report["features_drifted"],
            "check_n_alerts": len(report["alerts"]),
        }, step=check_idx)

        status = report["status"].upper()
        print(f"Check {check_idx}: accuracy={report['accuracy']:.4f}, "
              f"drifted={report['features_drifted']}, alerts={len(report['alerts'])} [{status}]")

        for alert in report["alerts"]:
            print(f"    [{alert['severity'].upper()}] {alert['message']}")

    # Save all reports
    with open("monitoring_reports.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("monitoring_reports.json")
```

---

## Monitoring Checklist for Production

| Check | Frequency | Alert Threshold |
|---|---|---|
| Model accuracy on labeled samples | Daily/Weekly | Drop > 5% from baseline |
| Feature distribution (KS test) | Daily | p-value < 0.05 for > 3 features |
| Prediction distribution | Hourly | p-value < 0.01 |
| Latency P99 | Real-time | > 200ms (or your SLA) |
| Error rate | Real-time | > 1% |
| Data quality (nulls, outliers) | Per batch | Any nulls in required features |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not monitoring at all | Model silently degrades | Set up at minimum daily accuracy checks |
| Monitoring only accuracy | Miss drift before it affects accuracy | Monitor feature distributions too |
| Too sensitive alerts | Alert fatigue | Tune thresholds based on your data |
| Not logging baselines | Cannot measure degradation | Always log training-time metrics as baseline |

---

## Self-Check Questions

1. What is the difference between data drift and concept drift?
2. How does the Kolmogorov-Smirnov test detect drift?
3. What is Population Stability Index (PSI) and when would you use it?
4. Why is monitoring prediction distribution useful even without ground truth?
5. How would you set up automated retraining triggered by drift detection?

---

## You Know You Have Completed This Module When...

- [ ] You built a performance monitor that tracks accuracy over batches
- [ ] You implemented data drift detection using the KS test
- [ ] You ran an automated monitoring pipeline with alerting
- [ ] You logged monitoring results and alerts to MLflow
- [ ] You can explain when to retrain vs when to investigate further
- [ ] Validation script passes: `bash modules/09-monitoring-integration/validation/validate.sh`

---

**Next: [Module 10 - Production MLflow Deployment -->](../10-production-mlflow/)**
