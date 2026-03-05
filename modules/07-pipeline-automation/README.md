# Module 07: A/B Testing with MLflow - Traffic Splitting and Metric Comparison

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 06 completed, understanding of model serving |

---

## Learning Objectives

By the end of this module, you will be able to:

- Implement A/B testing for ML models using traffic splitting
- Route requests between model versions based on configurable weights
- Collect and compare metrics across model variants
- Perform statistical significance testing on A/B results
- Build a FastAPI-based A/B testing router
- Make data-driven decisions about model promotion

---

## Concepts

### Why A/B Test ML Models?

Offline metrics (accuracy, F1) do not always correlate with real-world performance. A/B testing lets you compare models on live traffic before fully committing to a new version.

| Offline Metric | Online Metric |
|---|---|
| Test accuracy | Click-through rate |
| RMSE | Revenue per user |
| F1 score | User engagement |
| AUC-ROC | Conversion rate |

### A/B Testing Architecture

```
                        ┌── Model A (Production, 80%) ──┐
Client ──> AB Router ──>│                                 ├──> Response + Log
                        └── Model B (Challenger, 20%) ───┘
```

### Key Terms

| Term | Definition |
|---|---|
| **Control (A)** | The current Production model |
| **Treatment (B)** | The challenger model being tested |
| **Traffic Split** | Percentage of requests routed to each variant |
| **Statistical Significance** | Confidence that observed differences are not due to chance |
| **p-value** | Probability of observing results this extreme if there is no real difference |

---

## Hands-On Lab

### Exercise 1: Build an A/B Testing Router

**Goal:** Create a FastAPI service that routes traffic between two MLflow models.

```python
# ab_router.py
import hashlib
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MLflow A/B Testing Router", version="1.0.0")


class ABConfig(BaseModel):
    model_name: str = "wine-quality-classifier"
    model_a_stage: str = "Production"
    model_b_version: Optional[int] = None
    traffic_split: float = Field(0.8, ge=0.0, le=1.0, description="Fraction of traffic to Model A")


class PredictionRequest(BaseModel):
    features: List[float]
    user_id: Optional[str] = None


class ABPredictionResponse(BaseModel):
    prediction: int
    variant: str  # "A" or "B"
    model_version: str
    latency_ms: float
    timestamp: str


# In-memory experiment log
experiment_log: List[Dict] = []
config = ABConfig()


class ModelStore:
    """Holds loaded models for A and B variants."""
    model_a = None
    model_b = None
    model_a_info = ""
    model_b_info = ""


store = ModelStore()


def load_models():
    """Load both model variants from the MLflow registry."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Load Model A (Production)
    model_a_uri = f"models:/{config.model_name}/{config.model_a_stage}"
    store.model_a = mlflow.pyfunc.load_model(model_a_uri)
    store.model_a_info = f"{config.model_name}/{config.model_a_stage}"
    print(f"Model A loaded: {model_a_uri}")

    # Load Model B (specific version or Staging)
    if config.model_b_version:
        model_b_uri = f"models:/{config.model_name}/{config.model_b_version}"
    else:
        model_b_uri = f"models:/{config.model_name}/Staging"
    store.model_b = mlflow.pyfunc.load_model(model_b_uri)
    store.model_b_info = model_b_uri
    print(f"Model B loaded: {model_b_uri}")


def route_request(user_id: Optional[str] = None) -> str:
    """Determine which variant to use. Sticky routing by user_id if provided."""
    if user_id:
        # Consistent hashing ensures same user always gets same variant
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "A" if (hash_value % 100) < (config.traffic_split * 100) else "B"
    else:
        return "A" if np.random.random() < config.traffic_split else "B"


@app.on_event("startup")
async def startup():
    load_models()


@app.post("/predict", response_model=ABPredictionResponse)
async def predict(request: PredictionRequest):
    variant = route_request(request.user_id)

    start = time.time()

    if variant == "A":
        if not store.model_a:
            raise HTTPException(503, "Model A not loaded")
        input_df = pd.DataFrame([request.features])
        prediction = int(store.model_a.predict(input_df)[0])
        model_info = store.model_a_info
    else:
        if not store.model_b:
            raise HTTPException(503, "Model B not loaded")
        input_df = pd.DataFrame([request.features])
        prediction = int(store.model_b.predict(input_df)[0])
        model_info = store.model_b_info

    latency_ms = (time.time() - start) * 1000

    # Log the result for later analysis
    experiment_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "variant": variant,
        "user_id": request.user_id,
        "prediction": prediction,
        "latency_ms": latency_ms,
        "model_info": model_info,
    })

    return ABPredictionResponse(
        prediction=prediction,
        variant=variant,
        model_version=model_info,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ab/stats")
async def get_ab_stats():
    """Return current A/B experiment statistics."""
    if not experiment_log:
        return {"message": "No data collected yet"}

    a_logs = [l for l in experiment_log if l["variant"] == "A"]
    b_logs = [l for l in experiment_log if l["variant"] == "B"]

    return {
        "total_requests": len(experiment_log),
        "variant_a": {
            "count": len(a_logs),
            "avg_latency_ms": round(np.mean([l["latency_ms"] for l in a_logs]), 2) if a_logs else 0,
            "model": store.model_a_info,
        },
        "variant_b": {
            "count": len(b_logs),
            "avg_latency_ms": round(np.mean([l["latency_ms"] for l in b_logs]), 2) if b_logs else 0,
            "model": store.model_b_info,
        },
        "traffic_split": f"{config.traffic_split:.0%} A / {1 - config.traffic_split:.0%} B",
    }


@app.post("/ab/config")
async def update_config(new_config: ABConfig):
    """Update A/B testing configuration and reload models."""
    global config
    config = new_config
    load_models()
    return {"status": "updated", "config": config.dict()}


@app.get("/ab/export")
async def export_logs():
    """Export experiment logs for offline analysis."""
    return experiment_log


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Exercise 2: Simulate Traffic and Collect Data

**Goal:** Send simulated traffic through the A/B router and collect results.

```python
# exercise2_simulate_traffic.py
import random
import time

import numpy as np
import requests
from sklearn.datasets import load_wine

BASE_URL = "http://localhost:8000"
wine = load_wine()

# Simulate 500 requests from 100 unique users
n_requests = 500
n_users = 100
user_ids = [f"user_{i:04d}" for i in range(n_users)]

print(f"Sending {n_requests} requests from {n_users} users...")

results = []
for i in range(n_requests):
    sample_idx = random.randint(0, len(wine.data) - 1)
    features = wine.data[sample_idx].tolist()
    user_id = random.choice(user_ids)

    resp = requests.post(f"{BASE_URL}/predict", json={
        "features": features,
        "user_id": user_id,
    })

    if resp.status_code == 200:
        data = resp.json()
        results.append(data)
        # Simulate real user feedback (ground truth)
        actual_class = wine.target[sample_idx]
        results[-1]["actual"] = int(actual_class)
        results[-1]["correct"] = int(data["prediction"] == actual_class)

    if (i + 1) % 100 == 0:
        print(f"  Sent {i + 1}/{n_requests} requests")

# Analyze results
a_results = [r for r in results if r["variant"] == "A"]
b_results = [r for r in results if r["variant"] == "B"]

a_accuracy = np.mean([r["correct"] for r in a_results]) if a_results else 0
b_accuracy = np.mean([r["correct"] for r in b_results]) if b_results else 0
a_latency = np.mean([r["latency_ms"] for r in a_results]) if a_results else 0
b_latency = np.mean([r["latency_ms"] for r in b_results]) if b_results else 0

print(f"\nResults:")
print(f"  Model A: {len(a_results)} requests, accuracy={a_accuracy:.4f}, latency={a_latency:.1f}ms")
print(f"  Model B: {len(b_results)} requests, accuracy={b_accuracy:.4f}, latency={b_latency:.1f}ms")
```

### Exercise 3: Statistical Significance Testing

**Goal:** Determine if the difference between variants is statistically significant.

```python
# exercise3_significance.py
import numpy as np
from scipy import stats


def ab_significance_test(
    a_successes: int, a_total: int,
    b_successes: int, b_total: int,
    alpha: float = 0.05,
) -> dict:
    """
    Run a two-proportion z-test to compare A/B variants.

    H0: p_A = p_B (no difference)
    H1: p_A != p_B (there is a difference)
    """
    p_a = a_successes / a_total
    p_b = b_successes / b_total

    # Pooled proportion
    p_pool = (a_successes + b_successes) / (a_total + b_total)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / a_total + 1 / b_total))

    # Z-statistic
    z_stat = (p_a - p_b) / se if se > 0 else 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval for the difference
    se_diff = np.sqrt(p_a * (1 - p_a) / a_total + p_b * (1 - p_b) / b_total)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = (p_a - p_b) - z_crit * se_diff
    ci_upper = (p_a - p_b) + z_crit * se_diff

    significant = p_value < alpha

    return {
        "variant_a_rate": round(p_a, 4),
        "variant_b_rate": round(p_b, 4),
        "absolute_difference": round(p_a - p_b, 4),
        "relative_lift": round((p_b - p_a) / p_a * 100, 2) if p_a > 0 else 0,
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
        "significant": significant,
        "recommendation": "Adopt B" if significant and p_b > p_a else
                          "Keep A" if significant else "Continue testing",
    }


# Example: Model A got 312 correct out of 400, Model B got 85 out of 100
result = ab_significance_test(
    a_successes=312, a_total=400,
    b_successes=85, b_total=100,
    alpha=0.05,
)

print("A/B Test Results:")
for key, value in result.items():
    print(f"  {key}: {value}")
```

### Exercise 4: Log A/B Results to MLflow

**Goal:** Track A/B test experiments in MLflow for historical record.

```python
# exercise4_log_ab_results.py
import mlflow
import requests

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("07-ab-testing-results")

# Get stats from the A/B router
stats = requests.get("http://localhost:8000/ab/stats").json()

with mlflow.start_run(run_name="ab-test-wine-classifier"):
    # Log configuration
    mlflow.log_params({
        "model_a": stats.get("variant_a", {}).get("model", "unknown"),
        "model_b": stats.get("variant_b", {}).get("model", "unknown"),
        "traffic_split": stats.get("traffic_split", "unknown"),
        "total_requests": stats.get("total_requests", 0),
    })

    # Log metrics for each variant
    a_data = stats.get("variant_a", {})
    b_data = stats.get("variant_b", {})

    mlflow.log_metrics({
        "variant_a_count": a_data.get("count", 0),
        "variant_a_avg_latency_ms": a_data.get("avg_latency_ms", 0),
        "variant_b_count": b_data.get("count", 0),
        "variant_b_avg_latency_ms": b_data.get("avg_latency_ms", 0),
    })

    mlflow.set_tag("test_type", "A/B")
    mlflow.set_tag("status", "completed")

    # Save full logs as artifact
    logs = requests.get("http://localhost:8000/ab/export").json()
    import json
    with open("ab_logs.json", "w") as f:
        json.dump(logs, f, indent=2)
    mlflow.log_artifact("ab_logs.json")

    print("A/B test results logged to MLflow!")
```

---

## A/B Testing Best Practices

1. **Sample size matters** -- Run the test long enough to reach statistical significance
2. **Sticky routing** -- Same user should always see the same variant (use user ID hashing)
3. **Monitor both variants** -- Track latency, error rates, and business metrics
4. **Have a rollback plan** -- Be ready to route 100% traffic to A if B causes problems
5. **One change at a time** -- Only test one model change per experiment
6. **Document everything** -- Log configs, results, and decisions in MLflow

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Too small sample size | Inconclusive results | Calculate required sample size before starting |
| Not using sticky routing | Inconsistent user experience | Hash user ID for consistent variant assignment |
| Checking results too early | False positives | Wait for calculated minimum sample size |
| Testing multiple changes at once | Cannot attribute improvement | Change one variable at a time |

---

## Self-Check Questions

1. Why is A/B testing important even when offline metrics look good?
2. What is sticky routing and why is it needed?
3. How do you determine if an A/B test result is statistically significant?
4. What is the difference between absolute difference and relative lift?
5. When should you stop an A/B test early?

---

## You Know You Have Completed This Module When...

- [ ] You built and ran the A/B testing router
- [ ] You simulated traffic and collected results for both variants
- [ ] You performed a statistical significance test
- [ ] You logged A/B results to MLflow
- [ ] You can explain when to adopt variant B vs continue testing
- [ ] Validation script passes: `bash modules/07-pipeline-automation/validation/validate.sh`

---

**Next: [Module 08 - A/B Testing Models -->](../08-ab-testing/)**
