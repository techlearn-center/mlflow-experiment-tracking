# Module 06: Model Serving - MLflow Serve and Custom FastAPI Wrapper

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 05 completed, a model registered in Production stage |

---

## Learning Objectives

By the end of this module, you will be able to:

- Serve models using `mlflow models serve` (built-in REST endpoint)
- Build a custom FastAPI serving wrapper with health checks and batch prediction
- Compare built-in vs custom serving approaches
- Handle input validation using model signatures
- Deploy a serving endpoint with Docker
- Test serving endpoints with curl and Python requests

---

## Concepts

### Two Approaches to Serving MLflow Models

| Approach | Pros | Cons |
|---|---|---|
| **`mlflow models serve`** | Zero code, instant deployment, auto-input validation | Limited customization, single model, no custom endpoints |
| **Custom FastAPI wrapper** | Full control, multiple models, A/B testing, custom logic | More code to maintain, you handle input validation |

### Built-in Serving Architecture

```
Client ──HTTP POST──> MLflow Serve (port 5001) ──> Model.predict() ──> JSON Response
                      (auto-generated REST API)
```

### Custom Serving Architecture

```
Client ──HTTP POST──> FastAPI (port 8000) ──> Preprocessing ──> Model.predict()
              │                                                       │
              ├── GET /health                                         v
              ├── GET /model/info                              Postprocessing
              ├── POST /predict                                       │
              └── POST /predict/batch                          JSON Response
```

---

## Hands-On Lab

### Exercise 1: Serve with `mlflow models serve`

**Goal:** Deploy a registered model with zero custom code.

**Step 1:** Start the serving endpoint.

```bash
mlflow models serve \
  --model-uri "models:/wine-quality-classifier/Production" \
  --host 0.0.0.0 \
  --port 5001 \
  --no-conda
```

**Step 2:** Test with curl.

```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                   "total_phenols", "flavanoids", "nonflavanoid_phenols",
                   "proanthocyanins", "color_intensity", "hue",
                   "od280/od315_of_diluted_wines", "proline"],
      "data": [[13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]]
    }
  }'
```

**Step 3:** Test with Python.

```python
# test_builtin_serve.py
import requests

url = "http://localhost:5001/invocations"

payload = {
    "dataframe_split": {
        "columns": [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols",
            "proanthocyanins", "color_intensity", "hue",
            "od280/od315_of_diluted_wines", "proline",
        ],
        "data": [
            [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050],
            [12.4, 2.31, 2.16, 18.0, 95, 2.54, 2.18, 0.34, 1.44, 3.6, 1.04, 2.92, 735],
        ],
    }
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
print(f"Predictions: {response.json()}")
```

### Exercise 2: Custom FastAPI Serving Wrapper

**Goal:** Use the provided `src/serving/serve_model.py` for a full-featured serving API.

**Step 1:** Start the FastAPI server.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_NAME=wine-quality-classifier
export MODEL_STAGE=Production

uvicorn src.serving.serve_model:app --host 0.0.0.0 --port 8000 --reload
```

**Step 2:** Explore the auto-generated API docs at `http://localhost:8000/docs`.

**Step 3:** Test all endpoints.

```python
# test_custom_serve.py
import requests

BASE_URL = "http://localhost:8000"

# Health Check
resp = requests.get(f"{BASE_URL}/health")
print(f"Health: {resp.json()}")

# Model Info
resp = requests.get(f"{BASE_URL}/model/info")
print(f"Model Info: {resp.json()}")

# Single Prediction
payload = {
    "features": [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
}
resp = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"Prediction: {resp.json()}")

# Batch Prediction
batch_payload = {
    "instances": [
        [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050],
        [12.4, 2.31, 2.16, 18.0, 95, 2.54, 2.18, 0.34, 1.44, 3.6, 1.04, 2.92, 735],
        [12.0, 1.67, 2.24, 15.6, 98, 2.45, 2.06, 0.30, 1.38, 3.5, 1.01, 2.85, 700],
    ]
}
resp = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload)
print(f"Batch Predictions: {resp.json()}")

# Reload Model (after promoting a new version)
resp = requests.post(f"{BASE_URL}/model/reload")
print(f"Reload: {resp.json()}")
```

### Exercise 3: Dockerize the Serving Endpoint

**Goal:** Package the FastAPI server in a Docker container.

```dockerfile
# Dockerfile.serve
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow fastapi uvicorn scikit-learn pandas numpy boto3
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.serving.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -f Dockerfile.serve -t mlflow-serve:latest .

docker run -d --name model-server \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_NAME=wine-quality-classifier \
  -e MODEL_STAGE=Production \
  mlflow-serve:latest

curl http://localhost:8000/health
```

### Exercise 4: Load Testing

**Goal:** Verify the serving endpoint handles concurrent requests.

```python
# exercise4_load_test.py
import concurrent.futures
import time

import requests

BASE_URL = "http://localhost:8000"
PAYLOAD = {
    "features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]
}


def make_request(_):
    start = time.time()
    resp = requests.post(f"{BASE_URL}/predict", json=PAYLOAD)
    elapsed = time.time() - start
    return resp.status_code, elapsed


n_requests = 100
print(f"Sending {n_requests} concurrent requests...")

start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(make_request, range(n_requests)))
total_time = time.time() - start_time

latencies = [r[1] for r in results]
success = sum(1 for r in results if r[0] == 200)

print(f"Total time:   {total_time:.2f}s")
print(f"Throughput:   {n_requests / total_time:.1f} req/s")
print(f"Success rate: {success}/{n_requests}")
print(f"Avg latency:  {sum(latencies)/len(latencies)*1000:.1f}ms")
print(f"P99 latency:  {sorted(latencies)[int(len(latencies)*0.99)]*1000:.1f}ms")
```

---

## Comparing Serving Approaches

| Feature | `mlflow models serve` | Custom FastAPI |
|---|---|---|
| Setup time | Seconds | Hours |
| Health checks | Basic | Full customization |
| Batch prediction | Not built-in | Custom endpoint |
| Multiple models | No | Yes |
| A/B testing | No | Yes (Module 07) |
| Model reload | No | `/model/reload` endpoint |
| Production-ready | Demo/dev | Production |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Model not in Production stage | 503 on startup | Promote model first (Module 05) |
| Wrong feature count | 400 Bad Request | Check model signature |
| Missing deps with `--no-conda` | ImportError | Install deps in serving environment |
| Reload without blue-green | Brief downtime | Use blue-green deployment for zero-downtime |

---

## Self-Check Questions

1. What is the difference between `mlflow models serve` and a custom FastAPI wrapper?
2. What format does `mlflow models serve` expect for input data?
3. Why is a health check endpoint important for production?
4. How would you handle model reloading without downtime?
5. What metrics should you track for a serving endpoint?

---

## You Know You Have Completed This Module When...

- [ ] You served a model using `mlflow models serve` and tested it with curl
- [ ] You started the custom FastAPI server and tested all endpoints
- [ ] You dockerized the serving endpoint
- [ ] You ran a basic load test
- [ ] You can explain when to use built-in vs custom serving
- [ ] Validation script passes: `bash modules/06-model-serving/validation/validate.sh`

---

**Next: [Module 07 - Pipeline Automation -->](../07-pipeline-automation/)**
