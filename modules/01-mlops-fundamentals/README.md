# Module 01: MLOps Fundamentals - MLflow Installation and Tracking Server Setup

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Python 3.9+, pip, Docker installed, basic terminal knowledge |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain what MLOps is and why it matters for production ML systems
- Install MLflow and verify all components work correctly
- Launch a local MLflow tracking server and navigate the UI
- Understand the difference between local file tracking and a tracking server
- Configure environment variables for MLflow connectivity

---

## Concepts

### What is MLOps?

MLOps (Machine Learning Operations) is the set of practices that combines ML development with operations to reliably deploy and maintain ML systems in production. Just as DevOps transformed software delivery, MLOps transforms how organizations build, deploy, and monitor ML models.

**The ML lifecycle without MLOps:**
1. Data scientist trains model in a notebook
2. "Best" model parameters live in someone's memory or a sticky note
3. Deployment is a manual, error-prone copy-paste to production
4. No one knows which model is in production or how it performs

**The ML lifecycle with MLOps:**
1. Every training run is tracked with parameters, metrics, and artifacts
2. Models are versioned in a registry with clear lineage
3. Deployment is automated with rollback capability
4. Performance is continuously monitored with drift detection

### Where MLflow Fits

MLflow is an open-source platform that addresses four key pillars of the ML lifecycle:

| Component | Purpose | Module Coverage |
|---|---|---|
| **MLflow Tracking** | Log parameters, metrics, artifacts | Modules 02-03 |
| **MLflow Projects** | Package ML code for reproducibility | Module 05 |
| **MLflow Models** | Standard format for model packaging | Modules 04-06 |
| **MLflow Registry** | Centralized model store with versioning | Module 04 |

### Key Terminology

| Term | Definition |
|---|---|
| **Experiment** | A named group of related runs (e.g., "fraud-detection-v2") |
| **Run** | A single execution of ML code that logs params, metrics, artifacts |
| **Artifact** | Any file output from a run (model files, plots, data samples) |
| **Tracking URI** | The address of the MLflow tracking server |
| **Backend Store** | Database storing run metadata (SQLite, PostgreSQL, MySQL) |
| **Artifact Store** | File system or object store for artifacts (local, S3, GCS, Azure Blob) |

---

## Hands-On Lab

### Exercise 1: Install MLflow and Verify

**Goal:** Get MLflow installed and confirm all components work.

**Step 1:** Create a virtual environment and install dependencies.

```bash
python -m venv mlflow-env
source mlflow-env/bin/activate  # Linux/Mac
# mlflow-env\Scripts\activate   # Windows

pip install mlflow scikit-learn pandas numpy matplotlib
```

**Step 2:** Verify the installation.

```bash
mlflow --version
# Expected: mlflow, version 2.x.x

python -c "import mlflow; print(mlflow.__version__)"
```

**Step 3:** Run your first tracking test.

```python
# test_mlflow_install.py
import mlflow

# This uses a local ./mlruns directory by default
mlflow.set_experiment("installation-test")

with mlflow.start_run(run_name="hello-mlflow"):
    mlflow.log_param("framework", "test")
    mlflow.log_metric("score", 0.95)
    mlflow.log_metric("loss", 0.05)
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Artifact URI: {mlflow.active_run().info.artifact_uri}")

print("MLflow installation verified successfully!")
```

```bash
python test_mlflow_install.py
```

**What you should see:** A run ID printed, and a new `mlruns/` directory created.

### Exercise 2: Launch the MLflow Tracking Server

**Goal:** Start a local tracking server and explore the UI.

**Option A: Simple local server (file-based)**

```bash
mlflow server --host 0.0.0.0 --port 5000
```

This starts a server backed by the local `./mlruns` directory. Open `http://localhost:5000` in your browser.

**Option B: Server with SQLite backend (recommended for learning)**

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

This uses SQLite for metadata, which enables the Model Registry features.

**Option C: Full production stack with Docker Compose**

```bash
# From the repo root
cp .env.example .env
docker compose up -d

# Verify all services are running
docker compose ps
```

This starts PostgreSQL (metadata), MinIO (artifacts), and the MLflow server.

**Step 2:** Explore the UI at `http://localhost:5000`:
- Click on the "installation-test" experiment
- View the run you created in Exercise 1
- Examine the parameters and metrics tabs
- Note the artifact storage path

### Exercise 3: Connect a Script to the Tracking Server

**Goal:** Configure a Python script to log to the tracking server instead of local files.

```python
# connect_to_server.py
import mlflow
import numpy as np

# Point to the tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Verify connection
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Create an experiment
mlflow.set_experiment("server-connection-test")

with mlflow.start_run(run_name="remote-logging-test"):
    # Log various parameter types
    mlflow.log_param("algorithm", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Log metrics over multiple steps (simulating training epochs)
    for epoch in range(10):
        loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
        accuracy = 1.0 - loss + np.random.normal(0, 0.005)
        mlflow.log_metrics(
            {"training_loss": loss, "training_accuracy": accuracy},
            step=epoch,
        )

    # Log a text artifact
    with open("notes.txt", "w") as f:
        f.write("This run tests server connectivity.\n")
    mlflow.log_artifact("notes.txt")

    print("Run logged to tracking server successfully!")
```

```bash
python connect_to_server.py
```

Now refresh the MLflow UI and find the new experiment and run.

### Exercise 4: Environment Variable Configuration

**Goal:** Use environment variables so you never hardcode the tracking URI.

```bash
# Set the environment variable
export MLFLOW_TRACKING_URI=http://localhost:5000

# Now any MLflow call will use this URI automatically
python -c "
import mlflow
print(f'Tracking URI: {mlflow.get_tracking_uri()}')
mlflow.set_experiment('env-var-test')
with mlflow.start_run():
    mlflow.log_param('configured_via', 'environment_variable')
    print('Logged via env var configuration!')
"
```

Create a `.env` file for your projects:

```bash
# .env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=my-project
```

Load it in Python:

```python
from dotenv import load_dotenv
load_dotenv()

import mlflow
# mlflow.get_tracking_uri() will now return http://localhost:5000
```

---

## Architecture Overview

```
+------------------+     +-------------------+     +------------------+
|  Python Script   | --> | MLflow Tracking   | --> | Backend Store    |
|  (your ML code)  |     | Server (port 5000)|     | (SQLite/Postgres)|
+------------------+     +-------------------+     +------------------+
                                |
                                v
                         +------------------+
                         | Artifact Store   |
                         | (local/S3/MinIO) |
                         +------------------+
```

- **Backend Store** holds run metadata (parameters, metrics, tags, run status)
- **Artifact Store** holds large files (models, plots, datasets)
- The Tracking Server provides a REST API and a web UI

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not setting tracking URI | Runs saved to local `./mlruns` instead of server | Set `MLFLOW_TRACKING_URI` env var or call `mlflow.set_tracking_uri()` |
| Port conflict on 5000 | "Address already in use" error | Kill existing process or use `--port 5001` |
| SQLite backend without `sqlite:///` prefix | Backend store error on startup | Use three slashes: `sqlite:///mlflow.db` |
| Forgetting to activate venv | `mlflow: command not found` | Run `source mlflow-env/bin/activate` |

---

## Self-Check Questions

1. What are the four components of MLflow and what does each do?
2. What is the difference between the backend store and the artifact store?
3. Why would you use PostgreSQL instead of SQLite as a backend store?
4. How does setting `MLFLOW_TRACKING_URI` change where runs are logged?
5. What happens if the tracking server goes down while a training script is running?

---

## You Know You Have Completed This Module When...

- [ ] MLflow is installed and `mlflow --version` works
- [ ] You can start a tracking server (any of the three options)
- [ ] You logged a run via a Python script to the tracking server
- [ ] You can view runs, parameters, and metrics in the MLflow UI
- [ ] You configured `MLFLOW_TRACKING_URI` as an environment variable
- [ ] You can explain the MLflow architecture to someone else

---

## Troubleshooting

### Issue: `mlflow server` fails with database error
```bash
# Reset the SQLite database
rm mlflow.db
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Issue: Docker Compose services not starting
```bash
docker compose logs mlflow    # Check MLflow server logs
docker compose logs postgres  # Check database logs
docker compose down -v && docker compose up -d  # Full reset
```

### Issue: Cannot connect from Python script
```python
import requests
# Test connectivity manually
resp = requests.get("http://localhost:5000/api/2.0/mlflow/experiments/search")
print(resp.status_code, resp.json())
```

---

**Next: [Module 02 - MLflow Setup and Configuration -->](../02-mlflow-setup/)**
