# Module 10: MLflow at Scale - Remote Tracking Server, Artifact Stores, and Team Workflows

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Modules 01-09 completed, Docker and Docker Compose installed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Deploy a production-grade MLflow tracking server with PostgreSQL and S3/MinIO
- Configure remote artifact stores (S3, MinIO, GCS, Azure Blob)
- Set up multi-user access with experiment permissions
- Implement team workflows with shared experiments and model registry
- Optimize MLflow for high-throughput training pipelines
- Plan backup, disaster recovery, and scaling strategies

---

## Concepts

### Production Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              Load Balancer (nginx)           │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────v──────────────────────────┐
                 │         MLflow Tracking Server               │
                 │    (stateless, horizontally scalable)        │
                 └──────┬────────────────────────┬─────────────┘
                        │                        │
              ┌─────────v─────────┐    ┌─────────v─────────┐
              │  PostgreSQL       │    │  S3 / MinIO        │
              │  (metadata)       │    │  (artifacts)       │
              │  - run params     │    │  - models          │
              │  - metrics        │    │  - plots           │
              │  - tags           │    │  - data samples    │
              │  - registry       │    │                    │
              └───────────────────┘    └────────────────────┘
```

### Why Not SQLite in Production?

| Aspect | SQLite | PostgreSQL |
|---|---|---|
| Concurrent writes | Single writer | Many concurrent writers |
| Scalability | Single machine | Can replicate and scale |
| Backup | File copy (risky under load) | pg_dump, streaming replication |
| Crash recovery | Limited WAL | Full MVCC, WAL, point-in-time recovery |
| Team use | One person at a time | Entire team simultaneously |

### Artifact Store Options

| Store | URI Format | Best For |
|---|---|---|
| Local filesystem | `file:///path/to/artifacts` | Development |
| AWS S3 | `s3://bucket-name/path` | AWS production |
| MinIO | `s3://bucket-name/path` (with endpoint override) | Self-hosted S3-compatible |
| Google Cloud Storage | `gs://bucket-name/path` | GCP production |
| Azure Blob Storage | `wasbs://container@account.blob.core.windows.net/path` | Azure production |

---

## Hands-On Lab

### Exercise 1: Deploy Production MLflow with Docker Compose

**Goal:** Use the provided `docker-compose.yml` to spin up a full production stack.

**Step 1:** Configure environment variables.

```bash
# From the repo root
cp .env.example .env

# Edit .env to customize credentials
# POSTGRES_PASSWORD=your-strong-password
# MINIO_ROOT_PASSWORD=your-strong-password
```

**Step 2:** Start the stack.

```bash
docker compose up -d

# Verify all services are running
docker compose ps

# Check logs
docker compose logs mlflow
docker compose logs postgres
docker compose logs minio
```

**Step 3:** Verify connectivity.

```python
# verify_production_setup.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Test experiment creation
experiment_id = mlflow.set_experiment("10-production-test")
print(f"Created experiment: {experiment_id}")

# Test run logging
with mlflow.start_run(run_name="production-connectivity-test"):
    mlflow.log_param("environment", "production-docker")
    mlflow.log_metric("test_value", 1.0)

    # Test artifact logging (this goes to MinIO)
    with open("test_artifact.txt", "w") as f:
        f.write("This artifact is stored in MinIO (S3-compatible).\n")
    mlflow.log_artifact("test_artifact.txt")

    artifact_uri = mlflow.get_artifact_uri()
    print(f"Artifact URI: {artifact_uri}")
    assert artifact_uri.startswith("s3://"), f"Expected S3 URI, got: {artifact_uri}"

print("Production setup verified successfully!")
```

**Step 4:** Access the services.

- MLflow UI: `http://localhost:5000`
- MinIO Console: `http://localhost:9001` (login with MINIO_ROOT_USER/PASSWORD)
- PostgreSQL: `localhost:5432`

### Exercise 2: Configure S3 Artifact Access from Client

**Goal:** Set up your local environment to read/write artifacts to the remote store.

```python
# exercise2_remote_artifacts.py
import os

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configure S3/MinIO access from the client
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mlflow123"

mlflow.set_experiment("10-remote-artifacts")

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="model-to-minio"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = float(np.mean(y_test == model.predict(X_test)))

    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_test, model.predict(X_test))

    # This model is stored in MinIO, not local filesystem
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name="production-wine-classifier",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Artifact URI: {mlflow.get_artifact_uri()}")

# Load the model back -- it fetches from MinIO automatically
run_id = mlflow.last_active_run().info.run_id
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print(f"Model loaded from MinIO: {type(loaded_model)}")
```

### Exercise 3: Team Workflow Patterns

**Goal:** Implement experiment naming conventions and access patterns for teams.

```python
# exercise3_team_workflows.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()


def setup_team_experiments():
    """Create a structured experiment hierarchy for a team."""
    # Convention: /<team>/<project>/<phase>
    experiments = [
        "team-ml/fraud-detection/exploration",
        "team-ml/fraud-detection/tuning",
        "team-ml/fraud-detection/production",
        "team-ml/recommendation/exploration",
        "team-ml/recommendation/tuning",
        "team-ml/recommendation/production",
    ]

    for exp_name in experiments:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            client.create_experiment(exp_name)
            print(f"Created: {exp_name}")
        else:
            print(f"Exists:  {exp_name}")


def log_team_run(experiment_name: str, developer: str, run_params: dict):
    """Standard team run with consistent tagging."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Standard team tags
        mlflow.set_tags({
            "developer": developer,
            "team": experiment_name.split("/")[0],
            "project": experiment_name.split("/")[1],
            "phase": experiment_name.split("/")[2],
        })

        mlflow.log_params(run_params)
        mlflow.log_metric("placeholder_metric", 0.95)

        print(f"Run logged to {experiment_name} by {developer}")


def search_team_runs(team: str, project: str, metric_name: str = "placeholder_metric"):
    """Find the best runs for a given project across all developers."""
    experiments = client.search_experiments(
        filter_string=f"name LIKE '{team}/{project}/%'"
    )

    exp_ids = [e.experiment_id for e in experiments]
    if not exp_ids:
        print(f"No experiments found for {team}/{project}")
        return

    runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=10,
    )

    print(f"\nTop runs for {team}/{project}:")
    for run in runs:
        developer = run.data.tags.get("developer", "unknown")
        phase = run.data.tags.get("phase", "unknown")
        metric = run.data.metrics.get(metric_name, 0)
        print(f"  {run.info.run_id[:8]} | {developer} | {phase} | {metric_name}={metric:.4f}")


# Demo the workflow
setup_team_experiments()

log_team_run("team-ml/fraud-detection/exploration", "alice", {"model": "xgboost", "lr": 0.01})
log_team_run("team-ml/fraud-detection/exploration", "bob", {"model": "lightgbm", "lr": 0.05})
log_team_run("team-ml/fraud-detection/tuning", "alice", {"model": "xgboost", "lr": 0.001})

search_team_runs("team-ml", "fraud-detection")
```

### Exercise 4: Performance Optimization for High-Throughput Training

**Goal:** Optimize MLflow logging for pipelines that generate thousands of runs.

```python
# exercise4_high_throughput.py
import time

import mlflow
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("10-high-throughput")


def benchmark_logging_strategies():
    """Compare different MLflow logging strategies for performance."""
    n_params = 20
    n_metrics = 50
    n_steps = 100

    # Strategy 1: Individual log calls
    start = time.time()
    with mlflow.start_run(run_name="strategy-individual"):
        for i in range(n_params):
            mlflow.log_param(f"param_{i}", f"value_{i}")
        for step in range(n_steps):
            for i in range(n_metrics):
                mlflow.log_metric(f"metric_{i}", np.random.random(), step=step)
    individual_time = time.time() - start

    # Strategy 2: Batch log calls
    start = time.time()
    with mlflow.start_run(run_name="strategy-batch"):
        mlflow.log_params({f"param_{i}": f"value_{i}" for i in range(n_params)})
        for step in range(n_steps):
            mlflow.log_metrics(
                {f"metric_{i}": np.random.random() for i in range(n_metrics)},
                step=step,
            )
    batch_time = time.time() - start

    print(f"Individual logging: {individual_time:.2f}s")
    print(f"Batch logging:      {batch_time:.2f}s")
    print(f"Speedup:            {individual_time / batch_time:.1f}x")


benchmark_logging_strategies()
```

### Exercise 5: Backup and Disaster Recovery

**Goal:** Set up backup procedures for the MLflow stack.

```bash
#!/bin/bash
# backup_mlflow.sh - Backup MLflow metadata and artifacts

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Backing up MLflow..."

# 1. Backup PostgreSQL metadata
echo "  Dumping PostgreSQL..."
docker compose exec -T postgres pg_dump -U mlflow mlflow > "$BACKUP_DIR/mlflow_metadata.sql"

# 2. Backup MinIO artifacts (using mc client)
echo "  Syncing MinIO artifacts..."
docker compose exec -T minio mc mirror /data "$BACKUP_DIR/artifacts" 2>/dev/null || \
  echo "  (Skipping MinIO backup -- use mc alias for full backup)"

# 3. Backup docker-compose and .env
echo "  Copying configuration..."
cp docker-compose.yml "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/.env.backup"

echo "Backup complete: $BACKUP_DIR"
ls -la "$BACKUP_DIR"
```

```bash
# Restore from backup
docker compose down -v  # WARNING: destroys current data

docker compose up -d postgres
sleep 5  # Wait for PostgreSQL to start

# Restore metadata
cat backups/YYYYMMDD_HHMMSS/mlflow_metadata.sql | \
  docker compose exec -T postgres psql -U mlflow mlflow

# Start remaining services
docker compose up -d
```

---

## Production Checklist

| Category | Item | Status |
|---|---|---|
| **Infrastructure** | PostgreSQL with automated backups | |
| **Infrastructure** | S3/MinIO with versioning enabled | |
| **Infrastructure** | MLflow server behind load balancer | |
| **Security** | Database credentials in secrets manager | |
| **Security** | S3 bucket policies configured | |
| **Security** | MLflow server behind auth proxy | |
| **Reliability** | Health check endpoints monitored | |
| **Reliability** | Database connection pooling configured | |
| **Reliability** | Artifact store redundancy | |
| **Operational** | Experiment naming convention documented | |
| **Operational** | Model registry workflow documented | |
| **Operational** | Backup and restore procedure tested | |

---

## Scaling Strategies

### Horizontal Scaling

```
                    Load Balancer
                    /     |     \
                   /      |      \
            MLflow-1  MLflow-2  MLflow-3
                  \       |       /
                   \      |      /
                  PostgreSQL (shared)
                        +
                  S3 (shared artifact store)
```

MLflow tracking servers are **stateless** -- you can run multiple instances behind a load balancer. They all share the same PostgreSQL and S3 backend.

### Database Optimization

```sql
-- Useful indexes for large MLflow deployments
CREATE INDEX idx_runs_experiment_id ON runs(experiment_id);
CREATE INDEX idx_metrics_run_uuid ON metrics(run_uuid);
CREATE INDEX idx_params_run_uuid ON params(run_uuid);
CREATE INDEX idx_tags_run_uuid ON tags(run_uuid);
```

### Artifact Store Best Practices

1. **Enable versioning** on your S3 bucket for accidental deletion protection
2. **Set lifecycle policies** to move old artifacts to cheaper storage tiers
3. **Use presigned URLs** for large artifact downloads
4. **Enable server-side encryption** for compliance

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Single MLflow server, no backup | Complete data loss if server dies | Deploy with PostgreSQL + S3, automated backups |
| No experiment naming convention | Chaotic, unsearchable experiment list | Enforce `team/project/phase` convention |
| Logging from Docker without S3 env vars | Artifacts stored in container filesystem | Set `AWS_ACCESS_KEY_ID` and endpoint URL |
| Not setting connection pool limits | Database connection exhaustion | Configure `--workers` and connection pooling |

---

## Self-Check Questions

1. Why should you use PostgreSQL instead of SQLite for team deployments?
2. How do you configure a Python client to write artifacts to MinIO?
3. What makes MLflow tracking servers horizontally scalable?
4. How would you implement access control for different teams?
5. What is your backup and restore strategy for MLflow metadata and artifacts?

---

## You Know You Have Completed This Module When...

- [ ] You deployed the full Docker Compose stack (PostgreSQL + MinIO + MLflow)
- [ ] You logged runs and artifacts to the remote store from a Python client
- [ ] You implemented team experiment naming conventions
- [ ] You benchmarked batch vs individual logging performance
- [ ] You have a working backup and restore procedure
- [ ] You can explain the production architecture
- [ ] Validation script passes: `bash modules/10-production-mlflow/validation/validate.sh`

---

**Congratulations! You have completed all 10 modules. Proceed to the [Capstone Project -->](../../capstone/)**
