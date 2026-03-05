# Capstone Project: End-to-End ML Lifecycle with MLflow

## Overview

This capstone project combines everything you learned across all 10 modules into a single, production-grade ML system. You will build a complete pipeline that trains, registers, serves, A/B tests, and monitors a machine learning model -- all tracked and managed through MLflow.

This is the project you will showcase to hiring managers and discuss in technical interviews.

---

## The Challenge

Build a **production ML system** for wine quality classification that demonstrates mastery of the complete MLflow ecosystem:

### Phase 1: Training Pipeline (Modules 01-03)

- [ ] Set up MLflow tracking server with PostgreSQL backend and MinIO artifact store
- [ ] Train at least 3 different model types (e.g., RandomForest, XGBoost, GradientBoosting)
- [ ] Log all parameters, metrics, artifacts (confusion matrices, feature importance plots)
- [ ] Log models with proper signatures and input examples
- [ ] Compare models in the MLflow UI

### Phase 2: Hyperparameter Optimization (Module 08)

- [ ] Run Optuna hyperparameter search for the best model type from Phase 1
- [ ] Log each trial as a nested MLflow run
- [ ] Use pruning to stop unpromising trials early
- [ ] Identify the best hyperparameter configuration

### Phase 3: Model Registry (Modules 04-05)

- [ ] Register the top 3 models in the MLflow Model Registry
- [ ] Implement an automated validation pipeline that tests each version
- [ ] Promote the best validated model through Staging to Production
- [ ] Add descriptions and tags to all registered versions

### Phase 4: Model Serving (Module 06)

- [ ] Build a FastAPI serving endpoint that loads the Production model
- [ ] Implement health check, single prediction, and batch prediction endpoints
- [ ] Dockerize the serving endpoint
- [ ] Write a test script that verifies all endpoints

### Phase 5: A/B Testing (Module 07)

- [ ] Deploy an A/B testing router with the Production model as control
- [ ] Run a challenger model (from Phase 3) as the treatment
- [ ] Simulate 1000+ requests and collect results
- [ ] Run statistical significance testing on the results
- [ ] Log the A/B test results to MLflow

### Phase 6: Monitoring (Module 09)

- [ ] Build a monitoring pipeline that checks model performance on new batches
- [ ] Implement data drift detection using the Kolmogorov-Smirnov test
- [ ] Set up alerts for accuracy degradation and feature drift
- [ ] Simulate gradual data drift and show the monitoring system detecting it
- [ ] Log all monitoring results to MLflow

### Phase 7: Production Deployment (Module 10)

- [ ] Deploy the full stack using Docker Compose (PostgreSQL + MinIO + MLflow + serving)
- [ ] Document the complete architecture with a diagram
- [ ] Write a backup and restore procedure
- [ ] Create a runbook for common operational tasks

---

## Architecture

Your completed system should look like this:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                         │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ PostgreSQL   │  │ MinIO (S3)  │  │ MLflow Tracking Server  │  │
│  │ :5432        │  │ :9000/:9001 │  │ :5000                   │  │
│  └──────┬───────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         │                 │                       │               │
│         └─────────────────┴───────────────────────┘               │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │ Model Serving (FastAPI) │  │ A/B Testing Router          │   │
│  │ :8000                   │  │ :8001                       │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Monitoring Pipeline (scheduled)                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/techlearn-center/mlflow-experiment-tracking.git
cd mlflow-experiment-tracking

# 2. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Start the infrastructure
cp .env.example .env
docker compose up -d

# 4. Verify everything is running
docker compose ps
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print('Connected!')"

# 5. Start building your capstone solution in capstone/solution/
```

---

## Deliverables

| Deliverable | File/Location | Description |
|---|---|---|
| Training scripts | `capstone/solution/train.py` | Scripts for all model types |
| Tuning script | `capstone/solution/tune.py` | Optuna integration |
| Registry script | `capstone/solution/registry.py` | Model registration and promotion |
| Serving code | `capstone/solution/serve.py` | FastAPI serving endpoint |
| A/B testing | `capstone/solution/ab_test.py` | Router + simulation + analysis |
| Monitoring | `capstone/solution/monitor.py` | Performance + drift monitoring |
| Docker setup | `capstone/solution/docker-compose.yml` | Complete deployment stack |
| Architecture doc | `capstone/solution/ARCHITECTURE.md` | System design with diagrams |
| Runbook | `capstone/solution/RUNBOOK.md` | Operational procedures |

---

## Evaluation Criteria

| Criteria | Weight | What Evaluators Look For |
|---|---|---|
| **Functionality** | 25% | All phases work end-to-end; models train, register, serve, and are monitored |
| **MLflow Usage** | 20% | Proper use of tracking, registry, signatures, artifacts, and tags |
| **Architecture** | 15% | Clean separation of concerns; production-ready infrastructure |
| **Code Quality** | 15% | Well-structured, documented, and testable code |
| **Monitoring** | 10% | Drift detection works; alerts fire correctly |
| **Documentation** | 10% | Clear architecture doc and operational runbook |
| **Bonus** | 5% | CI/CD, advanced visualizations, custom pyfunc models |

---

## Interview Talking Points

When you present this capstone, be prepared to discuss:

1. **Architecture decisions** -- Why PostgreSQL over SQLite? Why MinIO over local filesystem?
2. **Model selection** -- How did you compare models? What metrics did you use?
3. **Registry workflow** -- How do models move from training to production?
4. **A/B testing** -- How do you determine statistical significance? What sample size do you need?
5. **Monitoring** -- How do you detect drift? What triggers retraining?
6. **Trade-offs** -- What would you change for 10x more traffic? For 100 models?
7. **Failure scenarios** -- What happens if the MLflow server goes down? If MinIO is unavailable?

---

## Solution

The `solution/` directory contains a reference implementation. Try to complete the capstone yourself first -- that is what builds real skills and interview confidence.

```bash
# If you get stuck, check the solution
ls capstone/solution/
```

---

## Showcasing to Hiring Managers

When you complete this capstone:

1. **Fork this repo** to your personal GitHub
2. **Add your solution** with detailed commit messages explaining your decisions
3. **Update this README** with screenshots of your MLflow UI showing experiments, model registry, and training curves
4. **Record a 5-minute demo video** walking through the full pipeline
5. **Reference it on your resume** as "Production ML Lifecycle Pipeline with MLflow"
6. **Be ready to demo it live** -- keep a Docker Compose setup ready to spin up

See [docs/portfolio-guide.md](../docs/portfolio-guide.md) for detailed guidance on presenting this project.
