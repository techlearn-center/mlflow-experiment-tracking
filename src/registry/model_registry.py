"""
MLflow Model Registry operations: register, promote, and load models.

This script demonstrates:
- Registering a model from an existing run
- Transitioning model versions through stages (Staging -> Production)
- Loading models by stage or version for inference
- Comparing model versions
- Adding descriptions and tags to registered models

Usage:
    python -m src.registry.model_registry register --run-id <RUN_ID> --model-name "wine-classifier"
    python -m src.registry.model_registry promote --model-name "wine-classifier" --version 1 --stage Production
    python -m src.registry.model_registry load --model-name "wine-classifier" --stage Production
    python -m src.registry.model_registry compare --model-name "wine-classifier" --versions 1,2,3
"""

import argparse
import os
import sys
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient


def get_client(tracking_uri: str = None) -> MlflowClient:
    """Create an MLflow client with the specified tracking URI."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
    description: str = None,
    tags: dict = None,
    tracking_uri: str = None,
):
    """
    Register a model from an existing MLflow run.

    Args:
        run_id: The MLflow run ID containing the logged model.
        model_name: Name for the registered model.
        artifact_path: Path within the run artifacts where the model is stored.
        description: Human-readable description of the model.
        tags: Dictionary of tags to attach to the model version.
        tracking_uri: MLflow tracking server URI.

    Returns:
        The ModelVersion object.
    """
    client = get_client(tracking_uri)
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Register the model (creates the registered model if it doesn't exist)
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(f"Registered model '{model_name}' version {model_version.version}")
    print(f"  Source run: {run_id}")
    print(f"  Model URI: {model_uri}")

    # Add description if provided
    if description:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description,
        )
        print(f"  Description: {description}")

    # Add tags if provided
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=key,
                value=str(value),
            )
        print(f"  Tags: {tags}")

    return model_version


def promote_model(
    model_name: str,
    version: int,
    stage: str = "Staging",
    archive_existing: bool = True,
    tracking_uri: str = None,
):
    """
    Transition a model version to a new stage.

    Stages: None -> Staging -> Production -> Archived

    Args:
        model_name: Name of the registered model.
        version: Version number to promote.
        stage: Target stage (Staging, Production, Archived).
        archive_existing: Whether to archive existing versions in the target stage.
        tracking_uri: MLflow tracking server URI.

    Returns:
        The updated ModelVersion object.
    """
    client = get_client(tracking_uri)

    valid_stages = ["None", "Staging", "Production", "Archived"]
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")

    # Get current stage for logging
    current_version = client.get_model_version(model_name, version)
    old_stage = current_version.current_stage

    # Transition the model
    updated_version = client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )

    print(f"Model '{model_name}' version {version}: {old_stage} -> {stage}")

    # Add a tag recording the transition
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key=f"promoted_to_{stage.lower()}",
        value=datetime.utcnow().isoformat(),
    )

    return updated_version


def load_model_by_stage(
    model_name: str,
    stage: str = "Production",
    tracking_uri: str = None,
):
    """
    Load a model from the registry by its stage.

    Args:
        model_name: Name of the registered model.
        stage: Stage to load from (Staging, Production).
        tracking_uri: MLflow tracking server URI.

    Returns:
        The loaded model object ready for inference.
    """
    get_client(tracking_uri)

    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully. Type: {type(model)}")

    return model


def load_model_by_version(
    model_name: str,
    version: int,
    tracking_uri: str = None,
):
    """
    Load a specific version of a registered model.

    Args:
        model_name: Name of the registered model.
        version: Version number to load.
        tracking_uri: MLflow tracking server URI.

    Returns:
        The loaded model object ready for inference.
    """
    get_client(tracking_uri)

    model_uri = f"models:/{model_name}/{version}"
    print(f"Loading model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully. Type: {type(model)}")

    return model


def compare_versions(
    model_name: str,
    versions: list,
    tracking_uri: str = None,
):
    """
    Compare multiple versions of a registered model by their metrics.

    Args:
        model_name: Name of the registered model.
        versions: List of version numbers to compare.
        tracking_uri: MLflow tracking server URI.
    """
    client = get_client(tracking_uri)

    print(f"\nComparing versions of '{model_name}':")
    print("-" * 80)
    print(f"{'Version':<10} {'Stage':<15} {'Status':<10} {'Run ID':<35} {'Created'}")
    print("-" * 80)

    for v in versions:
        try:
            mv = client.get_model_version(model_name, v)
            # Get run metrics
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics

            created = datetime.fromtimestamp(mv.creation_timestamp / 1000)
            print(
                f"{mv.version:<10} {mv.current_stage:<15} {mv.status:<10} "
                f"{mv.run_id:<35} {created.strftime('%Y-%m-%d %H:%M')}"
            )

            # Print key metrics
            for metric_name in sorted(metrics.keys()):
                if any(k in metric_name for k in ["accuracy", "f1", "rmse", "r2", "mae"]):
                    print(f"  {metric_name}: {metrics[metric_name]:.4f}")

        except Exception as e:
            print(f"{v:<10} ERROR: {str(e)}")

    print("-" * 80)


def list_models(tracking_uri: str = None):
    """List all registered models and their latest versions."""
    client = get_client(tracking_uri)

    models = client.search_registered_models()

    if not models:
        print("No registered models found.")
        return

    print(f"\nRegistered Models ({len(models)} total):")
    print("-" * 80)

    for rm in models:
        print(f"\n  Model: {rm.name}")
        if rm.description:
            print(f"  Description: {rm.description}")

        for mv in rm.latest_versions:
            print(
                f"    Version {mv.version} | Stage: {mv.current_stage} | "
                f"Status: {mv.status} | Run: {mv.run_id[:8]}..."
            )


def delete_model_version(
    model_name: str,
    version: int,
    tracking_uri: str = None,
):
    """Delete a specific model version (must be archived first in some setups)."""
    client = get_client(tracking_uri)

    # First archive the version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived",
    )

    # Then delete
    client.delete_model_version(name=model_name, version=version)
    print(f"Deleted model '{model_name}' version {version}")


def main():
    parser = argparse.ArgumentParser(description="MLflow Model Registry operations")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register
    reg_parser = subparsers.add_parser("register", help="Register a model from a run")
    reg_parser.add_argument("--run-id", required=True, help="MLflow run ID")
    reg_parser.add_argument("--model-name", required=True, help="Registered model name")
    reg_parser.add_argument("--artifact-path", default="model")
    reg_parser.add_argument("--description", default=None)
    reg_parser.add_argument("--tracking-uri", default=None)

    # Promote
    promo_parser = subparsers.add_parser("promote", help="Promote a model version")
    promo_parser.add_argument("--model-name", required=True)
    promo_parser.add_argument("--version", type=int, required=True)
    promo_parser.add_argument("--stage", default="Staging", choices=["Staging", "Production", "Archived"])
    promo_parser.add_argument("--tracking-uri", default=None)

    # Load
    load_parser = subparsers.add_parser("load", help="Load a model by stage")
    load_parser.add_argument("--model-name", required=True)
    load_parser.add_argument("--stage", default="Production")
    load_parser.add_argument("--version", type=int, default=None)
    load_parser.add_argument("--tracking-uri", default=None)

    # Compare
    cmp_parser = subparsers.add_parser("compare", help="Compare model versions")
    cmp_parser.add_argument("--model-name", required=True)
    cmp_parser.add_argument("--versions", required=True, help="Comma-separated version numbers")
    cmp_parser.add_argument("--tracking-uri", default=None)

    # List
    list_parser = subparsers.add_parser("list", help="List all registered models")
    list_parser.add_argument("--tracking-uri", default=None)

    args = parser.parse_args()

    if args.command == "register":
        register_model(
            run_id=args.run_id,
            model_name=args.model_name,
            artifact_path=args.artifact_path,
            description=args.description,
            tracking_uri=args.tracking_uri,
        )
    elif args.command == "promote":
        promote_model(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage,
            tracking_uri=args.tracking_uri,
        )
    elif args.command == "load":
        if args.version:
            load_model_by_version(
                model_name=args.model_name,
                version=args.version,
                tracking_uri=args.tracking_uri,
            )
        else:
            load_model_by_stage(
                model_name=args.model_name,
                stage=args.stage,
                tracking_uri=args.tracking_uri,
            )
    elif args.command == "compare":
        versions = [int(v.strip()) for v in args.versions.split(",")]
        compare_versions(
            model_name=args.model_name,
            versions=versions,
            tracking_uri=args.tracking_uri,
        )
    elif args.command == "list":
        list_models(tracking_uri=args.tracking_uri)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
