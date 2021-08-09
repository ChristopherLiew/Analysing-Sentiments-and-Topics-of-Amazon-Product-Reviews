"""
SVC sentiment classifier using processed amazon product review data.
"""

import typer
import wandb
import click_spinner as cs
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVC
from utils import tune_model
from utils.hf_clf import get_data_files


# Instatiate Typer App
app = typer.Typer()


WANDB_RUN_NAME = "amz-svc-sent-clf_" + str(datetime.now())
WANDB_SVC_PROJ_TAGS = ["Test-Run", "Support Vector Classifier"]
DEFAULT_SVC_PARAMS_GRID = {
    "kernel": ["linear"],
    "C": [0.5, 0.75, 1.0],
    "gamma": ["auto"],
    "class_weight": ["balanced"],
}


@app.command()
def train(
    data_dir: str,
    embeds_col: str = "embeds",
    hyperparam_grid=DEFAULT_SVC_PARAMS_GRID,
    wandb_entity: str = "chrisliew",
    wandb_proj_name: str = "amz-sent-analysis-classical-ml",
    wandb_run_name: str = WANDB_RUN_NAME,
    wandb_proj_tags: List[str] = WANDB_SVC_PROJ_TAGS,
) -> None:
    """
    Train a Support Vector Classifer.

    Args:\n
        data_dir (str): Path to a data directory containing train, val and test data in\n
        in the following structure:\n
        └─── data_dir\n
            ├─── train.json\n
            ├─── test.json\n
            └─── validation.json\n
        embeds_col (str): Name of column containing vectorised word embeddings for modelling.\n
        wandb_entity (str, optional): W and B username. Defaults to 'chrisliew'\n
        wandb_proj_name (str, optional): W and B project name. Defaults to 'amz-sent-analysis'.\n
        wandb_run_name (str, optional): W and B run name. Defaults to 'amz-hf-sent-clf'.
        wandb_proj_tags (List[str], optional): W and B project tags. Defaults to ['Test-Run', 'Albert-V2'].
    """
    typer.echo(f"Training a SVC on data from {data_dir}")

    # Log into W&B
    wandb.login()

    # Initialise W&B run
    run = wandb.init(
        project=wandb_proj_name,
        name=wandb_run_name,
        tags=wandb_proj_tags,
        entity=wandb_entity,
        job_type="training",
    )

    # Constants
    ROOT = Path(data_dir)
    MODEL_SAVE_DIR = Path("logs/svc_clf")
    data_files = get_data_files(ROOT, format="json")

    # Get data
    datasets = dict()
    datasets["train"] = pd.read_json(ROOT / "train.json")
    datasets["validation"] = pd.read_json(ROOT / "validation.json")
    datasets["test"] = pd.read_json(ROOT / "test.json")

    # Log data to W&B
    typer.secho(
        "Logging train, val and test datasets to W and B:", fg=typer.colors.YELLOW
    )

    ds_artifact = wandb.Artifact(
        name=wandb_proj_name + "_datasets",
        type="datasets",
        description="""Processed train, val and test data
            for SVC sequence clf models.""",
        metadata={"sizes": [len(v) for k, v in datasets.items()]},
    )

    for name, fp in data_files.items():
        ds_artifact.add_file(fp, name=name + ".json")
        typer.echo(f"Logged dataset from: {fp} as {name} to W&B")

    run.log_artifact(ds_artifact)

    # Get embeddings and labels
    X_train, y_train = (
        [embeddings for _, embeddings in datasets["train"][embeds_col].iteritems()],
        datasets["train"]["labels"],
    )

    X_val, y_val = (
        [
            embeddings
            for _, embeddings in datasets["validation"][embeds_col].iteritems()
        ],
        datasets["validation"]["labels"],
    )

    # Run grid search

    typer.echo("Performing hyperparam tuning with Randomized Search CV")

    with cs.spinner():
        svc_clf_optimal, svc_clf_score, svc_clf_params = tune_model(
            SVC(probability=True),
            (np.array(X_train), y_train),
            search_params=hyperparam_grid,
        )

    typer.secho(
        f"""Best SVC model has the optimal hyparams of: {svc_clf_params} and a score of {svc_clf_score}""",
        fg=typer.colors.GREEN,
    )

    y_probas = svc_clf_optimal.predict_proba(np.array(X_val))
    y_pred = svc_clf_optimal.predict(np.array(X_val))

    # Run validation
    typer.echo("Logging classification charts to W&B")

    # Super slow (Run only relevant charts)
    wandb.sklearn.plot_classifier(
        svc_clf_optimal,
        X_train,
        X_val,
        y_train,
        y_val,
        y_pred,
        y_probas,
        labels=["negative", "neutral", "positive"],
        model_name="SUPPORT VECTOR SEQUENCE CLASSIFIER",
        feature_names=None,
    )

    # Save and log model to W&B
    typer.secho(
        "Saving model locally and pushing model artifact to W&B",
        fg=typer.colors.BRIGHT_YELLOW,
    )

    svc_model_save_path = MODEL_SAVE_DIR / "svc_clf_model.joblib"
    joblib.dump(svc_clf_optimal, svc_model_save_path)

    trained_model_artifact = wandb.Artifact(
        wandb_proj_name + "_svc_model",
        type="model",
        description="Trained support vector classifier for sentiment analysis",
    )

    trained_model_artifact.add_file(svc_model_save_path, name="svc_model.joblib")
    run.log_artifact(trained_model_artifact)
    wandb.finish()
    typer.secho("Training complete!", fg=typer.colors.GREEN)


@app.command()
def predict(
    wandb_entity: Optional[str] = None,
    wandb_proj_name: str = "amz-sent-analysis-classical-ml",
    inf_model_path: Optional[str] = None,
    inf_data_path: Optional[str] = None,
    embeds_col: str = "embeds"
):

    inf_run_name = "svc_inference_" + str(datetime.now())

    with wandb.init(
        name=inf_run_name,
        entity=wandb_entity,
        project=wandb_proj_name,
        job_type="inference"
    ) as run:

        if inf_data_path is None:
            typer.secho("Pulling latest model from W&B", fg=typer.colors.YELLOW)
            with cs.spinner():
                my_model_name = f"{wandb_proj_name}_svc_model:latest"
                my_model_artifact = run.use_artifact(my_model_name)
                model_dir = my_model_artifact.download()
                model_path = Path(model_dir)
                model = joblib.load(model_path / "svc_model.joblib")
        else:
            model = joblib.load(inf_model_path)

        # Load test data
        if inf_data_path is None:
            typer.secho("Pulling latest test dataset from W&B", fg=typer.colors.YELLOW)
            with cs.spinner():
                my_ds_name = f"{wandb_proj_name}_datasets:latest"
                ds_artifact = run.use_artifact(my_ds_name)
                ds_dir = ds_artifact.download()
                inf_data_path = Path(ds_dir) / "test.json"

        test_data = pd.read_json(inf_data_path)

        # Make predictions
        typer.secho(
            f"Making predictions using test dataset from {inf_data_path}",
            fg=typer.colors.YELLOW,
        )

        X_test = [embeddings for _, embeddings in test_data[embeds_col].iteritems()]
        y_pred = model.predict(
            np.array(X_test)
        )

        typer.secho("Logging confusion matrix to W&B", fg=typer.colors.YELLOW)
        wandb.sklearn.plot_confusion_matrix(
            y_true=test_data["labels"],
            y_pred=y_pred,
            labels=["negative", "neutral", "positive"]
        )
        run.finish()

        return y_pred
