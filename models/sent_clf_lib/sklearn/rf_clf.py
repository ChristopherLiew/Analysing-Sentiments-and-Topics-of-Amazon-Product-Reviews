"""
Random forest sentiment classifier using processed amazon product review data.
"""
# 1) Add in inference code with W&B
# 2) Run a full test

import typer
import wandb
import click_spinner as cs
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from ast import literal_eval
from utils import tune_model
from utils.hf_clf import get_data_files


# Instatiate Typer App
app = typer.Typer()


WANDB_RUN_NAME = "amz-rf-sent-clf_" + str(datetime.now())
WANDB_HF_PROJ_TAGS = ["Test-Run", "Random Forest"]
DEFAULT_RF_PARAMS_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "class_weight": ["balanced"],
}


@app.command()
def train(
    data_dir: str,
    embeds_col: str = "embeds",
    hyperparam_grid = DEFAULT_RF_PARAMS_GRID,
    wandb_entity: str = "chrisliew",
    wandb_proj_name: str = "amz-sent-analysis-classical-ml",
    wandb_run_name: str = WANDB_RUN_NAME,
    wandb_proj_tags: List[str] = WANDB_HF_PROJ_TAGS,
) -> None:
    """
    Train a Random Forest Classifer.

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
    typer.echo(f"Training a Random Forest Classifier on data from {data_dir}")

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
    MODEL_SAVE_DIR = Path("logs/rf_clf")
    data_files = get_data_files(ROOT, format='json')

    # Get data
    datasets = dict()
    datasets["train"] = pd.read_json(ROOT / "train.json")
    datasets["validation"] = pd.read_json(ROOT / "validation.json")
    datasets["test"] = pd.read_json(ROOT / "test.json")

    # Log data to W&B
    typer.secho("Logging train, val and test datasets to W and B:",
                fg=typer.colors.YELLOW)

    ds_artifact = wandb.Artifact(
        name=wandb_proj_name + "_datasets",
        type="datasets",
        description="""Processed train, val and test data
            for random forest sequence clf models.""",
        metadata={"sizes": [len(v) for k, v in datasets.items()]},
    )

    for name, fp in data_files.items():
        ds_artifact.add_file(fp, name=name + ".json")
    run.log_artifact(ds_artifact)

    # Get embeddings and labels
    X_train, y_train = (
        [embeddings
         for _, embeddings in datasets["train"][embeds_col].iteritems()],
        datasets["train"]["labels"],
    )

    X_val, y_val = (
        [embeddings
         for _, embeddings in datasets["validation"][embeds_col].iteritems()],
        datasets["validation"]["labels"],
    )

    # Run grid search

    typer.echo('Performing hyperparam tuning with Randomized Search CV')

    with cs.spinner():
        rf_clf_optimal, rf_clf_score, rf_clf_params = tune_model(
            RandomForestClassifier(),
            (np.array(X_train), y_train),
            search_params=hyperparam_grid,
        )

    typer.secho(f"""Best RF model has the optimal hyparams of: {rf_clf_params} and a score of {rf_clf_score}""",
                fg=typer.colors.GREEN)

    y_probas = rf_clf_optimal.predict_proba(np.array(X_val))
    y_pred = rf_clf_optimal.predict(np.array(X_val))

    # Run validation
    typer.echo('Logging classification charts to W&B')

    wandb.sklearn.plot_classifier(
        rf_clf_optimal,
        X_train,
        X_val,
        y_train,
        y_val,
        y_pred,
        y_probas,
        labels=["negative", "neutral", "positive"],
        model_name='RANDOM FOREST SEQUENCE CLASSIFIER',
        feature_names=None
    )

    # Save and log model to W&B

    typer.secho('Saving model locally and pushing model artifact to W&B',
                fg=typer.colors.BRIGHT_YELLOW)

    rf_model_save_path = MODEL_SAVE_DIR / (f"rf_clf_{datetime.now()}.joblib")
    joblib.dump(rf_clf_optimal, rf_model_save_path)

    trained_model_artifact = wandb.Artifact(
        wandb_proj_name + "_rf_model",
        type="model",
        description="Trained random forest classifier for sentiment analysis",
    )

    trained_model_artifact.add_file(str(rf_model_save_path))
    run.log(trained_model_artifact)

    wandb.finish()

    typer.secho('Training complete!', fg=typer.colors.GREEN)


@app.command()
def predict(
    wandb_entity: Optional[str] = None,
    wandb_proj_name: str = 'amz-sent-analysis-classical-ml',
    inf_data = None,
    embeds_col: str = "embeds"
):
    with wandb.init(entity=wandb_entity, project=wandb_proj_name, job_type="inference") as run:
        my_model_name = f"{wandb_proj_name}_rf_model:latest"
        my_model_artifact = run.use_artifact(my_model_name)
        model_dir = my_model_artifact.download()
        model = joblib.load(model_dir)

        y_pred = model.predict(inf_data[embeds_col])

        return y_pred