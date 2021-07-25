"""
Random forest sentiment classifier using processed amazon product review data.
"""
# 1) Create CLI for training and inference
# - Process Training Data
# - Train Model + Log results
# - Make inference + Log eval results
# 2) Move logging into a sepearate file


from utils.hf_clf import get_data_files
import typer
import wandb
from typing import (
    List,
    Dict,
    Any
)
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from ast import literal_eval
from utils import (
    get_clf_results,
    tune_model
)
from utils.hf_clf import get_data_files


# Instatiate Typer App
app = typer.Typer()


WANDB_RUN_NAME = "amz-rf-sent-clf_" + str(datetime.now())
WANDB_HF_PROJ_TAGS = ["Test-Run", 'Random Forest']
DEFAULT_RF_PARAMS_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "class_weight": ["balanced"],
}


@app.command
def train(
    data_dir: str,
    embeds_col: str = 'embeds',
    hyperparam_grid: Dict[str, List[Any]] = DEFAULT_RF_PARAMS_GRID,
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
            ├─── train.csv\n
            ├─── test.csv\n
            └─── validation.csv\n
        embeds_col (str): Name of column containing vectorised word embeddings for modelling.\n
        wandb_entity (str, optional): W and B username. Defaults to 'chrisliew'\n
        wandb_proj_name (str, optional): W and B project name. Defaults to 'amz-sent-analysis'.\n
        wandb_run_name (str, optional): W and B run name. Defaults to 'amz-hf-sent-clf'.
        wandb_proj_tags (List[str], optional): W and B project tags. Defaults to ['Test-Run', 'Albert-V2'].
    """
    typer.echo(
        f"Training a Random Forest Classifier on data from {data_dir}")

    # Log into W&B
    wandb.login()

    # Initialise W&B run
    run = wandb.init(
        project=wandb_proj_name,
        name=wandb_run_name,
        tags=wandb_proj_tags,
        entity=wandb_entity,
        job_type='training'
    )

    # Constants
    ROOT = Path(data_dir)
    MODEL_SAVE_DIR = Path("logs/rf_clf")
    data_files = get_data_files(ROOT)

    # Get data
    datasets = dict()
    datasets['train'] = pd.read_csv(ROOT / 'train.csv')
    datasets['validation'] = pd.read_csv(ROOT / 'validation.csv')
    datasets['test'] = pd.read_csv(ROOT / 'test.csv')

    # Log data to W&B
    typer.echo("logging train, val and test datasets to W and B:")

    ds_artifact = wandb.Artifact(
        name=wandb_proj_name + "_datasets",
        type="datasets",
        description="""Processed train, val and test data
            for randomforest sequence clf models.""",
        metadata={"sizes": [v.num_rows for k, v in datasets.items()]}
    )

    for name, fp in data_files.items():
        ds_artifact.add_file(fp, name=name + ".csv")
    run.log_artifact(ds_artifact)

    # Get embeddings and labels
    X_train, y_train = (literal_eval(datasets['train'][embeds_col]),
                        datasets['train']['labels'])

    X_val, y_val = (literal_eval(datasets['validation'][embeds_col]),
                    datasets['validation']['labels'])

    # Run grid search
    rf_clf_optimal, rf_clf_score, rf_clf_params = tune_model(
        RandomForestClassifier(),
        (X_train.toarray(), y_train),
        search_params=hyperparam_grid
    )

    y_probas = rf_clf_optimal.predict_proba(X_val)

    # Run validation
    wandb.sklearn.plot_classifier(
        rf_clf_optimal,
        X_train,
        X_val,
        y_train,
        y_val,
        y_probas,
        labels=['negative', 'neutral', 'positive']
    )

    # Save and log model to W&B
    rf_model_save_path = MODEL_SAVE_DIR / (f'rf_clf_{datetime.now()}.joblib')
    joblib.dump(rf_clf_optimal, rf_model_save_path)

    trained_model_artifact = wandb.Artifact(
        'Random Forest Sentiment Classifier',
        type='model',
        description='Trained random forest classifier for sentiment analysis'
    )

    trained_model_artifact.add_dir(MODEL_SAVE_DIR)
    run.log(trained_model_artifact)



@app.command
def predict():
    pass
    # Pull latest model or selected model
    # Make prediction (inference)
    # Log results
