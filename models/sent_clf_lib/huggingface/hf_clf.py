"""
Huggingface Sequence Classifier.
"""
# TBD
# 1) Add classification metrics for test dataset + log to W and B
# 2) Figure out how to run remotely on colab
# 3) # Add in Typer Options (Argument with a flag) with Prompt

import os
import typer
import torch
import wandb
import numpy as np
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Union
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import TensorBoardCallback
from pathlib import Path
from utils.hf_clf import get_data_files, compute_clf_metrics, InferenceDataset

# Instatiate Typer App
app = typer.Typer()


WANDB_RUN_NAME = "amz-hf-sent-clf_" + str(datetime.now())
WANDB_HF_PROJ_TAGS = ["Test-Run", "HuggingFace"]


@app.command()  # Add in Typer Options (Argument with a flag) with Prompt
def train(
    model_name: str,
    data_dir: str,
    num_labels: int = 3,
    text_col: str = "text",
    wandb_entity: str = "chrisliew",
    wandb_proj_name: str = "amz-sent-analysis-deep-learning",
    wandb_run_name: str = WANDB_RUN_NAME,
    wandb_proj_tags: List[str] = WANDB_HF_PROJ_TAGS,
) -> None:
    """
    Train a Huggingface Sequence Classifer.

    Args:\n
        model_name (str): Name of a valid pre-trained hugging face model found on HF Hub.\n
        data_dir (str): Path to a data directory containing train, val and test data in\n
        in the following structure:\n
        └─── data_dir\n
            ├─── train.csv\n
            ├─── test.csv\n
            └─── validation.csv\n
        num_labels (int): Number of labels (target classes).\n
        text_col (str): Name of column containing text for modelling.\n
        wandb_entity (str, optional): W and B username. Defaults to 'chrisliew'\n
        wandb_proj_name (str, optional): W and B project name. Defaults to 'amz-sent-analysis'.\n
        wandb_run_name (str, optional): W and B run name. Defaults to 'amz-hf-sent-clf'.
        wandb_proj_tags (List[str], optional): W and B project tags. Defaults to ['Test-Run', 'Albert-V2'].
    """
    typer.echo(f"Training a {model_name} Sequence Classifier on data from {data_dir}")

    # Log into W&B
    wandb.login()

    # Initialize W and B run
    # see: https://docs.wandb.ai/guides/integrations/huggingface#getting-started-track-and-save-your-models
    os.environ["WANDB_LOG_MODEL"] = "true"  # Logs model as an artifact
    os.environ["WANDB_WATCH"] = "all"  # Logs Gradients and Params

    # Add model name to project tags
    wandb_proj_tags.append(model_name.upper())

    # Initialize WandB run
    run = wandb.init(
        project=wandb_proj_name,
        name=wandb_run_name,
        tags=wandb_proj_tags,
        entity=wandb_entity,
        job_type="training",
    )

    # Constants
    PRE_TRAINED_MODEL_NAME = model_name
    ROOT = Path(data_dir)
    MODEL_SAVE_DIR = Path("logs/hf_clf")

    # Check system for GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    typer.secho(f"Device type: {device}", fg=typer.colors.CYAN)

    # Dataset filepaths
    data_files = get_data_files(ROOT)

    # Load raw dataset and train-test split
    typer.echo(f"loading train, val and test data from {data_files}")

    datasets = load_dataset("csv", data_files=data_files)

    # Log datasets as Artifacts to W and B
    typer.echo("logging train, val and test datasets to W and B:")

    ds_artifact = wandb.Artifact(
        name=wandb_proj_name + "_datasets",
        type="datasets",
        description="""Processed train, val and test data
            for hugging face sequence clf models.""",
        metadata={"sizes": [v.num_rows for k, v in datasets.items()]},
    )

    for name, fp in data_files.items():
        ds_artifact.add_file(fp, name=name + ".csv")
    run.log_artifact(ds_artifact)

    # Load Tokenizer and Tokenize
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples[text_col], padding="max_length", truncation=True)

    typer.echo("Tokenizing data:")

    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Data Collator with Padding (For Dynamic Padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME, num_labels=num_labels
    )

    # Training (Change to **kwargs)
    training_args = TrainingArguments(
        report_to="wandb",
        output_dir=MODEL_SAVE_DIR,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        fp16=True if device == "cuda" else False,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=100,
        remove_unused_columns=True,
        run_name=wandb_run_name,
    )

    # Callbacks
    tb_cb = TensorBoardCallback()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_clf_metrics,
        callbacks=[tb_cb]
        )

    # Train model
    typer.secho("Starting up model training ...", fg=typer.colors.YELLOW)
    # trainer.train()

    # Evaluate model
    typer.secho("Starting up model evaluation ...", fg=typer.colors.YELLOW)
    trainer.evaluate()

    # End W and B session
    wandb.finish()

    typer.secho("Training and Evaluation Completed", fg=typer.colors.GREEN)


# Model Inference (Adapt to allow local model loading)
@app.command()
def predict(
    model_name: Optional[str],
    wandb_run_name: str,
    wandb_entity: str = "chrisliew",
    wandb_proj_name: str = "amz-sent-analysis-deep-learning",
    num_labels: int = 3,
    inf_data = None,  # Typer cannot support Any and Nested Dicts
    text_col: Optional[str] = "text"
) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
    """
    Performs inference on a given set of test data using the latest fine tuned or
    pretrained huggingface classifier trained on our training data.
    Args:\n
        inf_data (Optional[str]): Path to test data in the following format {'test': filepath}.
        If None, will pull latest test data artifact fro W and B.\n
        model_name (Optional[str]): Name of huggingface model used (Must be a valid HF Hub model) that
        is consistent with the models trained and stored in the given W and B project and run.\n
        wandb_proj_name (str): W and B project name to pull model and data from.\n
        wandb_run_name (str): W and B run name within the project.\n
        num_labels (int): Number of labels.\n
        text_col (Optional[str]): Name of text column. Defaults to 'text'.\n

    Returns:\n
        List[...]: Huggingface sequence classifier inference output.
    """

    with wandb.init(project=wandb_proj_name, job_type="inference") as run:

        # Load latest trained model (To be tested)
        my_model_name = f"{wandb_run_name}:latest"
        my_model_artifact = run.use_artifact(my_model_name)
        model_dir = my_model_artifact.download()

        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=num_labels
        )

        # Load and process test data
        if inf_data is not None:
            test_data = load_dataset("csv", data_files=inf_data)
        else:
            # Pull latest test dataset from W and B
            ds_wnb_fp = os.path.join(
                wandb_entity, wandb_proj_name, wandb_proj_name, "_datasets:latest"
            )
            artifacts = run.use_artifact(ds_wnb_fp, type="datasets")
            artifacts_dir = Path(artifacts.download())
            data_files = get_data_files(artifacts_dir)
            # Format artifacts_dir into train, val and test dict
            test_data = load_dataset("csv", data_files=data_files)["test"]

        # Preprocess data
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_test_data = tokenizer.batch_encode_plus(
            test_data[text_col], truncation=True, padding="max_length"
        )

        # Make Inference
        test_dataset = InferenceDataset(tokenized_test_data)
        test_trainer = Trainer(model)
        raw_pred, _, _ = test_trainer.predict(test_dataset)
        y_pred = np.argmax(raw_pred, axis=1)
        y_true = test_data["label"]

        # Log Test Results to W and B? with CLF results?
        return y_true, y_pred
