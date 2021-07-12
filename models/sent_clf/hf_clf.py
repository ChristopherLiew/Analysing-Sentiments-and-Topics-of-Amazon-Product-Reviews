# TBD
# 1) Add in Accelerate
# 2) Figure out how to run remotely on colab
import os
import typer
import torch
import numpy as np
import wandb
import logging
from datetime import datetime
from typing import Callable, Dict, List
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.integrations import (
    TensorBoardCallback
)
from pathlib import Path


# Instatiate Typer App
app = typer.Typer()


@app.command()
def train(
    model_name: str,
    data_dir: str,
    num_labels: int = 3,
    text_col: str = 'text',
    wandb_entity: str = 'chrisliew',
    wandb_proj_name: str = 'amz-sent-analysis',
    wandb_run_name: str = 'amz-hf-sent-clf',
    wandb_proj_tags: List[str] = ['Test-Run', 'Albert-V2']
    ) -> None:
    """
    Train a Hugging Face Sequence Classifer.

    Args:\n
        model_name (str): Name of a valid pre-trained hugging face model found on HF Hub.\n
        data_dir (str): Path to a data directory containing train, val and test data in\n
        in the following structure:\n
        |__ data_dir\n
            |__ train.csv\n
            |__ test.csv\n
            |__ validation.csv\n
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

    # Format run name
    run_name = wandb_proj_name + '-' + str(datetime.now())

    # Initialize W and B run
    # see: https://docs.wandb.ai/guides/integrations/huggingface#getting-started-track-and-save-your-models
    os.environ['WANDB_LOG_MODEL'] = 'true'  # Logs model as an artifact
    os.environ['WANDB_WATCH'] = 'all'  # Logs Gradients and Params
    wandb.init(project=wandb_proj_name,
               name=run_name,
               tags=wandb_proj_tags,
               entity=wandb_entity)

    # Constants
    PRE_TRAINED_MODEL_NAME = model_name
    ROOT = Path(data_dir)
    MODEL_SAVE_DIR = Path('logs/hf_clf')

    # Check system for GPU
    device = torch.device("cuda") if torch.cuda.is_available()\
        else torch.device("cpu")

    typer.secho(f'Device type: {device}', fg=typer.colors.CYAN)

    # Dataset filepaths
    data_files = dict()
    data_files['train'] = str(ROOT / 'train.csv')
    data_files['validation'] = str(ROOT / 'validation.csv')
    data_files['test'] = str(ROOT / 'test.csv')

    # Load raw dataset and train-test split
    logging.info(f'loading train, val and test data from {data_files}')
    datasets = load_dataset('csv', data_files=data_files)

    # Load Tokenizer and Tokenize
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


    def tokenize_function(examples):
        return tokenizer(examples[text_col],
                        padding='max_length',
                        truncation=True)

    typer.echo(f'Tokenizing data:')
    
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True
    )

    # Data Collator with Padding (For Dynamic Padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=num_labels
    )

    # Load Metrics
    acc_metric = load_metric('accuracy')
    f1_metric = load_metric('f1')
    prec_metric = load_metric('precision')
    rec_metric = load_metric("recall")


    def compute_metrics(eval_pred: EvalPrediction
                        ) -> Callable[[EvalPrediction], Dict]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = {
            'accuracy': acc_metric.compute(predictions=predictions,
                                           references=labels),
            'f1': f1_metric.compute(predictions=predictions,
                                    references=labels,
                                    average='macro'),
            'precision': prec_metric.compute(predictions=predictions,
                                            references=labels,
                                            average='macro'),
            'recall': rec_metric.compute(predictions=predictions,
                                        references=labels,
                                        average='macro')
        }
        return metrics


    # Training
    training_args = TrainingArguments(
        report_to='wandb',
        output_dir=MODEL_SAVE_DIR,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        fp16=True if device == 'cuda' else False,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=100,
        remove_unused_columns=True,
        run_name=wandb_run_name  # Turn into argument for CLI
    )

    # Callbacks
    tb_cb = TensorBoardCallback()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[tb_cb]
    )

    # Train model
    typer.echo(f'Starting up model training ...')
    # trainer.train()

    # Evaluate model
    typer.echo(f'Starting up model evaluation ...')
    trainer.evaluate()

    # End W and B session
    wandb.finish()
    
    typer.secho('Training and Evaluation Completed', fg=typer.colors.GREEN)



# Model Inference
# @app.command()
# def predict(test_data: str,
#             wandb_proj_name: str,
#             wandb_model_name: str
#             ):
#     pass

# with wandb.init(project='amz-sent-analysis') as run:

#   # Connect an Artifact to the run
#   my_model_name = "amz-pdt-reviews-sent-clf:latest"
#   my_model_artifact = run.use_artifact(my_model_name)

#   # Download model weights to a folder and return the path
#   model_dir = my_model_artifact.download()

#   # Load your Hugging Face model from that folder
#   #  using the same model class
#   model = AutoModelForSequenceClassification.from_pretrained(
#       model_dir, num_labels=num_labels)

#   # Do additional training, or run inference
