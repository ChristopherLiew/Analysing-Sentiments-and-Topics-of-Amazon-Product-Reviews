# TBD
# 1) Create a Typer CLI for this so that I can easily train models
# 2) Add in Accelerate
# 3) Add in Weights and Biases
# 4) Test on CoLab

import typer
import torch
import numpy as np
from typing import Callable, Dict
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
    TensorBoardCallback,
    WandbCallback,
)
from pathlib import Path


# Instatiate Typer App
app = typer.Typer()


# Train hugging face sequence classifier
# Params:
# Model path
# File path

@app.command()
def train(
    model_name: str,
    data_dir: str
):
    """
    Train a Hugging Face Sequence Classifer.

    Args:\n
        model_name (str): Name of a valid hugging face classifier found on HF Hub.\n
        data_dir (str): Path to a data directory containing train, val and test data in\n
        in the following structure:\n
        |__ data_dir\n
            |__ train.csv\n
            |__ test.csv\n
            |__ validation.csv\n
    """
    typer.echo(f"Training a {model_name} Sequence Classifier on data from {data_dir}")


# # Constants
# PRE_TRAINED_MODEL_NAME = 'albert-base-v2'
# ROOT = Path('data/processed_hf')
# MODEL_SAVE_DIR = Path('store/sent_clf_models')

# # Check system for GPU
# device = torch.device("cuda") if torch.cuda.is_available()\
#      else torch.device("cpu")
# print(f'Device type: {device}')

# # Dataset filepaths
# data_files = dict()
# data_files['train'] = str(ROOT / 'train.csv')
# data_files['validation'] = str(ROOT / 'validation.csv')
# data_files['test'] = str(ROOT / 'test.csv')


# # Load raw dataset and train-test split
# datasets = load_dataset('csv', data_files=data_files)

# # Tokenizer and function
# tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# def tokenize_function(examples):
#     return tokenizer(examples['text'],
#                      padding='max_length',
#                      truncation=True)


# tokenized_datasets = datasets.map(
#     tokenize_function,
#     batched=True
# )

# # Data Collator with Padding (For Dynamic Padding)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# # Modelling
# model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, 
#                                                            num_labels=3)

# # Metrics
# acc_metric = load_metric('accuracy')
# f1_metric = load_metric('f1')
# prec_metric = load_metric('precision')
# rec_metric = load_metric("recall")


# def compute_metrics(eval_pred: EvalPrediction
#                     ) -> Callable[[EvalPrediction], Dict]:
#     # Use glue
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     metrics = {
#         'accuracy': acc_metric.compute(predictions=predictions,
#                                        references=labels),
#         'f1': f1_metric.compute(predictions=predictions,
#                                 references=labels),
#         'precision': prec_metric.compute(predictions=predictions,
#                                          references=labels),        
#         'recall': rec_metric.compute(predictions=predictions,
#                                      references=labels)
#         }
#     return metrics


# # Training
# training_args = TrainingArguments(
#     output_dir=MODEL_SAVE_DIR,
#     evaluation_strategy='epoch',
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     learning_rate=5e-5,
#     weight_decay=0.01
# )


# # Callbacks
# tb_cb = TensorBoardCallback()
# wandb_cb = WandbCallback()
# callbacks = [tb_cb, wandb_cb]


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['validation'],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=callbacks
# )

# # Train model
# trainer.train()

# # Evaluate model
# trainer.evaluate()
