# TBD
# 1) Create a Typer CLI for this so that I can easily run
# multiple different HF models
# -> Given a model name, use regex to identify model type (ALBERT, XLNET ...)
# -> Pull config and pass into SequenceClf obj
# -> Process data and Train model
# 2) Create custom metrics (F1-macro, etc)
# 3) Train models on CoLab for GPU speed up
# 4) Hyperparameter tuning
# -> https://www.anyscale.com/blog/hyperparameter-search-hugging-face-transformers-ray-tune

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
    EvalPrediction
)
from pathlib import Path

# Constants
PRE_TRAINED_MODEL_NAME = 'albert-base-v2'
ROOT = Path('data/processed_hf')
MODEL_SAVE_DIR = Path('store/sent_clf_models')

# Check system for GPU
device = torch.device("cuda") if torch.cuda.is_available()\
     else torch.device("cpu")
print(f'Device type: {device}')

# Dataset filepaths
data_files = dict()
data_files['train'] = ROOT / 'train/train.csv'
data_files['validation'] = ROOT / 'validation/validation.csv'
data_files['test'] = ROOT / 'test/test.csv'


# Load raw dataset and train-test split
datasets = load_dataset('csv', data_files=data_files)

# Tokenizer and function
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


def tokenize_function(examples: str, 
                      text_col_name: str = 'text'):

    return tokenizer(examples[text_col_name],
                     padding='max_length',
                     truncation=True)


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True
)

# Data Collator with Padding (For Dynamic Padding)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Modelling
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, 
                                                           num_labels=3)

# Metrics
acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
prec_metric = load_metric('precision')
rec_metric = load_metric("recall")


def compute_metrics(eval_pred: EvalPrediction) -> Callable[[EvalPrediction], Dict]:
    # Use glue
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = {
        'accuracy': acc_metric.compute(predictions=predictions,
                                       references=labels),

        'f1': f1_metric.compute(predictions=predictions,
                                references=labels),

        'precision': prec_metric.compute(predictions=predictions,
                                         references=labels),
                                         
        'recall': rec_metric.compute(predictions=predictions,
                                     references=labels)
        }
    return metrics


# Training
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_DIR,
    evaluation_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model
trainer.evaluate()
