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
    AlbertConfig,  # Change for different models
    Trainer,
    TrainingArguments,
    EvalPrediction
)

# Constants
PRE_TRAINED_MODEL_NAME = 'albert-base-v2'

device = torch.device("cuda") if torch.cuda.is_available()\
     else torch.device("cpu")
print(f'Device type: {device}')

# Load raw dataset and train-test split
full_raw_datasets = load_dataset('csv',
                                 data_files={'train':
                                             'data/raw_data/raw_train.csv',
                                             'val':
                                             'data/raw_data/raw_test.csv'})

test_val_datasets = full_raw_datasets['val'].train_test_split(test_size=0.5)
test_val_datasets['val'] = test_val_datasets['train']
test_val_datasets.pop('train')
all_datasets = test_val_datasets
all_datasets['train'] = full_raw_datasets['train']

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


pretrained_config = AlbertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME,
                                                 num_labels=3,
                                                 id2label={0: -1, 1: 0, 2: 1},
                                                 label2id={-1: 0, 0: 1, 1: 2})


def tokenize_function(examples: str, text_col_name: str = 'text'):
    return tokenizer(examples[text_col_name],
                     padding='max_length',
                     truncation=True)


tokenized_datasets = all_datasets.map(tokenize_function, batched=True)

# Modelling
model = AutoModelForSequenceClassification.from_pretrained(
    PRE_TRAINED_MODEL_NAME,
    config=pretrained_config
    )

# Metrics
acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
prec_metric = load_metric('precision')
rec_metric = load_metric("recall")


def compute_metrics(eval_pred: EvalPrediction) -> Callable[[EvalPrediction], Dict]:
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
train_args = TrainingArguments(
    output_dir='models/saved_models',
    evaluation_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    compute_metrics=compute_metrics
)

trainer.train()

# Simple Evaluation
trainer.evaluate()
