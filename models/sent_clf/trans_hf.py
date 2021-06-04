# TBD
# Create a Typer CLI for this so that I can easily run
# multiple different HF models
# Train on CoLab

import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Constants
MODEL_NAME = 'albert-base-v2'

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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples, text_col_name='text'):
    return tokenizer(examples[text_col_name],
                     padding='max_length',
                     truncation=True)


tokenized_datasets = all_datasets.map(tokenize_function, batched=True)

# Modelling
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

acc_metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return acc_metric.compute(predictions=predictions, references=labels)


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
