"""
Huggingface Sequence Classifier Helper / Utility Functions.
"""
import numpy as np
from typing import Callable, Dict, Any
from datasets import load_metric
from transformers import EvalPrediction
import torch
from torch.utils.data import Dataset
from pathlib import Path


# Load Metrics
acc_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
prec_metric = load_metric("precision")
rec_metric = load_metric("recall")


def compute_clf_metrics(eval_pred: EvalPrediction) -> Callable[[EvalPrediction], Dict]:

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    metrics = dict()

    metrics["accuracy"] = acc_metric.compute(predictions=predictions, references=labels)

    metrics["f1"] = f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    metrics["precision"] = prec_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    metrics["recall"] = rec_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return metrics


def get_data_files(data_dir_root: Path, format: str = "csv") -> Dict[str, str]:
    data_files = dict()
    train_path, val_path, test_path = (
        f"train.{format}",
        f"validation.{format}",
        f"test.{format}",
    )
    data_files["train"] = str(data_dir_root / train_path)
    data_files["validation"] = str(data_dir_root / val_path)
    data_files["test"] = str(data_dir_root / test_path)
    return data_files


class InferenceDataset(Dataset):
    def __init__(self, encodings, labels=None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])
