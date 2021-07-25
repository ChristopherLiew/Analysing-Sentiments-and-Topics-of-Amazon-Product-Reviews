import os
import time
import pickle
from numpy.core.records import array
import pandas as pd
from time import sleep
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator
from ast import literal_eval
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from gensim import models
from tensorflow.keras import utils


def get_clf_results(y_true: array, y_pred: array) -> pd.DataFrame:
    print(multilabel_confusion_matrix(y_true, y_pred))
    return pd.DataFrame(classification_report(y_true, y_pred, 
                                              output_dict=True))


def tune_model(model: BaseEstimator, data: pd.DataFrame,
               search_params: Dict, verbosity: bool = True,
               n_jobs: int = -1) -> Tuple:

    tuner = RandomizedSearchCV(
        model,
        search_params,
        verbose=verbosity,
        n_jobs=n_jobs)

    train_data, train_labels = data
    search = tuner.fit(train_data, train_labels)

    return search.best_estimator_, search.best_score_, search.best_params_


def save_model(model: BaseEstimator, filepath: str) -> None:
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
        print('Model successfully saved at: ' + filepath)


def load_model(filepath: str):
    with open(filepath, 'rb') as file:
        print('Loading model from: ' + filepath)
        return pickle.load(file)


def convert_to_lists(string_data: str) -> List:
    '''
    Use when converting back from csv data.
    '''
    return [literal_eval(i) for i in string_data]


def create_bigrams(text: List[str], min_count: int = 20) -> List:
    sleep(0.1)
    # Higher min_words and threshold-> Harder to form bigrams
    bigram = models.Phrases(text, min_count)
    # Creates a memory efficient model of phrases w/o model state
    bigram_mod = models.phrases.Phraser(bigram)
    # Transform doc to bigrams
    return [bigram_mod[doc] for doc in text]  


def create_trigrams(bigrams: List[str], min_count: int = 5) -> List:
    trigram = models.Phrases(bigrams, min_count)
    trigram_mod = models.phrases.Phraser(trigram)
    return [trigram_mod[doc] for doc in bigrams]


def get_log_dir() -> str:
    root_log_dir = os.path.join(os.curdir, "Transformer Models/my_logs")
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_log_dir, run_id)
