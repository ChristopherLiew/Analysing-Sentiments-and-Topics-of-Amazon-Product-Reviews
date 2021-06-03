import pickle
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from gensim import models

def get_clf_results(y_true, y_pred):
    print(multilabel_confusion_matrix(y_true, y_pred))
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))


def tune_model(model, X, search_params, verbosity=True, n_jobs=-1):
    tuner = RandomizedSearchCV(
        model, search_params, verbose=verbosity, n_jobs=n_jobs)
    train_data, train_labels = X
    search = tuner.fit(train_data, train_labels)
    return search.best_estimator_, search.best_score_, search.best_params_


def save_model(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
        print('Model successfully saved at: ' + filepath)


def load_model(filepath):
    with open(filepath, 'rb') as file:
        print('Loading model from: ' + filepath)
        return pickle.load(file)

def convert_to_lists(string_data):
    '''
    Use when converting back from csv data.
    '''
    return [literal_eval(i) for i in string_data]


def create_bigrams(text, min_count=20):
    sleep(0.1)
    # Higher min_words and threshold-> Harder to form bigrams
    bigram = models.Phrases(text, min_count)
    bigram_mod = models.phrases.Phraser(
        bigram)  # Creates a memory efficient model of phrases w/o model state (Ok if we do not need to train on new docs)
    return [bigram_mod[doc] for doc in text]  # Transform doc to bigrams


def create_trigrams(bigrams, min_count=5):
    trigram = models.Phrases(bigrams, min_count)
    trigram_mod = models.phrases.Phraser(trigram)
    return [trigram_mod[doc] for doc in bigrams]
