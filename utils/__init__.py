import pickle
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import multilabel_confusion_matrix, classification_report

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
