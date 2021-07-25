"""
Random forest sentiment classifier using processed amazon product review data.
"""
# 1) Create CLI for training and inference
# - Process Training Data
# - Train Model + Log results
# - Make inference + Log eval results

## Import libraries
import pandas as pd
import logging
from joblib import dump, load
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from utils.embedding_vectoriser import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer
from utils import get_clf_results, tune_model, save_model, load_model
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10000)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## Load train and test data
full_data = pd.read_csv('data/processed_data/proc_train.csv')
train_data = pd.read_csv('data/processed_data/proc_train.csv')
test_data = pd.read_csv('data/processed_data/proc_test.csv')

## Load synonym augmented training data
aug_train_data = pd.read_csv('data/aug_data/synonym_aug_proc_train_data.csv')

## Convert text into machine readable form
## 1) TF-IDF
def identity_tokenizer(text):
    return text


tfidf = TfidfVectorizer(tokenizer=identity_tokenizer,
                        stop_words='english', lowercase=False)

train_tfidf = tfidf.fit_transform(train_data.text_processed)
test_tfidf = tfidf.fit_transform(test_data.text_processed)

## 2) W2V
w2v = KeyedVectors.load_word2vec_format(
    'models/word_embed/weights/GoogleNews-vectors-negative300.bin',
    binary=True)

w2v_mean_vectoriser = MeanEmbeddingVectorizer(w2v, 'w2v')
w2v_train_data = w2v_mean_vectoriser.fit_transform(train_data)
w2v_test_data = w2v_mean_vectoriser.fit_transform(test_data)

## 3) FastText
## Fine tuned on Amazon Product Review data
ft = KeyedVectors.load(
    'models/word_embed/weights/fast_text_model_2021-05-29.model'
)
ft_mean_vectoriser = MeanEmbeddingVectorizer(ft)
ft_train_data = ft_mean_vectoriser.fit_transform(train_data)
ft_test_data = ft_mean_vectoriser.fit_transform(test_data)

## Pre-trained on Wiki
ft_wiki = downloader.load('fasttext-wiki-news-subwords-300')

ft_wiki_mean_vectoriser = MeanEmbeddingVectorizer(ft_wiki, 'w2v')
ft_wiki_train_data = ft_wiki_mean_vectoriser.fit_transform(train_data)
ft_wiki_test_data = ft_wiki_mean_vectoriser.fit_transform(test_data)

## Model development
## Hyperparams
params_dict_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'class_weight': ['balanced']
}

rf_clf_optimal, rf_clf_score, rf_clf_params = tune_model(RandomForestClassifier(),
                                                         (train_tfidf.toarray(), train_data.sentiment),
                                                         search_params=params_dict_rf)

amz_pred_rf = rf_clf_optimal.predict(test_tfidf.toarray())
get_clf_results(test_data.sentiment.to_numpy(), amz_pred_rf)

# Results:
# -Accuracy: 0.649584
# -Weighted F1: 0.725631
# -Macro F1: 0.360333
# Strong improvement over SVC, slight dip in negative sentiment predictions.

# Save model
rf_tfidf_save_pth = 'models/saved_models/rf_tfidf.joblib'
dump(rf_clf_optimal, rf_tfidf_save_pth)
load(rf_tfidf_save_pth)

# Optimal RF models with Word Embeddings
# a. Word2Vec
rf_w2v_optimal, rf_w2v_score, rf_w2v_params = tune_model(RandomForestClassifier(),
                                                         (w2v_train_data,
                                                          train_data.sentiment),
                                                         search_params=params_dict_rf)

rf_w2v_pred = rf_w2v_optimal.predict(w2v_test_data)
get_clf_results(test_data.sentiment.to_numpy(), rf_w2v_pred)

# Results:
# -Accuracy: 0.710716
# -Weighted F1: 0.775801
# -Macro F1: 0.460367
# Balanced results, with strong accuracy improvements in negative and neutral sentiment categories

# Save model
rf_w2v_save_pth = 'models/saved_models/rf_w2v.joblib'
dump(rf_w2v_optimal, rf_w2v_save_pth)
load(rf_w2v_save_pth)

# b. Fast Text Trained
rf_fasttxt_optimal, rf_fasttxt_score, rf_fasttxt_params = tune_model(RandomForestClassifier(),
                                                                     (ft_train_data,
                                                                      train_data.sentiment),
                                                                     search_params=params_dict_rf)

rf_fasttxt_pred = rf_fasttxt_optimal.predict(ft_test_data)
get_clf_results(test_data.sentiment.to_numpy(), rf_fasttxt_pred)

# Results:
# -Accuracy: 0.660737
# -Weighted F1: 0.735803
# -Macro F1: 0.393156
# Overall drop in performance across all sentiment categories and metrics versus word2vec model.

# Save model
rf_ft_save_pth = 'models/saved_models/rf_ft.joblib'
dump(rf_fasttxt_optimal, rf_ft_save_pth)
load(rf_ft_save_pth)

# c. Fast Text (Wiki)
rf_fasttxt_wiki_optimal, rf_fasttxt_wiki_score, rf_fasttxt_wiki_params = tune_model(RandomForestClassifier(),
                                                                                    (ft_wiki_train_data,
                                                                                     train_data.sentiment),
                                                                                    search_params=params_dict_rf)

rf_fasttxt_wiki_pred = rf_fasttxt_wiki_optimal.predict(ft_wiki_test_data)
get_clf_results(test_data.sentiment.to_numpy(), rf_fasttxt_wiki_pred)

# Results:
# -Accuracy: 0.740505
# -Weighted F1: 0.794781
# -Macro F1: 0.479646
# Improvements across the board (metrics and sentiment categories). Best results thus far in terms of absolute metric performance tempered by holistic performance
# across sentiment categories.

# Save model
rf_ft_wiki_save_pth = 'models/saved_models/rf_ft_wiki.joblib'
dump(rf_fasttxt_wiki_optimal, rf_ft_wiki_save_pth)
load(rf_ft_wiki_save_pth)

### RF Conclusion ###
# Strong results when using fast text wiki data, out performs SVC in terms of Macro F1 but performs more poorly in terms of accuracy vis a vis Fast Text Trained
# SVC model. This is largely due to SVC overfitting and RF's decision boundary tending towards a more balanced performance. RF with pre-trained Fast Text word Embeddings
# thus gives us the best results so far on an severely imbalanced dataset.

## Improving on our Word Embeddings
# w2v
w2v_tfidf_vectoriser = TfidfEmbeddingVectorizer(w2v, 'w2v')
w2v_tfidf_train_data = w2v_tfidf_vectoriser.fit_transform(train_data)
w2v_tfidf_test_data = w2v_tfidf_vectoriser.fit_transform(test_data)

# fasttext loaded
fasttext_tfidf_vectoriser = TfidfEmbeddingVectorizer(ft_wiki)
ft_tfidf_train_data = w2v_tfidf_vectoriser.fit_transform(train_data)
ft_tfidf_test_data = w2v_tfidf_vectoriser.fit_transform(test_data)

rf_w2v_tfidf_clf_optimal, rf_w2v_tfidf_score, rf_w2v_tfidf_params = tune_model(RandomForestClassifier(),
                                                                               (w2v_tfidf_train_data,
                                                                                train_data.sentiment),
                                                                               search_params=params_dict_rf)

amz_rf_w2v_tfidf_pred = rf_w2v_tfidf_clf_optimal.predict(w2v_tfidf_test_data)
get_clf_results(test_data.sentiment.to_numpy(), amz_rf_w2v_tfidf_pred)

# RF Results with w2v
# - Accuracy = 0.703515
# - F2 Macro = 0.456013
# - F1 Weighted = 0.770530

rf_ft_tfidf_clf_optimal, rf_ft_tfidf_score, rf_ft_tfidf_params = tune_model(RandomForestClassifier(),
                                                                            (ft_tfidf_train_data,
                                                                             train_data.sentiment),
                                                                            search_params=params_dict_rf)

amz_rf_ft_tfidf_pred = rf_ft_tfidf_clf_optimal.predict(ft_tfidf_test_data)
get_clf_results(test_data.sentiment.to_numpy(), amz_rf_ft_tfidf_pred)

# RF Results with fasttext
# - Accuracy = 0.704786
# - F2 Macro = 0.456618
# - F1 Weighted = 0.771132

### Tfidf vectorised Results ###
# SVC outperforms RF, however on the overall results were poorer when using TF-IDF embeddings vs simple Mean Embeddings.
# This might largely be due TF-IDF 'overfitting' or capturing too much noise from the dominant positive sentiment
# category. As such, mean embeddings are better able to denoise and predict sentiment more accurately, especially for
# the minority classes.
