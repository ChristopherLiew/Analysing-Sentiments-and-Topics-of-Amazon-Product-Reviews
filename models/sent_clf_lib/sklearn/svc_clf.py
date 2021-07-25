"""
Support vector sentiment classifier using processed amazon product review data.
"""

## Import libraries
import pandas as pd
import logging
from joblib import dump, load
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from utils.embedding_vectoriser import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer
from utils import get_clf_results, tune_model, save_model, load_model

pd.set_option("display.width", 10000)
pd.set_option("display.max_columns", 10000)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

## Load train and test data
full_data = pd.read_csv("data/processed_data/proc_train.csv")
train_data = pd.read_csv("data/processed_data/proc_train.csv")
test_data = pd.read_csv("data/processed_data/proc_test.csv")

## Load synonym augmented training data
aug_train_data = pd.read_csv("data/aug_data/synonym_aug_proc_train_data.csv")

## Convert text into machine readable form
## 1) TF-IDF


def identity_tokenizer(text):
    return text


tfidf = TfidfVectorizer(
    tokenizer=identity_tokenizer, stop_words="english", lowercase=False
)

train_tfidf = tfidf.fit_transform(train_data.text_processed)
test_tfidf = tfidf.fit_transform(test_data.text_processed)

## 2) W2V
w2v = KeyedVectors.load_word2vec_format(
    "models/word_embed/weights/GoogleNews-vectors-negative300.bin", binary=True
)

w2v_mean_vectoriser = MeanEmbeddingVectorizer(w2v, "w2v")
w2v_train_data = w2v_mean_vectoriser.fit_transform(train_data)
w2v_test_data = w2v_mean_vectoriser.fit_transform(test_data)

## 3) FastText
## Fine tuned on Amazon Product Review data
ft = KeyedVectors.load("models/word_embed/weights/fast_text_model_2021-05-29.model")
ft_mean_vectoriser = MeanEmbeddingVectorizer(ft)
ft_train_data = ft_mean_vectoriser.fit_transform(train_data)
ft_test_data = ft_mean_vectoriser.fit_transform(test_data)

## Pre-trained on Wiki
ft_wiki = downloader.load("fasttext-wiki-news-subwords-300")

ft_wiki_mean_vectoriser = MeanEmbeddingVectorizer(ft_wiki, "w2v")
ft_wiki_train_data = ft_wiki_mean_vectoriser.fit_transform(train_data)
ft_wiki_test_data = ft_wiki_mean_vectoriser.fit_transform(test_data)

## Model development
## Hyperparameter Tuning
params_dict = {
    "kernel": ["linear"],
    "C": [0.5, 0.75, 1.0],
    "gamma": ["auto"],
    "class_weight": ["balanced"],
}

## Optimal SVC model with TFIDF
svc_clf_optimal, svc_clf_score, svc_clf_params = tune_model(
    SVC(), (train_data.toarray(), train_data.sentiment), search_params=params_dict
)

svc_tfidf_pred = svc_clf_optimal.predict(test_tfidf.toarray())
get_clf_results(test_data.sentiment.to_numpy(), svc_tfidf_pred)

# Results:
# -Accuracy: 0.482423
# -Weighted F1: 0.591577
# -Macro F1: 0.317604

# Save model
svc_tfidf_save_pth = "models/saved_models/svc_tfidf.joblib"
dump(svc_clf_optimal, svc_tfidf_save_pth)
load(svc_tfidf_save_pth)

## Optimal SVC model with w2v
svc_w2v = SVC(kernel="linear", gamma="auto", class_weight="balanced", verbose=1)
svc_w2v.fit(w2v_train_data, train_data.sentiment)

svc_w2v_pred = svc_w2v.predict(w2v_test_data)
get_clf_results(test_data.sentiment.to_numpy(), svc_w2v_pred)

# Results:
# -Accuracy: 0.720881
# -Weighted F1: 0.786658
# -Macro F1: 0.491202
# Improvement across the board in terms accuracy & F1 (weighted & macro) as well as across all sentiment classes

# Save model
svc_w2v_save_pth = "models/saved_models/svc_w2v.joblib"
dump(svc_w2v, svc_w2v_save_pth)
load(svc_w2v_save_pth)

## Optimal SVC model with FastText tuned on review data
svc_fast_txt = SVC(kernel="linear", gamma="auto", class_weight="balanced", verbose=1)
svc_fast_txt.fit(ft_train_data, train_data.sentiment)

svc_fast_txt_pred = svc_fast_txt.predict(ft_test_data)
get_clf_results(test_data.sentiment.to_numpy(), svc_fast_txt_pred)

# Results:
# -Accuracy: 0.84371
# -Weighted F1: 0.833509
# -Macro F1: 0.348814
# Overall improvement (Esp. for positive) but significant decrease in F1 (Esp. Recall) for negative sentiments
# Observable tradeoff between majority positive class against other minority classes.

# Save model
svc_ft_save_pth = "models/saved_models/svc_ft.joblib"
dump(svc_fast_txt, svc_ft_save_pth)
load(svc_ft_save_pth)

## Optimal SVC model with FastText (Wiki)
svc_fast_txt_wiki = SVC(
    kernel="linear", gamma="auto", class_weight="balanced", verbose=1
)
svc_fast_txt_wiki.fit(ft_wiki_train_data, train_data.sentiment)

svc_fast_txt_loaded_pred = svc_fast_txt_wiki.predict(ft_wiki_test_data)
get_clf_results(test_data.sentiment.to_numpy(), svc_fast_txt_loaded_pred)

# Results:
# -Accuracy: 0.713116
# -Weighted F1: 0.781780
# -Macro F1: 0.487965
# Similar results to preloaded w2v from GoogleNews vectors. Slightly better negative sentiment accuracy whilst trading off
# with a slightly poorer positive and neutral class accuracy

# Save model
svc_ft_wiki_save_pth = "models/saved_models/svc_ft_wiki.joblib"
dump(svc_fast_txt_wiki, svc_ft_wiki_save_pth)
load(svc_ft_wiki_save_pth)

### SVC Conclusion ###
# Generally tradeoff between 3 classes. Word vectors increase accuracy and F1 significantly and w2v seems to be the best
# compromise in terms of F1-Macro. Limiting factor remains imbalanced data with ~95% being positive sentiment.
# Trained word embeddings generally over-fit on words and semantics of the majority class = Positive.
# Test with augmented datasets to see if results improve.

## Improving on our Word Embeddings
# w2v
w2v_tfidf_vectoriser = TfidfEmbeddingVectorizer(w2v, "w2v")
w2v_tfidf_train_data = w2v_tfidf_vectoriser.fit_transform(train_data)
w2v_tfidf_test_data = w2v_tfidf_vectoriser.fit_transform(test_data)

# fasttext loaded
fasttext_tfidf_vectoriser = TfidfEmbeddingVectorizer(ft_wiki)
ft_tfidf_train_data = w2v_tfidf_vectoriser.fit_transform(train_data)
ft_tfidf_test_data = w2v_tfidf_vectoriser.fit_transform(test_data)

svc_w2v_tfidf_clf_optimal, svc_w2v_tfidf_score, svc_w2v_tfidf_params = tune_model(
    SVC(), (w2v_tfidf_train_data, train_data.sentiment), search_params=params_dict
)

amz_svc_w2v_tfidf_pred = svc_w2v_tfidf_clf_optimal.predict(w2v_tfidf_test_data)
get_clf_results(test_data.sentiment.to_numpy(), amz_svc_w2v_tfidf_pred)

# SVC Results with w2v
# - Accuracy = 0.722575
# - F2 Macro = 0.490665
# - F1 Weighted = 0.787518

svc_ft_tfidf_clf_optimal, svc_ft_tfidf_score, svc_ft_tfidf_params = tune_model(
    SVC(), (ft_tfidf_train_data, train_data.sentiment), search_params=params_dict
)

amz_svc_ft_tfidf_pred = svc_ft_tfidf_clf_optimal.predict(ft_tfidf_test_data)
get_clf_results(test_data.sentiment.to_numpy(), amz_svc_ft_tfidf_pred)

# SVC Results with fasttext
# - Accuracy = 0.722575
# - F2 Macro = 0.490665
# - F1 Weighted = 0.787518
# Same results as w2v
