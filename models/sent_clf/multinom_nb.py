"""
Multinomial naive bayes sentiment classifier using processed amazon product review data.
"""
# TBD:
# 1) Refactor to suit new dataset format
# 2) train and save fast text

## Import libraries
import pandas as pd
import logging
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.embedding_vectoriser import MeanEmbeddingVectorizer
from utils import get_clf_results, tune_model, save_model, load_model
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10000)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
gnb_clf = MultinomialNB()
gnb_clf.fit(train_tfidf.toarray(), train_data.sentiment)
gnb_pred = gnb_clf.predict(test_tfidf.toarray())
get_clf_results(test_data.sentiment.to_numpy(), gnb_pred)
