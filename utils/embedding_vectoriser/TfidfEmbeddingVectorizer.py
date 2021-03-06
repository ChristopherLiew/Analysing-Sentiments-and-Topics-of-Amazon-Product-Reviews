import numpy as np
from ast import literal_eval
from tqdm import tqdm
from time import sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from collections import defaultdict


class TfidfEmbeddingVectorizer(TransformerMixin):
    def __init__(self, model, model_type=None, string_input=True):
        self.model = model
        self.word2weight = None
        self.vector_dims = model.vector_size
        self.model_type = model_type
        self.string_input = string_input

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]
        )
        return self

    def transform(self, X):
        new_corpus = []
        progress_bar = tqdm(total=len(X))
        for row in X.itertuples():
            sleep(0.01)
            orig_doc = literal_eval(row[1]) if self.string_input else row[1]
            # filter out unseen words for w2v model
            if self.model_type == "w2v":
                # Numpy zeros if None/ NaN reviews
                doc = np.mean(
                    [
                        self.model[word] * self.word2weight[word]
                        for word in orig_doc
                        if word in self.model.vocab
                    ]
                    or [np.zeros(self.vector_dims)],
                    axis=0,
                )
            else:
                doc = np.mean(
                    [self.model.wv[word] * self.word2weight[word] for word in orig_doc]
                    or [np.zeros(self.vector_dims)],
                    axis=0,
                )
            new_corpus.append(doc)
            progress_bar.update(1)
        progress_bar.close()
        return np.vstack(new_corpus)
