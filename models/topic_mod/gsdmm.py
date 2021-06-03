import pandas as pd
import numpy as np
import pickle
from gsdmm import MovieGroupProcess
from gensim import corpora
from utils import convert_to_lists

### Split dataset into POS, NEG, NEU ###
# Load existing ngram data from Topic_Modelling_LDA.py file:

amz_ngrams = pd.read_csv('data/topic_data/full_ngram.csv', index_col=False)

amz_ngrams_pos = list(amz_ngrams[amz_ngrams['sentiment'] == 'Positive']['reviews_trigrams'])
amz_ngrams_neu = list(amz_ngrams[amz_ngrams['sentiment'] == 'Neutral']['reviews_trigrams'])
amz_ngrams_neg = list(amz_ngrams[amz_ngrams['sentiment'] == 'Negative']['reviews_trigrams'])

amz_ngrams_pos = convert_to_lists(amz_ngrams_pos)
amz_ngrams_neu = convert_to_lists(amz_ngrams_neu)
amz_ngrams_neg = convert_to_lists(amz_ngrams_neg)

### Construct BoW corpus ###
# Create Dictionary & Filter very rare words
id2wordPos = corpora.Dictionary(amz_ngrams_pos)
id2wordPos.filter_extremes(no_below=20, no_above=0.5)  # Extra layer of filtering on top of stop words etc

id2wordNeu = corpora.Dictionary(amz_ngrams_neu)
id2wordNeu.filter_extremes(no_below=20, no_above=0.5)

id2wordNeg = corpora.Dictionary(amz_ngrams_neg)
id2wordNeg.filter_extremes(no_below=20, no_above=0.5)

# Convert to Bag of Words using Dictionary
corpus_pos = [id2wordPos.doc2bow(text) for text in amz_ngrams_pos]
corpus_neu = [id2wordNeu.doc2bow(text) for text in amz_ngrams_neu]
corpus_neg = [id2wordNeg.doc2bow(text) for text in amz_ngrams_neg]

### Build GSDMM model aka Movie Group Process ###
def top_words(model, dictionary, top_cluster, top_n_words):
    for cluster in top_cluster:
        sorted_dicts = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:top_n_words]
        decoded_dicts = [dictionary[i[0][0]] for i in sorted_dicts]
        print('Cluster %s : %s' % (cluster, decoded_dicts))

## Positive Corpus
mgp = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=100)
topics = mgp.fit(corpus_pos, len(id2wordPos))

doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# Topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[::-1]
print('Most important clusters (by number of docs inside):', top_index)

# Show the top 10 words in terms of frequency for each cluster
top_words(mgp, id2wordPos, top_index, 20)

# Save Positive Model
with open('models/saved_models/gsdmm_positive.model', 'wb') as f:
    pickle.dump(mgp, f)
    f.close()

## Neutral Corpus
mgp_neu = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=100)
topics_neu = mgp_neu.fit(corpus_neu, len(id2wordNeu))

doc_count = np.array(mgp_neu.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# Topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[::-1]
print('Most important clusters (by number of docs inside):', top_index)

# Show the top 10 words in terms of frequency for each cluster
top_words(mgp_neu, id2wordNeu, top_index, 20)

# Save Neutral Model
with open('models/saved_models/gsdmm_neutral.model', 'wb') as f:
    pickle.dump(mgp_neu, f)
    f.close()

## Negative Corpus
mgp_neg = MovieGroupProcess(K=2, alpha=0.1, beta=0.1, n_iters=100)
topics_neg = mgp_neg.fit(corpus_neg, len(id2wordNeg))

doc_count = np.array(mgp_neg.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# Topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[::-1]
print('Most important clusters (by number of docs inside):', top_index)

# Show the top 10 words in terms of frequency for each cluster
top_words(mgp_neg, id2wordNeg, top_index, 20)

# Save Negative Model
with open('models/saved_models/gsdmm_negative.model', 'wb') as f:
    pickle.dump(mgp_neg, f)
    f.close()
