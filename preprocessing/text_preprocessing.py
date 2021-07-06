import re
import string
import spacy
import pandas as pd
from tqdm import tqdm
from time import sleep
from nltk.corpus import stopwords
from spacy.tokenizer import Tokenizer
import unicodedata
from preprocessing import contractions

# A. Preprocessing Functions
# 1. Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# 2. Expand contractions
# Check norm dataset -> didnt wasn't caught
CONTRACTION_MAP = contractions.CONTRACTION_MAP
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# 3. Removing special characters
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

# 5. Tokenizer: Set nlp.tokenizer = custom_tokenizer(nlp)
def custom_tokenizer(nlp):
    prefix_re = re.compile(r'(?<=[:;()[\]+.,!?\\-])[A-Za-z]')
    suffix_re = re.compile(r'(?<=[A-Za-z])[:;()[\]+.,!?\\-]')
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search)

# 6. Remove stopwords & punctuation
all_stopwords = stopwords.words('english')
all_punctuation = string.punctuation

def lemmatize_and_remove_stopwords(text,
                                   nlp, 
                                   tokenize=True,
                                   is_lower_case=True):
    tokens = nlp(text)
    tokens = [token.text.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in all_stopwords]
        filtered_tokens = [token for token in filtered_tokens if token not in all_punctuation]
    else:
        filtered_tokens = [token.lower() for token in tokens if token not in all_stopwords]
        filtered_tokens = [token for token in filtered_tokens if token not in all_punctuation]
    if tokenize:
        return filtered_tokens
    else:
        return ' '.join(filtered_tokens)

# 7. Correct apostrophe whitespaces
def remove_apos_whitespace(text):
    leading = r"(?<=[a-zA-Z])\s+(?=[a-z]*'\s*[a-z])"
    trailing = r"(?<=[a-zA-Z]')\s+(?=[a-zA-Z])"
    text_intermediate = re.sub(leading, '', text)
    text_result = re.sub(trailing, '', text_intermediate)
    return text_result

# 8. Complete pipeline
def preprocess_text(data, 
                    remove_accented_char=True, 
                    contraction_expansion=True, 
                    normalize=True, 
                    correct_apos_whitespace=True, 
                    remove_special_char=True, 
                    lemma_and_remove_stop_words=True,
                    tokenize=True):
    processed_corpus = []
    corpus = data['reviews.text']
    labels = data['reviews.rating']
    nlp = spacy.load("en_core_web_sm")
    tokenizer = custom_tokenizer(nlp)
    nlp.tokenizer = tokenizer
    for text in tqdm(corpus):
        sleep(0.1)
        if remove_accented_char:
            text = remove_accented_chars(text)
        if remove_special_char:
            text = remove_special_characters(text)
        if correct_apos_whitespace:
            text = remove_apos_whitespace(text)
        if contraction_expansion:
            text = expand_contractions(text)
        if normalize:
            text = text.lower()
        if lemma_and_remove_stop_words:
            text = lemmatize_and_remove_stopwords(text, nlp, tokenize)
        processed_corpus.append(text)
    return pd.DataFrame({'reviews.text': processed_corpus, 'reviews.rating': labels})
