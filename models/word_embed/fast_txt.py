"""
Fine tune and create fast text word embeddings using review data.
"""

# Import libraries
import os
import pandas as pd
from numpy.core.numeric import full
from datetime import datetime
from ast import literal_eval
from tqdm import tqdm
from gensim.models import FastText
from gensim.test.utils import get_tmpfile


def train_ft():
    # Load train and test data
    full_data = pd.read_csv("data/processed_data/proc_full_untok.csv")

    # Create an untokenized dataset (should just create a full proc but untok dataset)
    # tqdm.pandas(desc="Progress bar")
    # full_untok_data = pd.DataFrame(
    #     full_data.progress_apply(lambda x: str(' '.join(literal_eval(x['reviews']))), axis=1),
    #     columns=['text_processed'])
    # full_untok_data.to_csv('data/processed_data/proc_full.csv')

    # Train FastText Model
    fast_text_model = FastText(window=5, min_count=5)

    # Build vocab
    corpus_iter = [[i] for i in full_data["text_processed"].to_list()]
    fast_text_model.build_vocab(corpus_iterable=corpus_iter)

    # Train model
    fast_text_model.train(
        corpus_iterable=corpus_iter,
        epochs=fast_text_model.epochs,
        total_examples=fast_text_model.corpus_count,
        total_words=fast_text_model.corpus_total_words,
    )

    # Save trained model
    fname = f"fast_text_model_{datetime.now().strftime(format='%Y-%m-%d')}.model"
    save_path = os.path.join("models/word_embed/weights", fname)
    fast_text_model.save(save_path)


if __name__ == "__main__":
    print(
        "Training fast text word embeddings model using processed amazon product review data"
    )
    train_ft()
    print("Training complete")
