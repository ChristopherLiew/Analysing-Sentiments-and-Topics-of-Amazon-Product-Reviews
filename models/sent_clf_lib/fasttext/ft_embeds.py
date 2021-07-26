import os
import typer
import pandas as pd
from datetime import datetime
from gensim.models import FastText
from pathlib import Path
import click_spinner as cs
from utils.logging import logger


# Instantiate Typer App
app = typer.Typer()


@app.command()
def train(data_path: str,
          output_dir: str,
          text_col_name: str = 'text',
          window: int = 5,
          min_count: int = 5
          ) -> None:
    """
    Trains a Fast Text sub-word embeddings model given text data.

    Args:
        data_path (str): Path to training data (in .csv format).
        output_dir (str): Path to directory to save FT model to.
        text_col_name (str, optional): Name of column containing text data for training. Defaults to 'text'.
        window (int, optional): Max. distance between current and predicted word. Defaults to 5.
        min_count (int, optional): Minimum word frequency for word to be considered. Defaults to 5.
    """

    # Paths
    save_path = Path(output_dir)

    # Load training data
    logger.info(f"Loading raw training data from {data_path}")
    corpus = pd.read_csv(data_path)

    # Instantiate Fast Text model
    fast_text_model = FastText(window=window,
                               min_count=min_count)

    # Build Vocab
    logger.info("Building vocabulary")
    corpus_iter = [[i] for i in corpus[text_col_name].to_list()]
    fast_text_model.build_vocab(corpus_iterable=corpus_iter)

    # Train FT model
    logger.info("Training Fast-Text model:")
    with cs.spinner():
        fast_text_model.train(
            corpus_iterable=corpus_iter,
            epochs=fast_text_model.epochs,
            total_examples=fast_text_model.corpus_count,
            total_words=fast_text_model.corpus_total_words
        )

    # Save trained model
    updated_dir = save_path / f"trained_ft_model_{datetime.now()}"
    os.mkdir(str(updated_dir))
    file_path = str(updated_dir / "ft_model.model")
    fast_text_model.save(file_path)
    logger.info(f"Trained model saved to {file_path}")
