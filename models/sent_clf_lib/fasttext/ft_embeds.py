import typer
import pandas as pd
from datetime import datetime
from gensim.models import FastText
from pathlib import Path
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
    logger.info("Training Fast-Text model")
    fast_text_model.train(
        corpus_iterable=corpus_iter,
        epochs=fast_text_model.epochs,
        total_examples=fast_text_model.corpus_count,
        total_words=fast_text_model.corpus_total_words
    )

    # Save trained model
    file_name = f"ft_model_{datetime.now()}.model"
    fast_text_model.save(save_path / file_name)
    logger.info(f"Trained model saved to {save_path / file_name}")
