import typer
import click_spinner as cs
from typing import (
    Optional,
    Dict,
)
import pandas as pd
from pathlib import Path
from gensim import downloader as api
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText, load_facebook_vectors
from utils.embedding_vectoriser import (
    MeanEmbeddingVectorizer,
    TfidfEmbeddingVectorizer
)
from utils.logging import logger

# Instatiate Typer App
app = typer.Typer()


# TBD:
# Run a full test
# Fast Text pretrained download not working for some reason


@app.command()
def create_embeds(
    data_dir: str,
    output_dir: str,
    embed_type: str = "w2v",
    pretrained_embed_dir: Optional[str] = typer.Option(None, help="Filepath or name of word embeddings to use"),  # Can be None if fast-text
    vectorisation_mode: str = "mean",
) -> Dict[str, pd.DataFrame]:

    datasets = dict()
    root_dir = Path(data_dir)
    output_root_dir = Path(output_dir)

    # Load from data dir
    for filename in ["train", "validation", "test"]:
        filepath = root_dir / (filename + ".csv")
        logger.info(f"Loading tokenised text from {filepath}")
        datasets[filename] = pd.read_csv(filepath)

    # Load embedding model
    with cs.spinner():
        if embed_type == "w2v":

            if pretrained_embed_dir is None:
                logger.info(
                    """Dowloading a gloVe vectors trained on google news with 300 dims as default since no pretrained embed dir was specified"""
                )
                embed_model = api.load('word2vec-google-news-300')

            else:
                embed_model = KeyedVectors.load_word2vec_format(
                    pretrained_embed_dir,
                    binary=False
                )

        elif embed_type == "ft":

            if pretrained_embed_dir is None:
                logger.info(
                    """Dowloading a Fast Text model trained on Wiki News with 300 dims as default since no pretrained embed dir was specified"""
                )
                embed_model = api.load("fasttext-wiki-news-subwords-300")

            else:
                embed_model = FastText.load(pretrained_embed_dir)

        else:
            raise ValueError("Please specify a valid embedding type - w2v or ft")

    # Vectorise Embeddings
    logger.info(f"Vectorising word embeddings using {vectorisation_mode}")

    if vectorisation_mode == "mean":
        vec_embed_model = MeanEmbeddingVectorizer(embed_model,
                                                  model_type=embed_type)
    else:
        vec_embed_model = TfidfEmbeddingVectorizer(embed_model,
                                                   model_type=embed_type)

    # Convert text into embeddings
    for dataset_name, dataset in datasets.items():
        logger.info(
            f"""Converting {dataset_name} dataset
            into vectorised embeddings"""
        )

        dataset["embeds"] = list(vec_embed_model.fit_transform(dataset))
        datasets[dataset_name] = dataset
        dataset.to_csv(output_root_dir / (dataset_name + ".csv"))

    logger.info(
        f"""Written train, val and test datasets to {output_dir}
        with embeddings in the 'embed' column"""
    )

    return datasets
