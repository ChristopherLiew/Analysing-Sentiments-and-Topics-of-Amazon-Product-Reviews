import typer
from typing import (
    Optional,
    Dict,
)
import pandas as pd
from pathlib import Path
from gensim import downloader
from gensim.models import KeyedVectors
from utils.embedding_vectoriser import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer
from utils.logging import logger

# Instatiate Typer App
app = typer.Typer()


# TBD:
# Run a full test
# Add in tokenization and preprocessing (KIV)


@app.command()
def preprocess(
    data_dir: str,
    output_dir: str,
    embed_type: str = "w2v",
    pretrained_embed_dir: Optional[str] = None,
    vectorisation_mode: str = "mean",
) -> Dict[str, pd.DataFrame]:

    datasets = dict()
    root_dir = Path(data_dir)
    output_root_dir = Path(output_dir)

    # Load from data dir
    for filename in ["train", "validation", "test"]:
        filepath = root_dir / (filename + ".csv")
        datasets[filename] = pd.read_csv(filepath)

    # Load embedding model
    if embed_type == "w2v":

        assert (
            pretrained_embed_dir is not None
        ), """Please download GoogleNews Vectors or
            any other pretrained embeedgins and provide
            its dir path"""

        embed_model = KeyedVectors.load_word2vec_format(
            pretrained_embed_dir, binary=True
        )
    else:

        if pretrained_embed_dir is None:
            logger.info(
                """Dowloading a Fast Text model trained
                         on Wiki News with 300 dims"""
            )
            embed_model = downloader.load("fasttext-wiki-news-subwords-300")

        else:
            embed_model = KeyedVectors.load(pretrained_embed_dir)

    # Vectorise Embeddings
    logger.info(f"Vectorising word embeddings using {vectorisation_mode}")

    if vectorisation_mode == "mean":
        vec_embed_model = MeanEmbeddingVectorizer(embed_model)
    else:
        vec_embed_model = TfidfEmbeddingVectorizer(embed_model)

    # Convert text into embeddings
    for dataset_name, dataset in datasets.items():
        logger.info(
            f"""Converting {dataset_name} dataset
                     into vectorised embeddings"""
        )

        dataset["embeds"] = vec_embed_model.fit_transform(dataset)
        datasets[dataset_name] = dataset

        dataset.to_csv(output_root_dir / (dataset_name + ".csv"))

    logger.info(
        f"""Written train, val and test datasets to {output_dir}
                 with embeddings in the 'embed' column"""
    )

    return datasets
