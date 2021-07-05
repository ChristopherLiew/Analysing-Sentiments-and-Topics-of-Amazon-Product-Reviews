"""
Dataset utilities
"""
import os
from typing import Dict, Optional, Union
import pandas as pd
from tensorflow import data
from tensorflow.keras import utils
from pathlib import Path


def create_tf_ds(X: pd.DataFrame, y: Optional[pd.DataFrame] = None,
                 shuffle: bool = True) -> data.Dataset:
    if y:
        # convert labels into one_hot_encoded labels
        y = utils.to_categorical(y)
        ds = data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            # Buffer Size = Size of DS for perfect shuffling
            return ds.shuffle(len(ds))
        return ds
    else:
        ds = data.Dataset.from_tensor_slices(X)
        if shuffle:
            return ds.shuffle((len(ds)))
        return ds


def process_dataset_labels(df: pd.DataFrame, 
                               label_col: str,
                               mapping: Dict[str, str]
                               ) -> pd.DataFrame:
    """
    Converts sentiment labels into a suitable 0 indexed label for model training.

    Args:
        df (pd.DataFrame): Training, validation or testing data.
        label_col (str): Name of column containing target labels.

    Returns:
        pd.DataFrame: Processed training, validation or testing data.
    """
    df[label_col] = df[label_col].map(mapping)
    return df


def create_model_dev_dir(data_filepaths: Dict[str, Union[str, pd.DataFrame]],
                         output_filepath: str
                         ) -> str:
    """
    Creates a data directory suitable for model training. Data filepaths
    should be labelled with the following keys:
    
    > train
    > validation
    > testing

    Args:
        data_fps (Dict[str, str]): Dictionary mapping train, val, test datasets to
        their respective filepaths.
        output_filepath (str): Output filepath to save directory to.
    """
    data_dir = Path(output_filepath)
    train_dir = Path / 'train'
    val_dir = Path / 'validation'
    test_dir = Path / 'test'
    
    # Create parent directory
    os.mkdir(data_dir)
    
    # Read and process data
    if isinstance(next(iter(set(v for v in data_filepaths))), str):
        train_df = pd.read_csv(data_filepaths.get('train'))
        val_df = pd.read_csv(data_filepaths.get('validation'))
        test_df = pd.read_csv(data_filepaths.get('test'))
    else:
        train_df, val_df, test_df = (
            data_filepaths.get('train'),
            data_filepaths.get('validation'),
            data_filepaths.get('test')
        )
    
    # Save and return output path
    train_df.to_csv(train_dir / 'train.csv')
    val_df.to_csv(val_dir / 'validation.csv')
    test_df.to_csv(test_dir / 'test.csv')
    
    return data_dir
