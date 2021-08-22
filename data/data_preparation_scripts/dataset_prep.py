# Import libraries
import pandas as pd
import numpy as np
from preprocessing.text_preprocessing import preprocess_text
from utils.datasets import create_model_dev_dir, process_dataset_labels


# Load raw data
raw_df = pd.read_csv(
    "data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
)

# Get relevant rows
sel_cols = ["reviews.text", "reviews.rating"]
raw_df_sel = raw_df[sel_cols]
raw_size = len(raw_df_sel)
print(f"Size of dataset: {raw_size}")

train_size = int(0.9 * raw_size)
val_size = int(0.1 * train_size)

# Train - Test (90 - 10)
indices = np.random.permutation(raw_df_sel.shape[0])
train_idx, test_idx = indices[:train_size], indices[train_size:]
train_idx, val_idx = train_idx[val_size:], train_idx[:val_size]

# Raw splits
train_raw = raw_df_sel.iloc[train_idx, :]
val_raw = raw_df_sel.iloc[val_idx, :]
test_raw = raw_df_sel.iloc[test_idx, :]

# Add in col name remap ***
# reviews.text -> text
# reviews.rating -> rating

# Preprocess data for classical ml models
train_processed = preprocess_text(train_raw)
val_processed = preprocess_text(val_raw)
test_processed = preprocess_text(test_raw)

# Preprocess data for transformer models
train_processed_hf = preprocess_text(train_raw, tokenize=False)
val_processed_hf = preprocess_text(val_raw, tokenize=False)
test_processed_hf = preprocess_text(test_raw, tokenize=False)


# Convert product review ratings into sentiment labels
label_mapping = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# Preprocessed for classical ml
train_processed_mapped = process_dataset_labels(train_processed, label_mapping)
val_processed_mapped = process_dataset_labels(val_processed, label_mapping)
test_processed_mapped = process_dataset_labels(test_processed, label_mapping)

# Preprocessed for transformer models
train_processed_hf_mapped = process_dataset_labels(
    train_processed_hf, label_mapping
).dropna()
val_processed_hf_mapped = process_dataset_labels(
    val_processed_hf, label_mapping
).dropna()
test_processed_hf_mapped = process_dataset_labels(
    test_processed_hf, label_mapping
).dropna()

# Write data
# Raw split
raw_data_filepaths = {
    "train": train_raw.dropna(),
    "validation": val_raw.dropna(),
    "test": test_raw.dropna(),
}

create_model_dev_dir(raw_data_filepaths, "data/raw_split")

# Preprocessed for classical ml
raw_data_filepaths_ml = {
    "train": train_processed_mapped.dropna(),
    "validation": val_processed_mapped.dropna(),
    "test": test_processed_mapped.dropna(),
}

create_model_dev_dir(raw_data_filepaths_ml, "data/processed_ml")

# Preprocessed for transformer models
raw_data_filepaths_tf = {
    "train": train_processed_hf_mapped,
    "validation": val_processed_hf_mapped,
    "test": test_processed_hf_mapped,
}

create_model_dev_dir(raw_data_filepaths_tf, "data/processed_hf")
