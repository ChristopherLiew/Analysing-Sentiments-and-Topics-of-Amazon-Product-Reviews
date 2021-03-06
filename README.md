# Sentiment Prediction and Topic Modelling using Amazon Product Reviews
![](https://img.shields.io/github/issues/ChristopherLiew/Analysing-sentiments-and-topics-of-amazon-reviews) ![](https://img.shields.io/github/languages/code-size/ChristopherLiew/Analysing-sentiments-and-topics-of-amazon-reviews) ![](https://img.shields.io/github/last-commit/ChristopherLiew/Analysing-sentiments-and-topics-of-amazon-reviews)

Predicting consumer product review sentiments using classical ml and language models (SOTA transformers) and understanding paintpoints using topic modelling.

## Set Up
1. Build Docker Poetry Image
    ```zsh
    docker build . -t amz-sent-analysis
    ```
2. Poetry
   ```zsh
   poetry install
   poetry shell
   ```

## Sentiment Classification
### Overview
Sentiment categories:
1. Positive = 1
2. Neutral = 0
3. Negative = -1

### Approach
Constructed preprocessing, training and inference pipelines with ```Weights & Biases``` logging (metrics & artifacts) using a simple CLI application.

#### Other Experiments and Areas to Look At
1. Data Augmentation
    * Synonyms
    * Back Translation using Google Translate (EN -> DE -> EN)

### Sentiment Classification CLI Application Overview 
* Preprocessing Pipelines
  1. Wrangling
     * Normalisation
     * Expanding of contractions
     * Removal of punctuation and accented characters
     * Lemmatization
     * Stop word removal
     * Tokenization (if necessary) 
  3. Convert to Embeddings & Vectorisation / Pooling (For Classical ML only)
  ```zsh
  $ auto-sent text2embed create-embeds <DATA_PATH> <OUTPUT_DIR>
  ```
  ```zsh
  # For full documentation
  $ auto-sent text2embed train --help
  ```
* Word Embedding Generation Pipelines
  * Train fast text word embeddings
  ```zsh
  $ auto-sent ft-embeds train <DATA_PATH> <OUTPUT_DIR>
  ```
  ```zsh
  # For full documentation
  $ auto-sent ft-embeds train --help
  ```
* Modelling Pipelines for Training & Inference
  * Hugging Face Transformer
    1. Train
    ```zsh
    $ auto-sent hf-clf train <MODEL_NAME> <DATA_DIR>
    ```
    2. Inference (If options are not triggered pulls the latest model and test dataset from W&B)
    ```zsh
    $ auto-sent hf-clf predict --model-name <MODEL_NAME> --inf-data <TEST_DATASET_PATH>
    ```
  
  * Random Forest
    1. Train
    ```zsh
    $ auto-sent rf-clf train <DATA_DIR>
    ```
    2. Inference (If options are not triggered pulls the latest model and test dataset from W&B)
    ```zsh
    $ auto-sent rf-clf predict --inf-data <TEST_DATASET_PATH>
    ```
    
  * SVC
    1. Train
    ```zsh
    $ auto-sent svc-clf train <DATA_DIR>
    ```
    2. Inference (If options are not triggered pulls the latest model and test dataset from W&B)
    ```zsh
    $ auto-sent svc-clf predict --inf-data <TEST_DATASET_PATH>
    ``` 

### Experimentation Tracking & Results
Models employed ranged from statistical learning or classical ML models to state of the art transformer models form huggingface. The former required more extensive preprocessing since they are not inherently language models. As such, they were used in combination with word embeddings (i.e. gloVe and fast_text).

To view the model experimentation results please visit the W&B repo at:
* Classical ML: https://wandb.ai/chrisliew/amz-sent-analysis-classical-ml
    
* Transformer Models: https://wandb.ai/chrisliew/amz-sent-analysis-deep-learning

## Topic Modelling (Completed - Results to be summarised below)
### Experimentation
#### Models
##### LDA
##### LSI
##### GSDMM

## Running Training / Inference Jobs in Sagemaker Notebook Instance
```zsh
# Install via curl

$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# CD into relevant directory where the pyproject file is
$ cd /home/ec2-user/SageMaker/Analysing-sentiments-and-topics-of-amazon-reviews

# Activate venv
$ source $HOME/.poetry/env 

# Set up
$ poetry install
$ poetry shell
```

