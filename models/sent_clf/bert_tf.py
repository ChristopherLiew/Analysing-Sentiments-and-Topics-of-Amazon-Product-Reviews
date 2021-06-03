import os
import time
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report
from utils import create_tf_ds, get_log_dir

## TBD ##
# Train on 5 Epochs
# Try Monte Carlo Dropout?

## Configs ##
pd.options.display.max_columns = 20
tf.get_logger().setLevel('ERROR')
K = keras.backend
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### Load our preprocessed text data ###
def process_and_concat_text(X):
    list_data = [literal_eval(review) for review in X]
    concat_data = [' '.join(review) for review in list_data]
    return concat_data

## Train data ##
raw_train_ds = pd.read_csv('data/processed_data/proc_train.csv')
raw_train_text, raw_train_labels = raw_train_ds['text_processed'].to_list(), \
    raw_train_ds['sentiment'].map({-1: 0, 0: 1, 1: 2}).to_list()
raw_train_text = process_and_concat_text(raw_train_text)

## Test data ##
raw_test_ds = pd.read_csv('data/processed_data/proc_test.csv')
raw_test_text, raw_test_labels = raw_test_ds['text_processed'].to_list(), \
    raw_test_ds['sentiment'].map({-1: 0, 0: 1, 1: 2}).to_list()
raw_test_text = process_and_concat_text(raw_test_text)

## Create Tensorflow Dataset ##
# Create TF train ds
amz_train_ds = create_tf_ds(raw_train_text, y=raw_train_labels)
amz_test_ds = create_tf_ds(raw_test_text, y=raw_test_labels)

## Create Validation Set ##
VAL_SPLIT = 0.2
val_set_size = int((len(amz_train_ds) + len(amz_test_ds)) * VAL_SPLIT)  # 20% of entire DS
amz_val_ds = amz_train_ds.take(val_set_size)
amz_train_ds = amz_train_ds.skip(val_set_size)

## Cache and Create Padded Batches ##
# We want to pad the length of our sequences at the batch level (empty = pad to max len of that batch)
BATCH_SIZE = 32
amz_train_ds = amz_train_ds.padded_batch(BATCH_SIZE, padded_shapes=((), (3, )))
amz_val_ds = amz_val_ds.padded_batch(BATCH_SIZE, padded_shapes=((), (3, )))
amz_test_ds = amz_test_ds.padded_batch(BATCH_SIZE, padded_shapes=((), (3, )))
next(iter(amz_val_ds))

## Loading BERT model and pre-processing model ##
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2"

## Fine Tuned BERT model ##
def build_bert_classifier(preprocessing_model, bert_model):
    text_input = keras.layers.Input(shape=(), 
                                    dtype=tf.string, 
                                    name='text_input')

    bert_preprocessing_layer = hub.KerasLayer(preprocessing_model, 
                                                name='bert_preprocessing')  # Truncate input to 128

    encoder_inputs = bert_preprocessing_layer(text_input)
    
    bert_encoder = hub.KerasLayer(bert_model, 
                                trainable=True, 
                                name='BERT_encoder')  # Fine tune BERT params

    outputs = bert_encoder(encoder_inputs)
    
    # Embedding for the entire review dataset
    pooled_output = outputs['pooled_output']
    net = keras.layers.Dropout(0.2)(pooled_output)
    net = keras.layers.Dense(3, activation='softmax',
                             name='softmax_activation')(net)

    return keras.Model(inputs=text_input, outputs=net)


## Build model (! make sure to import tensorflow text) ##
classifier_model = build_bert_classifier(PREPROCESS_MODEL, BERT_MODEL)

## View model ##
IMG_NAME = 'my_bert_model.png'
SAVE_MODEL_IMG_PATH = os.path.join('models/saved_models', IMG_NAME)

keras.utils.plot_model(classifier_model,
                       SAVE_MODEL_IMG_PATH,
                       show_dtype=True,
                       show_layer_names=True,
                       show_shapes=True)

## Train model ##
# Loss function (Cat CrossEntropy since [n_obs, n_class]; Use SparseCategoricalEntropy if it is 1D)
# Softmax applied, thus normalised.
loss = keras.losses.CategoricalCrossentropy(from_logits=False)
metric = tf.keras.metrics.CategoricalAccuracy()

# Optimizer (Copy BERT pre-training process)
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(amz_train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1 * num_train_steps)  # 10% for warm up

init_lr = 2e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(loss=loss,
                         optimizer=optimizer,
                         metrics=metric)

## Fit BERT model ##
# Callbacks
RUN_LOG_DIR = get_log_dir()
MODEL_NAME = 'my_bert_model.h5'

tensorboard_cb = TensorBoard(RUN_LOG_DIR)
model_checkpoint_cb = ModelCheckpoint('models/saved_models/%s' % MODEL_NAME.split('.')[0], save_best_only=True)
early_stopping_cb = EarlyStopping(patience=20)

# Train
history = classifier_model.fit(x=amz_train_ds,
                               validation_data=amz_val_ds,
                               epochs=epochs,
                               callbacks=[early_stopping_cb, model_checkpoint_cb, tensorboard_cb])

## Evaluate Model
loss, accuracy = classifier_model.evaluate(amz_test_ds)
print('BERT model\n loss: {loss}\n accuracy: {accuracy}')

# Classification Results
y_pred = np.argmax(classifier_model.predict(amz_test_ds), axis=-1)
classification_rep = pd.DataFrame(classification_report(
    y_pred=y_pred, y_true=raw_test_labels, output_dict=True))

## Plot training and validation learning curves
# bash: $ tensorboard --logdir=./Transformer\ Models/my_logs --port=6006

# Loss & Accuracy vs. Epoch
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

## Reset
K.clear_session()
