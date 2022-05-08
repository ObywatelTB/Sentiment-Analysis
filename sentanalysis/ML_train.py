import os
import time
import numpy as np
from absl import logging
import matplotlib.pyplot as plt
from rich import print as rprint
from tqdm import tqdm   #progress bar
from itertools import compress  #do indeksów branych boolami
from typing import Tuple, List, Dict, Any
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

from sentanalysis.ML_prepare_dataset import get_encoded_kaggle_tweets_datasets, \
                                            get_encoded_selected_tweets


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def perform_machine_learning(model_parameters: Dict[str, Any], dirs: Dict[str, str]
                            ) -> None:
    """
    Set the model architecture, train the model and test corectness.

    Args:
        model_parameters (dict[str, Any]) : Parameters concerning model 
        structure, training parameters.
        dirs (dict[str, str]) : A set of needed directory paths.

    Returns:
        None

    Raises:
    """
    train_ds, test_ds_kaggle = get_encoded_kaggle_tweets_datasets(dirs)
    test_ds_selected = get_encoded_selected_tweets(dirs)

    logging.set_verbosity(logging.ERROR) # Reduce logging output 
    model  = encoded_phrase_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = train(model_parameters, dirs, model, train_ds) 
    visualize(history)

    test_ds = {'Kaggle':test_ds_kaggle, 'Our tweets':test_ds_selected} 
    test_correctness(model, test_ds)


def predict(model_parameters: Dict[str, Any], dirs: Dict[str, str], 
            dirname: str='1652012944') -> None:
    """
    Load the model and test its predictions.

    Args:
        model_parameters (dict[str, Any]) : Parameters concerning model structure.
        dirs (dirs[str, str]) : A set of needed directory paths.

    Returns:
        None

    Raises:
    """
    model_name = model_parameters.get('model_name', '')

    model  = encoded_phrase_model()
    checkpoint_path = os.path.join(dirs['machine_learning'], model_name,
                                    'models', dirname, 'cp.ckpt')
    model.load_weights(checkpoint_path)

    # _, test_ds_kaggle = get_encoded_kaggle_tweets_datasets(dirs)
    test_ds_selected = get_encoded_selected_tweets(dirs)

    test_ds = {'Our tweets': test_ds_selected} 
    test_correctness(model, test_ds)


def encoded_phrase_model() -> tf.keras.Model:
    """
    Define model structure. It is a model handling float ndarrays as an 
    input. So that it can compute the phrases encoded with Google USE. 
    The target is a boolean label.
    """
    rprint('[italic red] Defining the model... [/italic red]')

    model = Sequential()
    model.add(Input(shape=(512,),dtype='float32'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    return model


def train(model_parameters: Dict[str, Any], dirs: Dict[str, str], model: tf.keras.Model, 
        train_ds: np.ndarray,  limit: int = 50000) -> tf.keras.callbacks.History:
    """Train model with the fit() method.""" 
    rprint('[italic red] Training the model... [/italic red]')
    model_name = model_parameters.get('model_name', '')
    epochs_amount = model_parameters.get('epoch', '')
    X = train_ds[0][:limit]
    Y = train_ds[1][:limit]

    checkpoint_path = os.path.join(dirs['machine_learning'], model_name, 
                    'models', str(int(time.time())), 'cp.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    history = model.fit(X, Y,
                        epochs = epochs_amount,
                        validation_split=0.1, #validation_data = val_ds,
                        # steps_per_epoch=steps_per_epoch,#10000,
                        callbacks=[cp_callback])  
    return history


def test_correctness(trained_model: tf.keras.Model, test_datasets: Dict[str, Tuple],
                    test_outliers: bool = False) -> None:
    """
    Using model.predict to compare results with the targets.
    
    Args:
        test_datasets (dict[str,tuple]) : The key defines a dataset's name,
        the values is a tuple of 2 ndarrays - model's inputs and targets.
    """
    for ds_name in test_datasets:
        dataset = test_datasets[ds_name]
        X = dataset[0]
        Y = dataset[1]
        predictions = trained_model.predict(X)
        predictions_rounded = list(map(round, [x[0] for x  in predictions]))
        accu = accuracy_score(predictions_rounded, Y)
        print('Accuracy of {}: {}%'.format(ds_name, accu*100))


def visualize(history: tf.keras.callbacks.History) -> None:
    """Plot the training history - training accuracy and val. accuracy."""
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy') 
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()