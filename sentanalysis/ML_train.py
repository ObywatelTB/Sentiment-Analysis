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
from sklearn.model_selection import train_test_split

from sentanalysis.nltk_utils import test_correctness
from sentanalysis.ML_prepare_dataset import load_encoded_tweets


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def perform_machine_learning(model_parameters: Dict[str, Any], 
                            dirs: Dict[str, str], model: tf.keras.Model) -> None:
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
    X, Y = load_encoded_tweets(dirs, 'kaggle')
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, shuffle=True)
    train_ds = (Xtrain, Ytrain)
    test_ds_kaggle = (Xtest, Ytest)

    test_ds_selected = load_encoded_tweets(dirs, 'selection')

    logging.set_verbosity(logging.ERROR) # Reduce logging output 

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = train(model_parameters, dirs, model, train_ds) 
    visualize(history)

    kaggle_pre = model.predict(test_ds_kaggle[0]).T[0]
    selection_pre = model.predict(test_ds_selected[0]).T[0]

    test_ds = {'Kaggle': (kaggle_pre, test_ds_kaggle[1]), 
                'Selected tweets': (selection_pre, test_ds_selected[1])} 
    test_correctness(test_ds)


def predict(model_parameters: Dict[str, Any], dirs: Dict[str, str], 
            dataset_to_predict: Tuple[np.ndarray, Any],
            model: tf.keras.Model, cp_dirname: str='1652112389') -> None:
    """
    Load the model and test its predictions.

    Args:
        model_parameters (dict[str, Any]) : Parameters concerning model structure.
        dataset_to_predict tuple(ndarray, Any) : The first value of the tuple 
        matters, because we use it to perform the predictions.
        dirs (dirs[str, str]) : A set of needed directory paths.
        cp_dirname (str) : The chosen model checkpoint directory name.

    Returns:
        None

    Raises:
    """
    model_name = model_parameters.get('model_name', '')

    checkpoint_path = os.path.join(dirs['machine_learning'], model_name,
                                    'models', cp_dirname, 'cp.ckpt')
    model.load_weights(checkpoint_path).expect_partial()

    X = dataset_to_predict[0]
    predictions = model.predict(X).T[0]
    return predictions


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