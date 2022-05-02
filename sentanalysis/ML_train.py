import os
import numpy as np
from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
from rich import print as rprint
from tqdm import tqdm   #progress bar
import string
import re
from itertools import compress  #do indeksów branych boolami
from typing import Tuple, List, Dict, Any
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

from sentanalysis.ML_prepare_dataset import load_datasets, switch_to_numpy_tuple, \
                                            string_to_ndarray


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def perform_machine_learning(model_parameters: Dict[str, Any], dirs: Dict[str, str]
                            ) -> None:
    """
    Set the model architecture, train and test corectness.

    Args:
        model_parameters (dict[str, Any]) : Parameters concerning model 
        structure, training parameters.
        dirs (dict[str, str]) : A set of needed directory paths.

    Returns:
        None

    Raises:
    """
    train_ds, val_ds, test_ds_kaggle, test_ds_ours = load_datasets(dirs)

    logging.set_verbosity(logging.ERROR) # Reduce logging output 
    model  = encoded_phrase_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = train(model_parameters, dirs, model, train_ds, val_ds) 
    visualize(history)

    test_ds = {'Kaggle':test_ds_kaggle, 'Our tweets':test_ds_ours} 
    test_correctness(model, test_ds)

    print('Executed machine learning.')


def test_correctness(trained_model: tf.keras.Model, test_datasets: Dict[str, list]
                    ) -> None:
    """Using model.predict to compare resulsts with the targets."""
    for ds_name in test_datasets:
        dataset = test_datasets[ds_name]
        X = dataset[0]
        Y = dataset[1]
        pre = trained_model.predict(X)
        predictions = list(map(round, [x[0] for x  in pre]))
        print(f'Precision, {ds_name}:', accuracy_score(predictions, Y)*100, '%')


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
        train_ds: np.ndarray,  val_ds: np.ndarray) -> tf.keras.callbacks.History:
    """Train model with the fit() method.""" 
    rprint('[italic red] Training the model... [/italic red]')
    model_name = model_parameters.get('model_name', '')
    epochs_amount = model_parameters.get('epoch', '')

    X = train_ds[0]
    Y = train_ds[1]
    
    checkpoint_path = \
        os.path.join(dirs['machine_learning'], model_name, 'models', 'cp-.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    history = model.fit(X, Y, #train_ds,    # allows to create batches
                        epochs = epochs_amount,
                        validation_data = val_ds,
                        # validation_split=0.1,#validation_data = val_ds,
                        # steps_per_epoch=steps_per_epoch,#10000,
                        callbacks=[cp_callback])  
    return history


def predict(model_parameters: Dict[str, Any], dirs: Dict[str, str]) -> None:
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
    model.compile(loss='binary_crossentropy', 
            optimizer='adam',
            metrics=['acc'])
    checkpoint_path = os.path.join(dirs['models_checkpoints'], model_name, 'cp2.ckpt')
    model.load_weights(checkpoint_path)

    filenames = ['0.csv'] #['0.csv','2.csv', '5.csv','7.csv']
    for const in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]: #fn in filenames
        filepath = dirs['our_rated_tweets_encoded']
        filepath = os.path.join(os.path.dirname(filepath), '0.csv') #fn
        df_ours = pd.read_csv(filepath, header=None,usecols=[1,2],
                                names=['tweet','sentiment'])
        df_ours['tweet'] = df_ours['tweet'].apply(string_to_ndarray)
        ours_tuple = switch_to_numpy_tuple(df_ours)

        # loss, acc = model.evaluate(ours_ds[0], ours_ds[1], verbose=2)
        # print(f"Restored model, tested on {fn} dataset. Accuracy: {100 * acc:5.2f}%", loss)

        # Leaves outliers only: (0, treshold) U (1-treshold, 1) 
        cut = lambda a : a < const or a > 1-const

        pre = model.predict(ours_tuple[0])
        pre=([x[0] for x  in pre])
        fil = list(map(cut, pre))
        predictions = list(compress(pre,fil))
        predictions = list(map(round, predictions))
        check = list(compress(ours_tuple[1],fil))
        print(f'Len {len(predictions)}. Precision for tres={const}:', 
                accuracy_score(predictions, check)*100,'%')
        
    print('Done with predicting.')


def visualize(history: tf.keras.callbacks.History) -> None:
    """Plot the training history - training accuracy and val. accuracy."""
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy') 
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()