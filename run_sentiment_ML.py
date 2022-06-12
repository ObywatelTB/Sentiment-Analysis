import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from rich import print as rprint
from typing import List, Dict, Any


def encoded_phrase_model() -> tf.keras.Model:
    """
    Define model structure. It is a model handling float ndarrays as an 
    input. So that it can compute the phrases encoded with Google USE. 
    The target is a boolean label.
    """
    rprint('Defining the model...')

    model = Sequential()
    model.add(Input(shape=(512,),dtype='float32'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    return model


def compare_ML_and_NLTK(sentiment_ML_parameters: Dict[str, Any], dirs: Dict[str, str],
                        model: tf.keras.Model) -> None:
    """Load the selected, rated tweets dataset. Make the ML model prediction.
    Then execute the evaluation with nltk"""
    import sentanalysis.ML_prepare_dataset as ml_pd
    import sentanalysis.ML_train as ml_t
    from sourcing.CSVService import CSVService
    import sentanalysis.nltk_utils as nltk_u

    selected_ds = ml_pd.load_encoded_tweets(dirs, 'selection')
    selectedY = selected_ds[1]
    predictionsML = ml_t.predict(sentiment_ML_parameters, dirs, selected_ds, model)
    ML_to_test = (predictionsML, selectedY)

    csv_serv = CSVService(dirs['selected_tweets'])
    tweets = csv_serv.load_selected_tweets()
    tweets_X = tweets['tweet'].values
    tweets_Y = tweets['sentiment'].values
    scores = nltk_u.evaluate_opinions(tweets_X, normalize=True)
    NLTK_to_test = (scores, tweets_Y)

    ds = {'ML, selected tweets': ML_to_test,
        'NLTK, selected tweets': NLTK_to_test}
    nltk_u.test_correctness(ds, treshold_val = 0.1)


if __name__ == "__main__":
    import sentanalysis.ML_train as ml_t
    
    dir0 = os.path.dirname(__file__)
    dirt = os.path.join(dir0, 'twitter_scraped')
    dirs = {
        'machine_learning': os.path.join(dir0, 'machine_learning'),
        'kaggle_tweets': os.path.join(dir0, 'resources', 
                                'kaggle_1_6mln_tweets.csv'),
        'opinions_db': os.path.join(dirt,'tweetsIX_V.db'),
        'selected_tweets': os.path.join(dir0, 'resources', 'new_selection_xrp81k.csv'),
        'selected_tweets_encoded': os.path.join(dir0,'machine_learning',
                                'opinions_encoded','selected_tweets_encoded'),
        'kaggle_encoded': os.path.join(dir0, 'machine_learning','opinions_encoded',
                                    'kaggle_encoded'),
        'fil_opinions_encoded': os.path.join(dir0,'machine_learning','opinions_encoded',
                                        'filtered_opinions_encoded')
    }

    sentiment_ML_parameters = {
        'model_name': 'Sentiment_ML',
        'dataset_encoding': '',
        'batch_size': 32,   # used during encoding to split the ds (?)
        'epoch': 3
    }

    model = encoded_phrase_model()

    program = 1 # choose between 1 and 0
    if program: # 1
        ml_t.perform_machine_learning(sentiment_ML_parameters, dirs, model)
    else:       # 0
        compare_ML_and_NLTK(sentiment_ML_parameters, dirs, model)
