import nltk         #nltk.help.upenn_tagset() - list of parts of language 
from nltk import word_tokenize          #Participle function
from nltk.corpus import stopwords       #Stopword list, such as a, the and other unimportant words
from nltk.corpus import wordnet as wn   #WordNet - gives us synsets, groupings of synonyms
from nltk.corpus import sentiwordnet as swn #Get word emotion score
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize  #lemmatization - finds simplest form of a word
lemmatizer = WordNetLemmatizer()
from nltk.wsd import lesk
from nltk.sentiment.vader import SentimentIntensityAnalyzer #vader - competition to sentiwordnet
vader = SentimentIntensityAnalyzer()

import string           #To import punctuation, such as!"#$%& 
import sqlite3
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import os

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Any, Tuple
console = Console()


#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 99
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa     79
#-----------------------------------------------------------------------            72 comments


def evaluate_sentiment(opinions: pd.DataFrame, analysis_parameters: Dict[str, Any]
                        ) -> pd.Series:
    """
    Evaluate sentiment scores for given opinions using the nltk library.

    Args:
        opinions (DataFrame) : A DF with opinions on some specific topic.
        analysis_parameters (dict[str, Any]) : Parameters affecting the 
        sentiment evaluation (can be part of Monte Carlo parameters).
    Returns:
        scores (Series) : Evaluated sentiment scores of given opinions.
    Raises:
    """
    treshold = analysis_parameters.get('sentiment_tres', 0.25)
    is_score_binary = analysis_parameters.get('is_score_binary', False)
    keywords = analysis_parameters.get('keywords', [])      # rather not used

    scores = pd.Series(index=opinions.index, dtype=np.float64)
    scores.loc[:] = evaluate_opinions(opinions.content.values, 
                                            treshold, is_score_binary)
    scores = scores[scores!=0]
    return scores


def evaluate_opinions(opinions: np.ndarray, treshold: float, 
                        is_score_binary: bool) -> np.ndarray:
    """
    Evaluate sentiment of many opinions using the nltk library.

    Args:
        opinions (ndarray) : List of tweets, with date index.
        treshold (float) : T. below which the abs(score) will equal to 0.
        is_score_binary (bool) : If True the score will be within {-1,0,1}.
    Returns:
        scores (ndarray): List of scores, with date index.
    Raises:
    """
    scores = np.array([vader.polarity_scores(t)['compound'] for t in opinions])

    are_over_treshold = abs(scores) > treshold
    scores = np.where(are_over_treshold, scores, np.zeros(len(opinions)))
    if is_score_binary:
        scores = np.sign(scores)
    return scores


def find_common_words_in_tweets(all_tweets: pd.DataFrame, key_words_nr: int, 
                                progress_bar: tqdm) -> List[Tuple[Any, int]]:
    """
    Find the bigrams/trigrams present in tweets the most often, along
    with the number of repetitions.

    Args:

    Returns:
        counted_ngrams (list[tuple[Any,int]]) : List of found ngrams and
        their amounts, sorted by their amounts.
    Raises:
    """
    ngrams = []
    CUT_TRES = 5
    for index, row in all_tweets.iterrows(): 
        text =  row['content']
        # here space for code deleting all the words like 'the', 'in', etc. 
        if key_words_nr == 2: 
            # finder below contains all the word combinations per tweet
            finder = nltk.BigramCollocationFinder.from_words(text.split(), 
                                                            window_size = 3)
        else:
            finder = nltk.TrigramCollocationFinder.from_words(text.split(), 
                                                            window_size = 3)
        ngrams = [*ngrams, *[k for k,v in finder.ngram_fd.items()]]
        progress_bar.update(1)
    # below sorting based on the position amount
    counted_ngrams = sorted(Counter(ngrams).items(), key=lambda item: item[1])
    # below we filter out the rare combinations (rarely present in tweets)
    counted_ngrams = list([cn for cn in counted_ngrams if cn[1]>CUT_TRES])
    return counted_ngrams


def aggregate_filters(word_filters: pd.DataFrame, counted_ngrams) -> pd.DataFrame:
    """

    Args:
        word_filters (DataFrame) :
        counted_ngrams () :
    Returns:
        filters_final (DataFrame) :
    Raises:
    """
    counted_ngrams.reverse()
    new_filters = pd.DataFrame(counted_ngrams, columns=['words','count'])
    filters_concat = pd.concat([word_filters, new_filters])
    filters_grouped = filters_concat.groupby(['words']).agg(['mean','count']
                                )['count'].sort_values(by=['count'], ascending=False)
    
    filters_final = filters_grouped['mean'] * filters_grouped['count']
    filters_final.sort_values(ascending=False, inplace=True)
    filters_final = pd.DataFrame(filters_final, columns=['count']).reset_index() 
    if len(filters_final) > 410:
        filters_final = filters_final.loc[0:400]  # leaves the last 400 filters
    return filters_final


def save_filters(word_filters: pd.DataFrame, filters_path: str) -> None:
    """Save the filters to a file."""
    if not os.path.exists(os.path.dirname(filters_path)): 
        os.mkdir(os.path.dirname(filters_path))
    word_filters.to_csv(filters_path)


def tokenize(post):
    """Get tokens from a phrase."""
    stop = stopwords.words("english") + list(string.punctuation)
    tokens = word_tokenize(str(post).lower())
    tokens_lem = list(map(lemmatizer.lemmatize, tokens))
    tags = nltk.pos_tag([t for t in tokens_lem if t not in stop]) 
    return tags


def to_wn(tupla):
    """Creating wordnet tags"""
    [word, tag] = tupla
    di = {('N','U'):'n', 
            'V':'v', 
            'J':'a', 
            ('R','W'):'r'}
    for key in di:
        if tag.startswith(key):
            return (word, di[key])
    return (word, '')


def evaluate_phrase(phrase: str, tokenized):
    """Evaluate phrase's sentiment, tokenizing with synsets."""
    score = 0
    s=[]
    for tupla in tokenized:
        [word, tag] = tupla   
        synsets = wn.synsets(word, pos=tag)
        if not synsets:
            return 0 
        synset = lesk(phrase, word, tag)
        if synset:
            swn_synset = swn.senti_synset(synset.name())
            score += swn_synset.pos_score() - swn_synset.neg_score()
            s.append((word, swn_synset.pos_score(), swn_synset.neg_score()))
    return score


def print_evaluation(df: pd.DataFrame, rows: List[int]):
    """Diagnose evaluation quickly"""
    for r in rows:
        print('\nScore:', df['values'].loc[r], 'post: ', df['posts'].loc[r])


def word_in_text(text: str, forbidden_words: List[str]) -> bool:
    """Simple checking if any word is in a text, for filtering."""
    if any(fw in text for fw in forbidden_words):
        return False
    return True


# Poniższe do paczki diagnostycznej/bazodanowej
def write_out_tweets_with_words(word_filters, db_name, key_words_nr, shown_bigrams_nr, shown_tweets_nr):
    inx = 0
    for index, row in word_filters.iterrows():
        key = row['words']
        value = row['count']
        conn = sqlite3.connect(db_name)
        if key_words_nr == 2:
            tweets_db = pd.read_sql("SELECT* FROM tweets_table WHERE content LIKE '%"+key[0]+"%' AND content LIKE '%"+key[1]+
            "%'  LIMIT "+str(shown_tweets_nr)+"; ", conn)
        else:
            tweets_db = pd.read_sql("SELECT* FROM tweets_table WHERE content LIKE '%"+key[0]+"%' AND content LIKE '%"+key[1]+
            "%' AND content LIKE '%"+key[2]+"%'  LIMIT " + str(shown_tweets_nr)+"; ", conn)
        conn.close()

        rprint(f'[italic red]Key: {key}, count:{value}[/italic red] \n')
        for i in range(0, shown_tweets_nr):
            print(tweets_db['content'][i])
            print('\n')
        print('================================================')
        inx += 1
        if inx > shown_bigrams_nr:
            return


#===printing out
def write_out_scores(posts, tab_amount, treshold):
    tables_printed = 0
    inx = 0
    while tables_printed < tab_amount:
        table = Table(show_lines=True)#title=)
        #console.print(f'[bold]post nr {inx}[/bold]:', style="white")
        post = posts['content'].values[inx]
        table.add_column("Valuation", style="cyan", no_wrap=True)
        table.add_column(f"post nr {inx+1}")#, style="green")
        for phrase in nltk.sent_tokenize(post):
            score = vader.polarity_scores(phrase)['compound']
            if score > treshold:
                score = 1
            elif score < -treshold:
                score = -1
            else:
                score = 0 
            if score:
                table.add_row(str(score), phrase)
        if table.row_count > 0:
            console.print(table)
            tables_printed += 1
        inx += 1
