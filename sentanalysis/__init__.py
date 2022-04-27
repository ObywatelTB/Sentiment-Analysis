"""
Sentanalysis
===

A package with various tools for the sentiment analysis.

Gives access to functions:
- analyse_opinions(MCparams: Dict[str, Any], dp: Dict[str, str],
        dbtable: TableInterface, last_percents: float = 100.0) -> None
- find_common_words_in_tweets(all_tweets: pd.DataFrame, key_words_nr: int, 
                                progress_bar: tqdm) -> List[Tuple[Any, int]]
- aggregate_filters(word_filters: pd.DataFrame, counted_ngrams) -> pd.DataFrame
- save_filters(word_filters: pd.DataFrame, filters_path: str) -> None

And machine learning tools:
- predict(dirs: Dict[str, str]) -> None
- vectorize_and_save_ds(dirs: Dict[str, str], which_df: str = 'kaggleNOT') -> None
- create_model(dirs: Dict[str, str]) -> None
"""
from sentanalysis.sentiment_analysis import analyse_opinions
from sentanalysis.nltk_utils import find_common_words_in_tweets
from sentanalysis.nltk_utils import aggregate_filters
from sentanalysis.nltk_utils import save_filters
from sentanalysis.train_sentiment_ML import predict, vectorize_and_save_ds, create_model