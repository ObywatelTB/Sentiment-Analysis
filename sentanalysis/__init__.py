"""
Sentanalysis
===

A package with various tools for the sentiment analysis.

Gives access to functions:
- analyse_opinions(MCparams: Dict[str, Any], dp: Dict[str, str],
        dbtable: TableInterface, last_percents: float = 100.0) -> None
- evaluate_opinions(opinions: np.ndarray, treshold: float, 
                        is_score_binary: bool) -> np.ndarray
- find_common_words_in_tweets(all_tweets: pd.DataFrame, key_words_nr: int, 
                                progress_bar: tqdm) -> List[Tuple[Any, int]]
- aggregate_filters(word_filters: pd.DataFrame, counted_ngrams) -> pd.DataFrame
- save_filters(word_filters: pd.DataFrame, filters_path: str) -> None

And machine learning tools:
- vectorize_dataset(model_parameters: Dict[str, Any], dirs: Dict[str, str], 
                        dataset_name: str = 'selection') -> None
- perform_machine_learning(model_parameters: Dict[str, Any], dirs: Dict[str, str]
                            ) -> None
- predict(model_parameters: Dict[str, Any], dirs: Dict[str, str], 
            dirname: str='1652112389') -> None
"""
from sentanalysis.sentiment_analysis import analyse_opinions
from sentanalysis.nltk_utils import evaluate_opinions
from sentanalysis.nltk_utils import find_common_words_in_tweets, \
                                aggregate_filters, \
                                save_filters

from sentanalysis.ML_prepare_dataset import vectorize_dataset
from sentanalysis.ML_train import perform_machine_learning, predict