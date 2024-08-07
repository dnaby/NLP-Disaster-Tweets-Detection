import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, List

def get_dataset(raw: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads and returns the train and test datasets based on the 'raw' parameter.

    Parameters:
    raw (bool): If True, reads the raw datasets. Otherwise, reads the processed datasets.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test datasets.
    """
    return (
        pd.read_csv(f"../data/{'raw' if raw else 'processed'}/train.csv"),
        pd.read_csv(f"../data/{'raw' if raw else 'processed'}/test.csv")
    )

def create_corpus(disaster: bool, raw: bool) -> list:
    """
    Creates a corpus of words from the tweets.

    Parameters:
    disaster (bool): If True, creates corpus for disaster tweets. Otherwise, for non-disaster tweets.
    raw (bool): If True, uses the raw dataset. Otherwise, uses the processed dataset.

    Returns:
    list: A list of words from the tweets.
    """
    # Load the dataset
    train, _ = get_dataset(raw)
    
    corpus = []
    
    # Filter out NaN values and create corpus
    for x in train[train['target'] == int(disaster)]['text'].dropna().str.split():
        for i in x:
            corpus.append(i)
    
    return corpus

def get_top_tweet_bigrams(corpus: List[str], top: int = 10) -> List[Tuple[str, int]]:
    """
    Extracts and returns the top 'top' bigrams from the corpus.

    Parameters:
    corpus (List[str]): The list of words to extract bigrams from.
    top (int): The number of top bigrams to return.

    Returns:
    List[Tuple[str, int]]: A list of tuples, each containing a bigram and its frequency.
    """
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top]

def get_top_tweet_trigrams(corpus: List[str], top: int = 10) -> List[Tuple[str, int]]:
    """
    Extracts and returns the top 'top' trigrams from the corpus.

    Parameters:
    corpus (List[str]): The list of words to extract trigrams from.
    top (int): The number of top trigrams to return.

    Returns:
    List[Tuple[str, int]]: A list of tuples, each containing a trigram and its frequency.
    """
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top]

def get_top_keywords(dataset: pd.DataFrame, top: int = 10) -> pd.Series:
    """
    Extracts and returns the top 'top' keywords from the 'keyword' column of the dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to extract keywords from.
    top (int): The number of top keywords to return.

    Returns:
    pd.Series: A series containing the top keywords and their frequencies.
    """
    keywords = dataset['keyword'].dropna()
    keyword_counts = keywords.value_counts().head(top)
    return keyword_counts

def get_top_locations(dataset: pd.DataFrame, top: int = 10) -> pd.Series:
    """
    Extracts and returns the top 'top' locations from the 'location' column of the dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to extract locations from.
    top (int): The number of top locations to return.

    Returns:
    pd.Series: A series containing the top locations and their frequencies.
    """
    locations = dataset['location'].dropna()
    location_counts = locations.value_counts().head(top)
    return location_counts
