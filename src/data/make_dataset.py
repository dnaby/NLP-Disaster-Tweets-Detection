import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset(raw: bool = True) -> (pd.DataFrame, pd.DataFrame):
  """_summary_

  Args:
      raw (bool): set to true if you want to get raw data and to false if you want to get processed data

  Returns:
      (pd.DataFrame, pd.DataFrame): a tuple containing train dataframe and test dataframe
  """
  return (
    pd.read_csv(f"../data/{'raw' if raw else 'processed'}/train.csv"),
    pd.read_csv(f"../data/{'raw' if raw else 'processed'}/test.csv")
  )

def create_corpus(disaster: bool = True, raw: bool = True):
  train, _ = get_dataset(raw)
  corpus=[]
    
  for x in train[train['target']==int(disaster)]['text'].str.split():
    for i in x:
      corpus.append(i)
  return corpus

def get_top_tweet_bigrams(corpus, top: int = 10):
  vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0) 
  words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
  words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
  return words_freq[:top]

def get_top_tweet_trigrams(corpus, top: int = 10):
  vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
  bag_of_words = vec.transform(corpus)
  sum_words = bag_of_words.sum(axis=0) 
  words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
  words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
  return words_freq[:top]

def get_top_keywords(dataset, top: int = 10):
  keywords = dataset['keyword'].dropna()
  keyword_counts = keywords.value_counts().head(top)
  return keyword_counts

def get_top_locations(dataset, top=10):
  locations = dataset['location'].dropna()
  location_counts = locations.value_counts().head(top)
  return location_counts
