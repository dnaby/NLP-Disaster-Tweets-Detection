import numpy as np
import pandas as pd
from src.data.make_dataset import (
  get_dataset, 
  create_corpus, 
  get_top_tweet_bigrams, 
  get_top_tweet_trigrams,
  get_top_keywords,
  get_top_locations
)
from ydata_profiling import ProfileReport


def make_profile_report(raw: bool = True) -> None:
    """Creates a profile report for the data set

  Parameters:
  raw (bool): If True, uses the raw dataset. Otherwise, uses the processed dataset.
  """
    train, test = get_dataset(raw)
    
    train_profile = ProfileReport(train, title="Train Set Profiling Report")
    train_profile.to_file(f"../reports/{'raw' if raw else 'processed'}/profiles/train_profile.html")
    
    test_profile = ProfileReport(test, title="Test Set Profiling Report")
    test_profile.to_file(f"../reports/{'raw' if raw else 'processed'}/profiles/test_profile.html")
    
    
    
