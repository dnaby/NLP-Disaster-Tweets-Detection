import sys
sys.path.append('..') 

# Importations
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from collections import  Counter
from wordcloud import WordCloud

from src.data.make_dataset import (
  get_dataset, 
  create_corpus, 
  get_top_tweet_bigrams, 
  get_top_tweet_trigrams,
  get_top_keywords,
  get_top_locations
)

plt.style.use('ggplot')
stop=set(stopwords.words('english'))


def plot_disaster_and_non_disaster_bar_distribution(raw: bool = True) -> None:
  train, _ = get_dataset(raw)
  # Distribution of disaster and non-disaster tweets in the train set
  train_distribution = train['target'].value_counts().reset_index()
  train_distribution.columns = ['target', 'count']

  # Plot the distribution for the train set
  fig = px.bar(train_distribution, x='target', y='count', barmode='group',
              labels={'target': 'Disaster Tweet', 'count': 'Number of Tweets'},
              title='Distribution of Disaster and Non-Disaster Tweets in Train Set')
  fig.show()
  
def plot_disaster_and_non_disaster_pie_distribution(raw: bool = True) -> None:
  train, _ = get_dataset(raw)
  # Distribution of disaster and non-disaster tweets in the train set
  train_distribution = train['target'].value_counts().reset_index()
  train_distribution.columns = ['target', 'count']

  # Distribution of disaster and non-disaster tweets in the train set as a pie chart
  fig_pie = px.pie(train_distribution, values='count', names='target',
                  labels={'target': 'Disaster Tweet'},
                  title='Percentage of Disaster and Non-Disaster Tweets in Train Set')
  fig_pie.show()
  
def plot_tweet_length_histogram(raw: bool = True) -> None:
  train, _ = get_dataset(raw)
  # Histogram of tweet lengths for disaster and non-disaster tweets
  train['text_length'] = train['text'].apply(len)

  # Separate the data into disaster and non-disaster tweets
  disaster_tweets = train[train['target'] == 1]
  non_disaster_tweets = train[train['target'] == 0]

  # Plot histograms for disaster and non-disaster tweets side by side
  fig_hist = make_subplots(rows=1, cols=2, subplot_titles=('Disaster Tweets', 'Non-Disaster Tweets'))

  fig_hist_disaster = px.histogram(disaster_tweets, x='text_length', barmode='overlay',
                                  labels={'text_length': 'Tweet Length'},
                                  color_discrete_sequence=['red'])
  fig_hist_non_disaster = px.histogram(non_disaster_tweets, x='text_length', barmode='overlay',
                                      labels={'text_length': 'Tweet Length'},
                                      color_discrete_sequence=['green'])

  for trace in fig_hist_disaster['data']:
      fig_hist.add_trace(trace, row=1, col=1)

  for trace in fig_hist_non_disaster['data']:
      fig_hist.add_trace(trace, row=1, col=2)

  fig_hist.update_layout(title_text='Histogram of Tweet Lengths for Disaster and Non-Disaster Tweets')
  fig_hist.show()
  
def plot_tweet_word_length_histogram(raw: bool = True) -> None:
  train, _ = get_dataset(raw)
  # Histogram of word lengths for disaster and non-disaster tweets
  train['word_length'] = train['text'].apply(lambda x: len(x.split()))

  # Separate the data into disaster and non-disaster tweets
  disaster_tweets_word = train[train['target'] == 1]
  non_disaster_tweets_word = train[train['target'] == 0]

  # Plot histograms for disaster and non-disaster tweets side by side
  fig_hist_word = make_subplots(rows=1, cols=2, subplot_titles=('Disaster Tweets', 'Non-Disaster Tweets'))

  fig_hist_disaster_word = px.histogram(disaster_tweets_word, x='word_length', barmode='overlay',
                                        labels={'word_length': 'Word Length'},
                                        color_discrete_sequence=['red'])
  fig_hist_non_disaster_word = px.histogram(non_disaster_tweets_word, x='word_length', barmode='overlay',
                                            labels={'word_length': 'Word Length'},
                                            color_discrete_sequence=['green'])

  for trace in fig_hist_disaster_word['data']:
      fig_hist_word.add_trace(trace, row=1, col=1)

  for trace in fig_hist_non_disaster_word['data']:
      fig_hist_word.add_trace(trace, row=1, col=2)

  fig_hist_word.update_layout(title_text='Histogram of Word Lengths for Disaster and Non-Disaster Tweets')
  fig_hist_word.show()
  
def plot_average_word_length_for_each_tweet_histogram(raw: bool = True) -> None:
  train, _ = get_dataset(raw)
  # Calculate the average word length for each tweet
  train['avg_word_length'] = train['text'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))

  # Separate the data into disaster and non-disaster tweets
  disaster_tweets_avg_word = train[train['target'] == 1]['avg_word_length']
  non_disaster_tweets_avg_word = train[train['target'] == 0]['avg_word_length']

  # Create subplots
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

  # Plot for disaster tweets
  sns.histplot(disaster_tweets_avg_word, kde=True, ax=ax1, color='red')
  ax1.set_title('Disaster')
  ax1.set_xlabel('Average Word Length')
  ax1.set_ylabel('Density')

  # Plot for non-disaster tweets
  sns.histplot(non_disaster_tweets_avg_word, kde=True, ax=ax2, color='green')
  ax2.set_title('Not Disaster')
  ax2.set_xlabel('Average Word Length')
  ax2.set_ylabel('Density')

  # Set the main title
  fig.suptitle('Average Word Length in Each Tweet')

  # Show the plot
  plt.tight_layout()
  plt.show()
  
def plot_most_common_stopwords(raw: bool = True, top: int = 10) -> None:
  # Create corpus for non-disaster tweets
  non_disaster_corpus = create_corpus(disaster=False, raw=raw)
  disaster_corpus = create_corpus(disaster=True, raw=raw)

  # Count stopwords in non-disaster tweets
  dic_non_disaster = defaultdict(int)
  for word in non_disaster_corpus:
      if word in stop:
          dic_non_disaster[word] += 1

  non_disaster_top = sorted(dic_non_disaster.items(), key=lambda x: x[1], reverse=True)[:top]

  # Count stopwords in disaster tweets
  dic_disaster = defaultdict(int)
  for word in disaster_corpus:
      if word in stop:
          dic_disaster[word] += 1

  disaster_top = sorted(dic_disaster.items(), key=lambda x: x[1], reverse=True)[:top]

  # Prepare data for plotting
  x_non_disaster, y_non_disaster = zip(*non_disaster_top)
  x_disaster, y_disaster = zip(*disaster_top)

  # Create subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Non-Disaster Tweets', 'Disaster Tweets'))

  # Add bar chart for non-disaster tweets
  fig.add_trace(go.Bar(x=x_non_disaster, y=y_non_disaster, name='Non-Disaster', marker_color='green'), row=1, col=1)

  # Add bar chart for disaster tweets
  fig.add_trace(go.Bar(x=x_disaster, y=y_disaster, name='Disaster', marker_color='red'), row=1, col=2)

  # Update layout
  fig.update_layout(title_text=f'Top {top} Common Stopwords in Tweets',
                    xaxis_title='Stopwords',
                    yaxis_title='Frequency',
                    showlegend=False)

  fig.show()

def plot_most_common_words(raw: bool = True, top: int = 10) -> None:
  # Create corpus for non-disaster tweets
  non_disaster_corpus = create_corpus(disaster=False, raw=raw)
  disaster_corpus = create_corpus(disaster=True, raw=raw)
  
  # Count most common words in non-disaster tweets
  counter_non_disaster = Counter(non_disaster_corpus)
  most_non_disaster = counter_non_disaster.most_common()
  x_non_disaster = []
  y_non_disaster = []
  for word, count in most_non_disaster[:top]:
      if word not in stop:
          x_non_disaster.append(word)
          y_non_disaster.append(count)

  # Count most common words in disaster tweets
  counter_disaster = Counter(disaster_corpus)
  most_disaster = counter_disaster.most_common()
  x_disaster = []
  y_disaster = []
  for word, count in most_disaster[:top]:
      if word not in stop:
          x_disaster.append(word)
          y_disaster.append(count)

  # Create subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Non-Disaster Tweets', 'Disaster Tweets'))

  # Add bar chart for non-disaster tweets
  fig.add_trace(go.Bar(x=y_non_disaster, y=x_non_disaster, orientation='h', name='Non-Disaster', marker_color='green'), row=1, col=1)

  # Add bar chart for disaster tweets
  fig.add_trace(go.Bar(x=y_disaster, y=x_disaster, orientation='h', name='Disaster', marker_color='red'), row=1, col=2)

  # Update layout
  fig.update_layout(title_text=f'Top {top} Common Words in Tweets (Excluding Stopwords)',
                    xaxis_title='Count',
                    yaxis_title='Words',
                    showlegend=False)

  fig.show()

def plost_most_common_bigrams(raw: bool = True, top: int = 10) -> None:
  train, _ = get_dataset(raw)
  
  # Create corpus for disaster and non-disaster tweets
  non_disaster_corpus = train[train['target'] == 0]['text']
  disaster_corpus = train[train['target'] == 1]['text']

  # Get top bigrams
  top_non_disaster_bigrams = get_top_tweet_bigrams(non_disaster_corpus, top=top)
  top_disaster_bigrams = get_top_tweet_bigrams(disaster_corpus, top=top)

  # Prepare data for plotting
  x_non_disaster, y_non_disaster = map(list, zip(*top_non_disaster_bigrams))
  x_disaster, y_disaster = map(list, zip(*top_disaster_bigrams))

  # Create subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Non-Disaster Tweets', 'Disaster Tweets'))

  # Add bar chart for non-disaster tweets
  fig.add_trace(go.Bar(x=y_non_disaster, y=x_non_disaster, orientation='h', name='Non-Disaster', marker_color='green'), row=1, col=1)

  # Add bar chart for disaster tweets
  fig.add_trace(go.Bar(x=y_disaster, y=x_disaster, orientation='h', name='Disaster', marker_color='red'), row=1, col=2)

  # Update layout
  fig.update_layout(title_text=f'Top {top} Bigrams in Tweets',
                    xaxis_title='Count',
                    yaxis_title='Bigrams',
                    showlegend=False)

  fig.show()
  
def plost_most_common_trigrams(raw: bool = True, top: int = 10) -> None:
  train, _ = get_dataset(raw)
  
  # Create corpus for disaster and non-disaster tweets
  non_disaster_corpus = train[train['target'] == 0]['text']
  disaster_corpus = train[train['target'] == 1]['text']

  # Get top trigrams
  top_non_disaster_trigrams = get_top_tweet_trigrams(non_disaster_corpus, top=top)
  top_disaster_trigrams = get_top_tweet_trigrams(disaster_corpus, top=top)

  # Prepare data for plotting
  x_non_disaster_tri, y_non_disaster_tri = map(list, zip(*top_non_disaster_trigrams))
  x_disaster_tri, y_disaster_tri = map(list, zip(*top_disaster_trigrams))

  # Create subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Non-Disaster Tweets', 'Disaster Tweets'))

  # Add bar chart for non-disaster tweets
  fig.add_trace(go.Bar(x=y_non_disaster_tri, y=x_non_disaster_tri, orientation='h', name='Non-Disaster', marker_color='blue'), row=1, col=1)

  # Add bar chart for disaster tweets
  fig.add_trace(go.Bar(x=y_disaster_tri, y=x_disaster_tri, orientation='h', name='Disaster', marker_color='orange'), row=1, col=2)

  # Update layout
  fig.update_layout(title_text=f'Top {top} Trigrams in Tweets',
                    xaxis_title='Count',
                    yaxis_title='Trigrams',
                    showlegend=False)

  fig.show()

def plot_most_common_keywords(raw: bool = True, top: int = 10) -> None:
  train, _ = get_dataset(raw)
  
  # Define non_disaster_df and disaster_df
  non_disaster_df = train[train['target'] == 0]
  disaster_df = train[train['target'] == 1]

  # Get top keywords
  top_non_disaster_keywords = get_top_keywords(non_disaster_df, top=top)
  top_disaster_keywords = get_top_keywords(disaster_df, top=top)

  # Prepare data for plotting
  x_non_disaster_kw, y_non_disaster_kw = top_non_disaster_keywords.index, top_non_disaster_keywords.values
  x_disaster_kw, y_disaster_kw = top_disaster_keywords.index, top_disaster_keywords.values

  # Create subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Non-Disaster Keywords', 'Disaster Keywords'))

  # Add bar chart for non-disaster keywords
  fig.add_trace(go.Bar(x=y_non_disaster_kw, y=x_non_disaster_kw, orientation='h', name='Non-Disaster', marker_color='blue'), row=1, col=1)

  # Add bar chart for disaster keywords
  fig.add_trace(go.Bar(x=y_disaster_kw, y=x_disaster_kw, orientation='h', name='Disaster', marker_color='orange'), row=1, col=2)

  # Update layout
  fig.update_layout(title_text=f'Top {top} Keywords in Tweets',
                    xaxis_title='Count',
                    yaxis_title='Keywords',
                    showlegend=False)

  fig.show()

def plot_most_common_locations(raw: bool = True, top: int = 10) -> None:
  train, _ = get_dataset(raw)
  
  # Define non_disaster_df and disaster_df
  non_disaster_df = train[train['target'] == 0]
  disaster_df = train[train['target'] == 1]
  
  # Get top locations
  top_non_disaster_locations = get_top_locations(non_disaster_df, top=top)
  top_disaster_locations = get_top_locations(disaster_df, top=top)

  # Prepare data for plotting
  x_non_disaster_loc, y_non_disaster_loc = top_non_disaster_locations.index, top_non_disaster_locations.values
  x_disaster_loc, y_disaster_loc = top_disaster_locations.index, top_disaster_locations.values

  # Create subplots
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Non-Disaster Locations', 'Disaster Locations'))

  # Add bar chart for non-disaster locations
  fig.add_trace(go.Bar(x=y_non_disaster_loc, y=x_non_disaster_loc, orientation='h', name='Non-Disaster', marker_color='blue'), row=1, col=1)

  # Add bar chart for disaster locations
  fig.add_trace(go.Bar(x=y_disaster_loc, y=x_disaster_loc, orientation='h', name='Disaster', marker_color='orange'), row=1, col=2)

  # Update layout
  fig.update_layout(title_text=f'Top {top} Locations in Tweets',
                    xaxis_title='Count',
                    yaxis_title='Locations',
                    showlegend=False)

  fig.show()

def plot_wordcloud(raw: bool = True) -> None:
  train, _ = get_dataset(raw)
  
  # Define non_disaster_df and disaster_df
  non_disaster_df = train[train['target'] == 0]
  disaster_df = train[train['target'] == 1]
  
  # Generate word cloud for non-disaster tweets
  non_disaster_text = ' '.join(non_disaster_df['text'].dropna().values)
  non_disaster_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_disaster_text)

  # Generate word cloud for disaster tweets
  disaster_text = ' '.join(disaster_df['text'].dropna().values)
  disaster_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(disaster_text)

  # Plot the word clouds
  plt.figure(figsize=(15, 8))

  # Non-disaster word cloud
  plt.subplot(1, 2, 1)
  plt.imshow(non_disaster_wordcloud, interpolation='bilinear')
  plt.title('Non-Disaster Tweets Word Cloud')
  plt.axis('off')

  # Disaster word cloud
  plt.subplot(1, 2, 2)
  plt.imshow(disaster_wordcloud, interpolation='bilinear')
  plt.title('Disaster Tweets Word Cloud')
  plt.axis('off')

  plt.show()
