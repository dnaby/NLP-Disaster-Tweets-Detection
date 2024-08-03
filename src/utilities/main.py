import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import string
import re


nltk.download('stopwords')


spell = SpellChecker()

def remove_URL(text):
  url = re.compile(r'https?://\S+|www\.\S+')
  return url.sub(r'',text)

def remove_html(text):
  html=re.compile(r'<.*?>')
  return html.sub(r'',text)
  
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
  emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
  )
  return emoji_pattern.sub(r'', text)
  
def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def correct_spellings(text):
    if not isinstance(text, str):
        return text
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        corrected_word = spell.correction(word) if word in misspelled_words else word
        corrected_text.append(corrected_word if corrected_word is not None else "")
    return " ".join(corrected_text)  
  

def remove_stopwords(text):
  text = text.lower()
  text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
  return text
