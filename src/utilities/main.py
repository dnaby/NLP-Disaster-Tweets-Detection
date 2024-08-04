import contractions
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer


import re
import string
import spacy
from spellchecker import SpellChecker
from unidecode import unidecode


nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()




def remove_URL(text: str) -> str:
    """
    Removes URLs from the given text.

    Parameters:
    text (str): The text from which URLs will be removed.

    Returns:
    str: The text with URLs removed.
    """
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_html(text: str) -> str:
    """
    Removes HTML tags from the given text.

    Parameters:
    text (str): The text from which HTML tags will be removed.

    Returns:
    str: The text with HTML tags removed.
    """
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text: str) -> str:
    """
    Removes emojis from the given text.

    Parameters:
    text (str): The text from which emojis will be removed.

    Returns:
    str: The text with emojis removed.
    """
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

def remove_punctuation(text: str) -> str:
    """
    Removes punctuation from the given text.

    Parameters:
    text (str): The text from which punctuation will be removed.

    Returns:
    str: The text with punctuation removed.
    """
    if not isinstance(text, str):
      return text  # Return the original value if it's not a string
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def correct_spellings(text: str) -> str:
    """
    Corrects spellings in the given text.

    Parameters:
    text (str): The text in which spellings will be corrected.

    Returns:
    str: The text with spellings corrected.
    """
    if not isinstance(text, str):
        return text
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        corrected_word = spell.correction(word) if word in misspelled_words else word
        corrected_text.append(corrected_word if corrected_word is not None else "")
    return " ".join(corrected_text)
  
def replace_percent20_with_space(text: str) -> str:
    """
    Replaces all occurrences of "%20" with a space in the given text.

    Parameters:
    text (str): The text in which "%20" will be replaced with a space.

    Returns:
    str: The text with all "%20" replaced with a space.
    """
    if not isinstance(text, str):
        return text  # Return the original value if it's not a string
    return text.replace("%20", " ")

def remove_weird_content(text: str) -> str:
    """
    Removes weird content like Û, Ï, Ò from the given text.

    Parameters:
    text (str): The text from which weird content will be removed.

    Returns:
    str: The text with weird content removed.
    """
    weird_chars = ['Û', 'Ï', 'Ò']
    for char in weird_chars:
        text = text.replace(char, '')
    return text

def remove_accents(text: str) -> str:
    """
    Removes accents from the given text.

    Parameters:
    text (str): The text from which accents will be removed.

    Returns:
    str: The text with accents removed.
    """
    if not isinstance(text, str):
        return text  # Return the original value if it's not a string
    return unidecode(text)

def remove_non_necessary_spaces(text: str) -> str:
    """
    Removes non-necessary spaces from the given text.

    Parameters:
    text (str): The text from which non-necessary spaces will be removed.

    Returns:
    str: The text with non-necessary spaces removed.
    """
    if not isinstance(text, str):
      return text  # Return the original value if it's not a string
    text = text.replace("  ", " ")  # Replace double spaces with single space
    text = text.strip()  # Remove leading and trailing spaces
    return text

def stem_text(text: str) -> str:
    """
    Stems the given text using Porter Stemmer.

    Parameters:
    text (str): The text to be stemmed.

    Returns:
    str: The stemmed text.
    """
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def remove_stopwords(text: str) -> str:
    if not isinstance(text, str):
        return text
    else:
        our_stopwords = ["amp", "https", "one", "new", "go", "see","say","know","come","think","make","want","new","via","s","u","news","rt", "im"]
        stopwords = our_stopwords + nltk_stopwords.words('english')

        text = text.lower()
        # Remove stopwords
        cleaned_text = " ".join([word for word in text.split() if word not in stopwords])
    
        return cleaned_text


def remove_numerical_values(text: str) -> str:
    """
    Removes numerical values from the given text using regular expressions.

    Parameters:
    text (str): The text from which numerical values will be removed.

    Returns:
    str: The text with numerical values removed.
    """
    if not isinstance(text, str):
      return text  # Return the original value if it's not a string
    numerical_pattern = re.compile(r'\d+')
    return numerical_pattern.sub(r'', text)

def expand_contractions(text: str) -> str:
    """
    Expands common contractions in the given text using the contractions library.

    Parameters:
    text (str): The text to be expanded.

    Returns:
    str: The expanded text.
    """
    return contractions.fix(text)

def remove_ampersand(text: str) -> str:
    """
    Removes HTML encoded ampersands from the given text.

    Parameters:
    text (str): The text from which HTML encoded ampersands will be removed.

    Returns:
    str: The text with HTML encoded ampersands removed.
    """
    if not isinstance(text, str):
        return text  # Return the original value if it's not a string
    return text.replace('amp', '')  # Remove &amp; entirely

