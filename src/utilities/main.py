import re
import string
from spellchecker import SpellChecker

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