import pytest
import sys
import os

# Add the parent of the parent directory to sys.path to import from src
parent_of_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_of_parent_dir)

from src.utilities.main import (
    remove_unknown_words,
    remove_single_chars,
    remove_URL,
    remove_html,
    remove_emoji,
    remove_punctuation,
    correct_spellings,
    replace_percent20_with_space,
    remove_accents,
    remove_non_necessary_spaces,
    stem_text,
    has_lengthening,
    reduce_lengthening,
    remove_numerical_values,
    expand_contractions,
    remove_ampersand,
    lemmatize_text,
    remove_stopwords,
    expand_glued_text, 

)

def test_remove_unknown_words():
    text = "Hello goaa I am here"
    expected = "Hello I am here"
    assert remove_unknown_words(text) == expected

def test_remove_single_chars():
    text = "a b c abc"
    expected = "abc"
    assert remove_single_chars(text) == expected

def test_remove_URL():
    text = "Check this link http://example.com for more info."
    expected = "Check this link  for more info."
    assert remove_URL(text) == expected

def test_remove_html():
    text = "<p>This is a <b>bold</b> statement.</p>"
    expected = "This is a bold statement."
    assert remove_html(text) == expected

def test_remove_emoji():
    text = "Hello 😊 World 🌍"
    expected = "Hello  World "
    assert remove_emoji(text) == expected

def test_remove_punctuation():
    text = "Hello, World!"
    expected = "Hello  World "
    assert remove_punctuation(text) == expected

def test_correct_spellings():
    text = "speling"
    expected = "spelling"
    assert correct_spellings(text) == expected

def test_replace_percent20_with_space():
    text = "Hello%20World"
    expected = "Hello World"
    assert replace_percent20_with_space(text) == expected


def test_remove_accents():
    text = "café naïve résumé"
    expected = "cafe naive resume"
    assert remove_accents(text) == expected

def test_remove_non_necessary_spaces():
    text = "  This  is  a test.  "
    expected = "This is a test."
    assert remove_non_necessary_spaces(text) == expected

def test_stem_text():
    text = "running runs"
    expected = "run run"
    assert stem_text(text) == expected

def test_has_lengthening():
    assert has_lengthening("sooooon") is True
    assert has_lengthening("soon") is False

def test_reduce_lengthening():
    text = "sooooon"
    expected = "soon"
    assert reduce_lengthening(text) == expected

def test_remove_numerical_values():
    text = "The year is 2024."
    expected = "The year is ."
    assert remove_numerical_values(text) == expected

def test_expand_contractions():
    text = "I'm going to the store."
    expected = "I am going to the store."
    assert expand_contractions(text) == expected

def test_remove_ampersand():
    text = "Hello &amp World"
    expected = "Hello  World"
    assert remove_ampersand(text) == expected

def test_lemmatize_text():
    text = "running runs"
    expected = "run run"
    assert lemmatize_text(text) == expected

def test_remove_stopwords():
    text = "This is a test sentence."
    expected = "test sentence."
    assert remove_stopwords(text) == expected

def test_expand_glued_text():
    text = "hello_world"
    expected = "hello world"
    assert expand_glued_text(text) == expected
