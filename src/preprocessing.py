import re # regular expressions for text cleaning(removing digits,pattern matching)
import string # contains punctuation(string.punctuation)
from typing import Iterable  # iterable(tuple,list,set,string,dictionary,pandas series) anything that u can loop  over (for loop)

import pandas as pd
from nltk.stem import PorterStemmer # for stemming words ,nltk(nl toolkit)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS) # y only set ?? -> faster lookup , O(1) time complexity, if list then O(n).
STEMMER = PorterStemmer() # creates stemmer object


def to_lower(text: str) -> str: # output string in lowercase
    """Convert text to lowercase."""
    return str(text).lower()#str() converts anything into string( text= 123, str(text) = "123")
 # if text = None,  "None" str() here prevents errors like direct None.lower()

def remove_punctuation_and_digits(text: str) -> str:
    """Remove punctuation and digits; keep words and spaces."""
    text = str(text).translate(str.maketrans("", "", string.punctuation)) #str.maketrans(old,new,remove),, .translate() = remove rule
    text = re.sub(r"\d+", " ", text) #re.sub(pattern,replacement,text) 
    text = re.sub(r"\s+", " ", text).strip() #\s= whitespace,strip() = remove leading and trailing spaces
    return text


def remove_stopwords(text: str, stopwords: Iterable[str] | None = None) -> str: # None(optional parameter) = None(default), that meanms u can call in 2 ways 1st provide default stopwords or custom stopwords(iterable)
    """Remove stopwords from text."""
    active_stopwords = set(stopwords) if stopwords is not None else STOPWORDS #ternery operator (if user give custom use them else use default)
    # if stopwords is not None:
    # active_stopwords = set(stopwords)
    # else:
    # active_stopwords = STOPWORDS
    tokens = [token for token in str(text).split() if token not in active_stopwords] #split() = split sentences into words.
    return " ".join(tokens) # join words back into sentence.


def apply_stemming(text: str) -> str:
    """Apply Porter stemming token-wise."""
    return " ".join(STEMMER.stem(token) for token in str(text).split())


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for one text value."""
    if text is None:
        return ""

    text = to_lower(text)
    text = remove_punctuation_and_digits(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    return text


def preprocess_series(text_series: pd.Series) -> pd.Series:
    """Apply preprocessing to a pandas Series."""
    return text_series.fillna("").astype(str).apply(preprocess_text)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy with processed text column."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    processed = df.copy()
    processed[text_column] = preprocess_series(processed[text_column])
    return processed
