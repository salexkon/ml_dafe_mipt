from __future__ import annotations

import csv
import json
import re


import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}