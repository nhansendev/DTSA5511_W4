import numpy as np
import urllib
from string import punctuation
from nltk import word_tokenize


PREM = str.maketrans("", "", punctuation)


def rem_punc(word):
    return word.translate(PREM).strip()


# Remove non-ascii characters
def rem_non_ascii(word):
    return word.encode("ascii", errors="ignore").decode()


# Decode special URL character encodings
def decode_URL(word):
    return urllib.parse.unquote(word)


# Remove double spaces, tabs, etc.
def remove_whitespace(word):
    return " ".join(word.split())


def clean_word(word):
    return remove_whitespace(rem_punc(rem_non_ascii(decode_URL(word))))


def lem_words(words, lemmatizer, only_unique=True, make_map=True):
    # Lemmatize each word in a list of words
    new_words = []
    if make_map:
        keyword_map = {}
    for word in words:
        subset = []
        # If the "word" is actually multiple words, then lemmatize them separately and recombine
        for tmp in np.unique(word.split(" ")):
            for t in ["n", "v", "a", "r"]:
                tmp = lemmatizer.lemmatize(tmp, t)
            subset.append(tmp)
        newword = " ".join(subset)
        if make_map:
            # Keep track of the new word/old word pairs
            keyword_map[word] = newword
        new_words.append(newword)

    if only_unique:
        if make_map:
            return np.unique(new_words), keyword_map
        else:
            return np.unique(new_words)
    else:
        if make_map:
            return new_words, keyword_map
        else:
            return new_words


def lem_sentences(sentences, stop_words, lemmatizer):
    # Apply lemmatization and cleaning to the dataset sentences
    lems = []
    for line in sentences:
        words = []
        for word in word_tokenize(line):
            if word.isalpha() and not word in stop_words:
                words.append(clean_word(word))

        lemmed_words = lem_words(words, lemmatizer, only_unique=False, make_map=False)

        if len(lemmed_words) < 1:
            # Nothing left after cleaning this sentence
            lemmed_words = "nan"

        lems.append(lemmed_words)

    return lems
