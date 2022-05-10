import nltk
import math
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer, UnigramTagger
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import warnings

def main():
    warnings.filterwarnings('ignore', '.*The least populated*', )

    unigram_standalone()
    unigram_with_regexp()

def unigram_standalone():
    results = np.zeros(10)
    with open('text_data/shakespeare_merged.txt', 'r') as f:
        text = f.read()
    labeled_text = preprocess_text(text)
    # print(labeled_text)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    X, y = map(list, zip(*labeled_text))
    # print(list(zip(X, y)))
    X = np.array(X)
    y = np.array(y)
    for j, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        unigram_tagger = UnigramTagger([list(zip(X_train, y_train))])
        results[j] = unigram_tagger.accuracy([list(zip(X_test, y_test))])
    print(f"Unigram (standalone): {results.mean(axis=0):.3f}")
    return results


def unigram_with_regexp():
    results = np.zeros(10)
    with open('text_data/shakespeare_merged.txt', 'r') as f:
        text = f.read()
    labeled_text = preprocess_text(text)
    k = math.floor(len(labeled_text) / 4)
    # print(labeled_text)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    X, y = map(list, zip(*labeled_text))
    # print(list(zip(X, y)))
    X = np.array(X)
    y = np.array(y)
    backoff = RegexpTagger([
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'(The|the|A|a|An|an)$', 'AT'),  # articles
        (r'.*able$', 'JJ'),  # adjectives
        (r'.*ness$', 'NN'),  # nouns formed from adjectives
        (r'.*ly$', 'RB'),  # adverbs
        (r'.*s$', 'NNS'),  # plural nouns
        (r'.*ing$', 'VBG'),  # gerunds
        (r'.*ed$', 'VBD'),  # past tense verbs
        (r'.*', 'NN')  # nouns (default)
    ])
    for j, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        unigram_tagger = UnigramTagger([list(zip(X_train, y_train))],
            backoff=backoff)
        results[j] = unigram_tagger.accuracy([list(zip(X_test, y_test))])
    print(f"Unigram (with Regexp): {results.mean(axis=0):.3f}")
    return results

def preprocess_text(text):

    # Get the tokens
    tokens = nltk.word_tokenize(text)
    # Tags the tokens
    tagging = nltk.pos_tag(tokens)
    # Returns the list of tuples
    return tagging


if __name__ == '__main__':
    main()
