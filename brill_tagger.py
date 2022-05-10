# Natural Language Toolkit: Transformation-based learning

import os
import pickle
import time
import nltk
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def verbose_rule_format():
    """
    Exemplify Rule.format("verbose")
    """
    start_postag_brill_tagger(rule_format="verbose")


def high_accuracy_rules():
    start_postag_brill_tagger(num_sentences=3000, min_acc=0.96, min_score=10)


def start_postag_brill_tagger(
        templates=None,
        tagged_data=None,  # maximum number of rule instances to create
        num_sentences=1000,  # how many sentences of training and testing data to use
        max_rules=300,  # maximum number of rule instances to create
        min_score=3,  # the minimum score for a rule in order for it to be considered
        min_acc=None,  # the minimum score for a rule in order for it to be considered
        train=0.8,  # the fraction of  the corpus to be used for training (1=all)
        trace=3,  # the level of diagnostic tracing output to produce (0-4)
        randomize=False,
        rule_format="str",
        baseline_backoff_tagger=None,
        separate_baseline_data=False,
        cache_baseline_tagger=None,
):
    results = np.zeros(10)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

    # defaults
    baseline_backoff_tagger = baseline_backoff_tagger or REGEXP_TAGGER
    if templates is None:
        from nltk.tag.brill import brill24, describe_template_sets

        # some pre-built template sets taken from typical systems or publications are
        # available. Print a list with describe_template_sets()
        # for instance:
        templates = brill24()
    X, y = _demo_prepare_data(tagged_data, train, num_sentences, randomize, separate_baseline_data)

    # creating (or reloading from cache) a baseline tagger (unigram tagger)
    # this is just a mechanism for getting deterministic output from the baseline between
    # python versions
    for j, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if cache_baseline_tagger:
            if not os.path.exists(cache_baseline_tagger):
                baseline_tagger = UnigramTagger([list(zip(X_train, y_train))], backoff=baseline_backoff_tagger
                                                )
                with open(cache_baseline_tagger, "w") as print_rules:
                    pickle.dump(baseline_tagger, print_rules)
                print(
                    "Trained baseline tagger, pickled it to {}".format(
                        cache_baseline_tagger
                    )
                )
            with open(cache_baseline_tagger) as print_rules:
                baseline_tagger = pickle.load(print_rules)
                print(f"Reloaded pickled tagger from {cache_baseline_tagger}")
        else:
            baseline_tagger = UnigramTagger([list(zip(X_train, y_train))], backoff=baseline_backoff_tagger)
            print("Trained baseline tagger")
        print("Accuracy on test set (baseline tagger): {:0.4f}".format(
            baseline_tagger.accuracy([list(zip(X_test, y_test))])
        )
        )

        # creating a Brill tagger
        tbrill = time.time()
        trainer = BrillTaggerTrainer(
            baseline_tagger, templates, trace, ruleformat=rule_format
        )
        print("Training tbl tagger...")
        brill_tagger = trainer.train([list(zip(X_train, y_train))], max_rules, min_score, min_acc)
        print(f"Trained tbl tagger in {time.time() - tbrill:0.2f} seconds")
        results[j] = brill_tagger.accuracy([list(zip(X_test, y_test))])
        print("Accuracy on test set: %.4f" % results[j])

        # printing the learned rules, if learned silently
        if trace == 1:
            print("\nLearned rules: ")
            for (ruleno, rule) in enumerate(brill_tagger.rules(), 1):
                print(f"{ruleno:4d} {rule.format(rule_format):s}")
    print(f"Brill Tagger Trainer: {results.mean(axis=0):.3f}")


def _demo_prepare_data(
        tagged_data, train, num_sents, randomize, separate_baseline_data
):
    with open('text_data/shakespeare_merged.txt', 'r') as f:
        text = f.read()
    labeled_text = preprocess_text(text)
    X, y = map(list, zip(*labeled_text))
    # print(list(zip(X, y)))
    X = np.array(X)
    y = np.array(y)
    return X, y


NN_CD_TAGGER = RegexpTagger([(r"^-?[0-9]+(\.[0-9]+)?$", "CD"), (r".*", "NN")])

REGEXP_TAGGER = RegexpTagger(
    [
        (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),  # cardinal numbers
        (r"(The|the|A|a|An|an)$", "AT"),  # articles
        (r".*able$", "JJ"),  # adjectives
        (r".*ness$", "NN"),  # nouns formed from adjectives
        (r".*ly$", "RB"),  # adverbs
        (r".*s$", "NNS"),  # plural nouns
        (r".*ing$", "VBG"),  # gerunds
        (r".*ed$", "VBD"),  # past tense verbs
        (r".*", "NN"),  # nouns (default)`
    ]
)


def corpus_size(seqs):
    return len(seqs), sum(len(x) for x in seqs)


def preprocess_text(text):
    # Get the tokens
    tokens = nltk.word_tokenize(text)
    # Tags the tokens
    tagging = nltk.pos_tag(tokens)
    # Returns the list of tuples
    return tagging


if __name__ == "__main__":
    start_postag_brill_tagger()
    # start_postag_brill_tagger(ruleformat="verbose")
