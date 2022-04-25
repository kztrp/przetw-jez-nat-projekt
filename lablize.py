import nltk
import math
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer, UnigramTagger

def main():

    with open('text_data/screenshot_20220329_193633.txt', 'r') as f:
        text = f.read()
        labeled_text = preprocess_text(text)
        k = math.floor(len(labeled_text)/4)
        # print(labeled_text)
        training_data = labeled_text[:k]
        baseline_data = labeled_text[k:3*k]
        gold_data = list(labeled_text[3*k:])
        # print(gold_data)
        testing_data = [untag([s]) for s in gold_data]
        backoff = RegexpTagger([
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
            (r'(The|the|A|a|An|an)$', 'AT'),   # articles
            (r'.*able$', 'JJ'),                # adjectives
            (r'.*ness$', 'NN'),                # nouns formed from adjectives
            (r'.*ly$', 'RB'),                  # adverbs
            (r'.*s$', 'NNS'),                  # plural nouns
            (r'.*ing$', 'VBG'),                # gerunds
            (r'.*ed$', 'VBD'),                 # past tense verbs
            (r'.*', 'NN')                      # nouns (default)
            ])

        unigram_tagger = UnigramTagger([baseline_data], backoff=backoff)
        print(unigram_tagger.accuracy([gold_data]))

def preprocess_text(text):
    """
    This function takes a text. Split it in tokens using word_tokenize.
    And then tags them using pos_tag from NLTK module.
    It outputs a list of tuples. Each tuple consists of a word and the tag with its
    part of speech.
    """
    # Get the tokens
    tokens = nltk.word_tokenize(text)
    # Tags the tokens
    tagging = nltk.pos_tag(tokens)
    # Returns the list of tuples
    return tagging


if __name__ == '__main__':
    main()
