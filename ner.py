import numpy as np
import warnings
import nltk
from nltk.chunk import ne_chunk
from unigram_tagger import unigram_standalone, unigram_with_regexp
from brill_tagger import start_postag_brill_tagger

def main():
    warnings.filterwarnings('ignore', '.*The least populated*', )
    _, unigram_data = unigram_standalone()
    _, unigram_regex_data = unigram_with_regexp()
    _, brill_tagger_data = start_postag_brill_tagger()
    for i, word in enumerate(unigram_data):
        if word[1] is None:
            unigram_data[i] = (word[0], '')
    for i, word in enumerate(unigram_regex_data):
        if word[1] is None:
            unigram_data[i] = (word[0], '')
    for i, word in enumerate(brill_tagger_data):
        if word[1] is None:
            unigram_data[i] = (word[0], '')
    # print([t[-1].replace(None, '') for t in unigram_data])
    ne_tree_unigram = ne_chunk(unigram_data, binary=False)
    ne_tree_unigram_regex = ne_chunk(unigram_regex_data, binary=False)
    ne_tree_brill = ne_chunk(brill_tagger_data, binary=False)
    print(ne_tree_unigram)
    print(ne_tree_unigram_regex)
    print(ne_tree_brill)



if __name__ == '__main__':
    main()
