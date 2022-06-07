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
    ne_tree_unigram = ne_chunk(unigram_data, binary=False)
    ne_tree_unigram_regex = ne_chunk(unigram_regex_data, binary=False)
    ne_tree_brill = ne_chunk(brill_tagger_data, binary=False)
    i=0
    for chunk in ne_tree_unigram:
       if hasattr(chunk, 'label'):
          i+=1
          print(chunk.label(), ' '.join(c[0] for c in chunk))
    j=0
    for chunk in ne_tree_unigram_regex:
       if hasattr(chunk, 'label'):
          j+=1
          print(chunk.label(), ' '.join(c[0] for c in chunk))
    k=0
    for chunk in ne_tree_brill:
       if hasattr(chunk, 'label'):
          k+=1
          print(chunk.label(), ' '.join(c[0] for c in chunk))
    print(f'Unigram tagger, liczba tagów: {i}')
    print(f'Unigram tagger+regex, liczba tagów: {j}')
    print(f'Brill tagger, liczba tagów: {k}')

if __name__ == '__main__':
    main()
