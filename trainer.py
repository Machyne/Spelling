# from tokenizer import condense_tokens
from collections import defaultdict
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from tokenizer import is_word
from nltk.corpus import brown
import cProfile
import cPickle
import sys
import os
SER_FILE = 'ser/gutenberg_data.pickle'

"""
The corpora include:

BROWN - nltk

GUTENBERG -nltk
  Austen: Emma
  Austen: Persuasion
  Austen: Sense
  Bible: Kjv
  Blake: Poems
  Bryant: Stories
  Burgess: Busterbrown
  Carroll: Alice
  Chesterton: Ball
  Chesterton: Brown
  Chesterton: Thursday
  Edgeworth: Parents
  Melville: Moby Dick
  Milton: Paradise
  Shakespeare: Caesar
  Shakespeare: Hamlet
  Shakespeare: Macbeth
  Whitman: Leaves

REUTERS - nltk
"""


# returns a dictionary with a default value of another instance of this kind of
# dictionary and the initial value of the dictionary at key 0 is 0.
def get_count_dict():
    d = defaultdict(get_count_dict)
    d[0] = 0
    return d


# Creates a dictionary that will contain a dictionary for each unigram,
# which then contains a dictionary holding the second element of a bigram,
# each bi-gram consists of a dictionary that contains the third element of the
# trigram. The key 0 gives the count.
def get_counts(token_list_list):
    counts = get_count_dict()
    for token_list in token_list_list:
        for i, t in enumerate(token_list):
            counts[0] += 1
            counts[t][0] += 1
            if i + 1 < len(token_list):
                counts[t][token_list[i + 1]][0] += 1
            if i + 2 < len(token_list):
                counts[t][token_list[i + 1]][token_list[i + 2]][0] += 1
    return counts


# This returns the vocab and the count dict. If they don't already
# exists in a serialized form, they are created.
def get_data():
    try:
        with open(SER_FILE, 'r') as f:
            vocab, fld, counts = cPickle.load(f)
            return vocab, fld, counts
    except:
        vocab, sents = create_data()
        counts = get_counts(sents)
        fld = get_fl_dict(vocab)
        with open(SER_FILE, 'w+') as f:
            cPickle.dump((vocab, fld, counts), f)
        return vocab, fld, counts


# Read the nltk gutenberg data, and create vocab and sentence list.
def create_data():
    vocab = set()
    sentences = []
    corpora = [gutenberg, brown, reuters]
    for C in corpora:
        for fileid in C.fileids():
            for sent in C.sents(fileid):
                sent = [w.lower() for w in sent if is_word(w)]
                # sent = condense_tokens(sent)
                sentences.append(['<BEGIN>'] + sent)
                for w in sent:
                    vocab.add(w)
    return vocab, sentences


# Returns a dictionary where the key is the first letter and the
# value is a list of all words that start with that letter.
def get_fl_dict(vocab):
    d = {}
    keys = set()
    for word in vocab:
        k = word[0]
        if k not in keys:
            d[k] = [word]
            keys.add(k)
        else:
            d[k].append(word)
    return d


def train():
    if os.path.isfile(SER_FILE):
        msg = 'Error: {} already exists.\n'
        msg += 'To force update, run \'{} -r\'\n'
        msg = msg.format(SER_FILE, sys.argv[0])
        sys.stderr.write(msg)
        exit(1)
    print 'Reading data...'
    vocab, sents = create_data()
    print 'Getting counts...'
    counts = get_counts(sents)
    print 'Creating first letter dictionary...'
    fld = get_fl_dict(vocab)
    with open(SER_FILE, 'w+') as f:
        print 'Saving...'
        cPickle.dump((vocab, fld, counts), f)
    print 'Done.'


def main():
    if '-r' in sys.argv[1:] or '-rp' in sys.argv[1:] or '-pr' in sys.argv[1:]:
        if os.path.isfile(SER_FILE):
            os.remove(SER_FILE)
    if '-p' in sys.argv[1:] or '-rp' in sys.argv[1:] or '-pr' in sys.argv[1:]:
        cProfile.run('train()')
    else:
        train()


if __name__ == '__main__':
    main()
