from keyboarmap import str_dist
from tokenizer import tokenize
from tokenizer import is_word
from trainer import get_data
# import keyboarmap
import nltk
import sys
import os
import re
if False:
    print os


# The maximum probability for rechecking a word.
max_check_prob = 0.02
# The minimum probability for potentially not rechecking.
min_skip_prob = 0.0001


# From my classifier.py
def raw_prob(counts, w1, w2=None, w3=None):
    if w3 is None:
        if w2 is None:
            return 0.0 if counts[w1][0] == 0 else float(counts[w1][0]) / counts[0]
        else:
            return 0.0 if counts[w1][w2][0] == 0 \
                else float(counts[w1][w2][0]) / counts[w1][0]
    else:
        return 0.0 if counts[w1][w2][w3][0] == 0 \
            else float(counts[w1][w2][w3][0]) / counts[w1][w2][0]


# The linearly interpolated probability
def interp_prob(words, counts):
    uni_gr = raw_prob(counts, words[-1])
    bii_gr = raw_prob(counts, words[-2], words[-1])
    tri_gr = raw_prob(counts, words[0], words[1], words[2]) if len(words) == 3 else 0
    return 0.1 * uni_gr + 0.4 * bii_gr + 0.5 * tri_gr


# The average between the interpolated prob of the trigram
# and the next trigram.
def prob(trigram, counts, next=None):
    p1 = interp_prob(trigram, counts)
    if next is None:
        return p1
    gram2 = trigram + [next]
    if len(gram2) > 3:
        gram2 = gram2[-3:]
    p2 = interp_prob(gram2, counts)
    return p1 * 0.6 + p2 * 0.4


# Returns a list of the most likely words with their probabilities
def get_candidates(words, next, counts, fld):
    scale = 675.0
    old_word = words[-1]
    words = words[:-1]
    d = counts[words[0]][words[1]] if len(words) == 2 else counts[words[0]]
    keys = set([x for x in d.keys() if x != 0 and d[x][0] > 0])
    cands = [(w, scale * prob(words + [w], counts, next)) for w in keys] if d[0] > 0 else []
    # gen = (x for x in keyboarmap.alphabet if keyboarmap.csub(x, old_word[0]) < keyboarmap.no_adj)
    # for k in gen:
    for w in fld[old_word[0]]:
        if w not in keys:
            cands.append((w, 0))
    return cands


# This will return the n best matches to 'word' in the list 'candidates'
# weighted by probability, and excluding those with distance greater than 10
def get_best_matches(word, candidates):
    number_to_return = 10
    max_distance = 7.5
    match_values = []
    for w, score in candidates:
        if(abs(len(w) - len(word)) > 3):
            continue
        dist = str_dist(word, w)
        if dist <= max_distance:
            match_values.append((w, dist - score))
    match_values.sort(key=lambda pair: pair[1])
    return [pair[0] for pair in match_values[:number_to_return]]


# Tries to convert a string to an int, returns None if it fails
def toi(n):
    try:
        return int(n)
    except ValueError:
        return None


def get_choice(word, context, options):
    # os.system('cls' if os.name == 'nt' else 'clear')
    sys.stderr.write("In the phrase:\n" + ' '.join(context) + "\nYou typed \"" + word + '".')
    sys.stderr.write(" Did you mean:\n")
    sys.stderr.write("0: <no change>\n")
    sys.stderr.write('\n'.join([str(i + 1) + ': ' + w for i, w in enumerate(options)]))
    sys.stderr.write("\n<or input a word>\n")
    choice = sys.stdin.readline().strip()
    if re.match(r'\s*$', choice):
        choice = '1'
    c = toi(choice)
    if c is not None and c >= 0 and c - 1 < len(options):
        choice = word if c == 0 else options[c - 1]
    return choice


def merge_changes(old_s, new_s):
    for i, token in enumerate(old_s):
        if is_word(token.lower()):
            w = new_s.pop(0)
            if(token[0].isupper()):
                w = w[0].upper() + w[1:]
            old_s[i] = w
    return old_s


def main():
    global min_prob
    infile = None
    is_stdin = False
    if len(sys.argv) < 2:
        infile = sys.stdin
        is_stdin = True
    else:
        infile = open(sys.argv[1])
    sents = [tokenize(s) for s in nltk.sent_tokenize(infile.read())]
    if not is_stdin:
        infile.close()
    vocab, fl_dict, counts = get_data()
    new_sents = []
    for s in sents:
        s2 = ['<BEGIN>'] + [w.lower() for w in s if is_word(w.lower())]
        for i, w in enumerate(s2):
            if i == 0:
                continue
            trigram = s2[max(i - 2, 0):i + 1]
            next = s2[i + 1] if i + 1 < len(s2) else None
            context = s2[max(i - 4, 1):min(len(s2), i + 4)]
            p = prob(trigram, counts)
            if w not in vocab or p < max_check_prob:
                # print "Error: '" + w + "'", p
                candidates = get_candidates(trigram, next, counts, fl_dict)
                best_matches = get_best_matches(w, candidates)
                # print best_matches
                if w not in best_matches[:3] or (p < min_skip_prob and best_matches[0] != w):
                    choice = get_choice(w, context, best_matches)
                    s2[i] = choice
        new_sents.append(merge_changes(s, s2[1:]))
    print ' '.join(''.join(w) for w in new_sents)


def test():
    global min_prob
    infile = None
    is_stdin = False
    if len(sys.argv) < 2:
        infile = sys.stdin
        is_stdin = True
    else:
        infile = open(sys.argv[1])
    sents = [tokenize(s) for s in nltk.sent_tokenize(infile.read())]
    if not is_stdin:
        infile.close()
    vocab, fl_dict, counts = get_data()
    # fixes, doublechecks, couldnt_fix, unwanted_removals
    error_count = [0, 0, 0, 0]
    for s in sents:
        s2 = ['<BEGIN>'] + [w for w in s if is_word(w.lower())]
        for i in xrange(1, len(s2)):
            if 'XXX' in s2[i]:
                l, _, r = s2[i].partition('XXX')
                s2[i] = (l.lower(), r.lower())
            else:
                s2[i] = s2[i].lower()
        for i, w in enumerate(s2):
            if i == 0:
                continue
            # Wrong?
            W = w[1] if type(w) == tuple else w
            # Right.
            R = w[0] if type(w) == tuple else w
            trigram = s2[max(i - 2, 0):i + 1]
            for i in xrange(len(trigram)):
                if type(trigram[i]) == tuple:
                    trigram[i] = trigram[i][1]
            next = s2[i + 1] if i + 1 < len(s2) else None
            if type(next) == tuple:
                next = next[1]
            p = prob(trigram, counts)
            if W not in vocab or p < max_check_prob:
                candidates = get_candidates(trigram, next, counts, fl_dict)
                best_matches = get_best_matches(W, candidates)
                if W not in best_matches[:2]:
                    if R in best_matches:
                        print W + ':' + R
                        if W != R:
                            # fix
                            error_count[0] += 1
                        else:
                            # doublecheck
                            error_count[1] += 1
                    else:
                        if W != R:
                            print W + ':?'
                            # couldnt_fix
                            error_count[2] += 1
                        else:
                            print W + ':-'
                            # unwanted_removal
                            error_count[3] += 1
    print 'Total Corrections:', error_count[0]
    print 'Total Double Checks:', error_count[1]
    print 'Total Couldnt Fix:', error_count[2]
    print 'Total Unwanted Removals:', error_count[3]


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '-t':
        test()
    else:
        main()
