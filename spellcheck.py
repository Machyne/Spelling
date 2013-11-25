from keyboardmap import str_dist
from tokenizer import tokenize
from tokenizer import is_word
from trainer import get_data
import keyboardmap
import nltk
import sys
import re


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


# A weighted average between the interpolated prob of the trigram
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
    old_word = words[-1]
    words = words[:-1]
    d = counts[words[0]][words[1]] if len(words) == 2 else counts[words[0]]
    keys = set([x for x in d.keys() if x != 0 and d[x][0] > 0])
    cands = [(w, prob(words + [w], counts, next)) for w in keys] if d[0] > 0 else []
    gen = (x for x in keyboardmap.alphabet
           if keyboardmap.csub(x, old_word[0]) <= keyboardmap.similar_letter)
    for k in gen:
        for w in fld[k]:
            if w not in keys:
                cands.append((w, 0))
    return cands


# This will return the n best matches to 'word' in the list 'candidates'
# weighted by probability, and excluding those with distance greater than 10
def get_best_matches(word, candidates):
    # this is the scale factor for probabilities
    scale = 675.0
    # the maximum (and target) number of matches to be returned
    number_to_return = 10
    # nothing with a final score greater than this will be returned
    max_distance = 7.5
    matches = []
    for w, score in candidates:
        if(abs(len(w) - len(word)) > 3):
            continue
        dist = str_dist(word, w)
        if dist <= max_distance:
            matches.append((w, dist - scale * score))
    matches.sort(key=lambda pair: pair[1])
    return [pair[0] for pair in matches[:number_to_return]]


# Tries to convert a string to an int, returns None if it fails
def toi(n):
    try:
        return int(n)
    except ValueError:
        return None


# This displays the potential word options and returns what the user chose.
def get_choice(word, context, options):
    sys.stderr.write("\n\nIn the phrase:\n" + ' '.join(context) + "\nYou typed \"" + word + '".')
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


# This merges the changes in the new sentence into the old sentence preserving
# first-letter-capitolization. This is used to display the corrected version
# when not run in testing mode.
def merge_changes(old_s, new_s):
    for i, token in enumerate(old_s):
        if is_word(token.lower()):
            w = new_s.pop(0)
            if(token[0].isupper()):
                w = w[0].upper() + w[1:]
            old_s[i] = w
    return old_s


# This method determines whether or not a word will be marked as 'incorrect' and
# will therefore need to be changed.
def needs_change(w, best_matches, p):
    global min_skip_prob
    a = w not in best_matches[:3] or (p < min_skip_prob and best_matches[0] != w)
    W = filter(lambda x: x in keyboardmap.alphabet, w)
    b = W not in best_matches[:3] or (p < min_skip_prob and best_matches[0] != W)
    c = (w == 'i') and w in best_matches
    return (a and b) and not c


# The main method for non-training mode
def main():
    global max_check_prob
    # first get the input (either from stdin or the first arg)
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
    # read in the trained data
    vocab, fl_dict, counts = get_data()
    # an array of fixed sentences (represented as token arrays)
    new_sents = []
    for s in sents:
        # s2 is the words only, lowercase version of the sentence
        s2 = ['<BEGIN>'] + [w.lower() for w in s if is_word(w.lower())]
        # iterate through the words
        for i, w in enumerate(s2):
            # ignore '<BEGIN>'
            if i == 0:
                continue
            trigram = s2[max(i - 2, 0):i + 1]
            # The next word in the sentence
            next = s2[i + 1] if i + 1 < len(s2) else None
            # The surrounding few words
            context = s2[max(i - 4, 1):min(len(s2), i + 4)]
            p = prob(trigram, counts)
            if w not in vocab or p < max_check_prob:
                candidates = get_candidates(trigram, next, counts, fl_dict)
                best_matches = get_best_matches(w, candidates)
                if needs_change(w, best_matches, p):
                    choice = get_choice(w, context, best_matches)
                    s2[i] = choice
        new_sents.append(merge_changes(s, s2[1:]))
    print ' '.join(''.join(w) for w in new_sents)


def test():
    global max_check_prob
    # first get the input
    infile = open(sys.argv[1])
    sents = [tokenize(s) for s in nltk.sent_tokenize(infile.read())]
    infile.close()
    # read in the trained data
    vocab, fl_dict, counts = get_data()
    # [fixes, double_checks, couldnt_fixes, unwanted_removals]
    error_count = [0, 0, 0, 0]
    for s in sents:
        # s2 is the words only, lowercase version of the sentence
        s2 = ['<BEGIN>'] + [w for w in s if is_word(w.lower())]
        # tagged errors are represented by "wordXXXwrrrd"
        # this next part splits those in to tuples.
        for i in xrange(1, len(s2)):
            if 'XXX' in s2[i]:
                l, _, r = s2[i].partition('XXX')
                s2[i] = (l.lower(), r.lower())
            else:
                s2[i] = s2[i].lower()
        # iterate through the words
        for i, w in enumerate(s2):
            # ignore '<BEGIN>'
            if i == 0:
                continue
            # The Wrong word (if there is an error)
            W = w[1] if type(w) == tuple else w
            # The Right word (either way)
            R = w[0] if type(w) == tuple else w
            trigram = s2[max(i - 2, 0):i + 1]
            # if there are tuples in the trigram, use the original input
            for i in xrange(len(trigram)):
                if type(trigram[i]) == tuple:
                    trigram[i] = trigram[i][1]
            next = s2[i + 1] if i + 1 < len(s2) else None
            # if next is a tuple, use the original input
            if type(next) == tuple:
                next = next[1]
            p = prob(trigram, counts)
            if W not in vocab or p < max_check_prob:
                candidates = get_candidates(trigram, next, counts, fl_dict)
                best_matches = get_best_matches(W, candidates)
                if needs_change(W, best_matches, p):
                    # The end of what will be printed
                    end = R + ',"' + str(best_matches) + '"'
                    # if the Right word is in the best matches
                    if R in best_matches:
                        # if the corpus marked this as an error
                        if W != R:
                            # fix
                            print W + ',~>,' + end
                            error_count[0] += 1
                        # else the corpus did not mark this as an error
                        else:
                            # double_check
                            print W + ',~~,' + end
                            error_count[1] += 1
                    # else the Right word is not in the best matches
                    else:
                        # if the corpus marked this as an error
                        if W != R:
                            # couldnt_fix
                            print W + ',??,' + end
                            error_count[2] += 1
                        # else the corpus did not mark this as an error
                        else:
                            # unwanted_removal
                            print W + ',--,' + end
                            error_count[3] += 1
    sys.stderr.write('Total Corrections: ' + str(error_count[0]) + '\n')
    sys.stderr.write('Total Double Checks: ' + str(error_count[1]) + '\n')
    sys.stderr.write('Total Couldnt Fix: ' + str(error_count[2]) + '\n')
    sys.stderr.write('Total Unwanted Removals: ' + str(error_count[3]) + '\n')


# The -t flag runs this under testing mode
if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '-t':
        test()
    else:
        main()
