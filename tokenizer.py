import nltk
import re


# Returns true if w is a word (for a global definition)
def is_word(w):
    ret = w == '<BEGIN>'
    ret = ret or (re.match(r'[a-z\'\-.]{2,}|[a-z]', w) is not None)
    ret = ret and w != "''"
    return ret


# This will be used for condensing clitics split from their original word
def is_clitic(w):
    return w in ["'ll", "n't", "'ve", "'m", "'s", "s", "ll", "t", "'d'", "d", "ve"]


# If there is a word adjacent to a clitic, combine them.
def condense_tokens(wordlist):
    i = 0
    newlist = []
    while i < len(wordlist):
        curr = wordlist[i]
        if i == len(wordlist) - 1:
            newlist.append(curr)
            break
        next = wordlist[i + 1]
        if is_word(curr) and is_clitic(next):
            newlist.append(curr + next)
            i += 2
        else:
            newlist.append(curr)
            i += 1
    return newlist


# This uses the nltk word tokenizer, but adds in whitespace.
# It also uses the condense function above.
def tokenize(sentence):
    s = nltk.word_tokenize(sentence)
    s = [(x if x != "''" and x != '``' else '"') for x in s]
    newlist = []
    dummy = ''
    for word in s:
        while not sentence.startswith(word):
            dummy += sentence[0]
            sentence = sentence[1:]
        if dummy != '':
            newlist.append(dummy)
            dummy = ''
        sentence = sentence[len(word):]
        newlist.append(word)
    if sentence != '':
        newlist.append(sentence)
    return condense_tokens(newlist)
