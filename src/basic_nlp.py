from functools import reduce
from itertools import zip_longest, filterfalse
from re import search, findall
from string import punctuation
from nltk import ngrams, word_tokenize, pos_tag, FreqDist, ne_chunk, Tree
from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet, cmudict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer
import re


# Ignore twython library missing, we aren't using it's functionality
# Must use nltk.download() and get the Opinion Lexicon and Vader Lexicon

PUNCTUATION_RE = "[\'\!\"\#\$\%\&\/\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\}\|\~\\u2026\\u2018\\u2019]"
TWEET_LINK_RE = "https://t.co/(\w)+"
TWEET_HANDLE_RE = "@(\w)+"

lemma = WordNetLemmatizer()
negWords = frozenset(opinion_lexicon.negative())
posWords = frozenset(opinion_lexicon.positive())
suffixes = list(line.rstrip() for line in open("suffixes.txt"))
vader = SentimentIntensityAnalyzer()
d = cmudict.dict()

chunk       = lambda posTagged: ne_chunk(posTagged, binary=True)
freq        = lambda grams: FreqDist(grams)
lemmatize   = lambda posTokens: [processPosTagsAndLemmatize(*wordPos) for wordPos in posTokens]
grams       = lambda tokens, n: list(ngrams(tokens, n))
pos         = lambda tokens: pos_tag(tokens)
posTagOnly  = lambda tagged: [tag for word, tag in tagged]
tokenize    = lambda text: word_tokenize(text)
wordCases   = lambda ulls: [wordCase(*ull) for ull in ulls]
tokPosToTok = lambda listTokPos: list(list(zip(*listTokPos))[0])
tokNoNE     = lambda chunked, removeNumbers=False: tokPosToTok(removeNamedEntities(chunked, removeNumbers))


def tokenizeFindAllRegex(r):
    regex = re.compile(r)
    return lambda s: re.findall(regex, s)

def capLetterFreq(ull):
    return reduce(lambda i, u: i + u[0], ull, 0) / reduce(lambda i, l: i + l[1], ull, 0)

def processPosTagsAndLemmatize(word, pos):
    return lemma.lemmatize(word, treebankToWordnetPOS(pos))

def removeNamedEntities(chunked, removeNumbers=True):
    def rec(t, r):
        if type(t) == Tree:
            if t.label() == 'NE':
                r.append(('NameTOK', '[NE]'))
            else:
                for child in t:
                    r.extend(rec(child, []))
        else:
            if removeNumbers:
                r.append(('NumTok', '[CD]')) if t[1] == "CD" else r.append(t)
            else:
                r.append(t)
        return r
    return rec(chunked, [])

def sentimentGrams(grams):
    return [{"LiuHu": sentimentLiuHu(gram), "Vader": sentimentVader(gram)} for gram in grams]

def sentimentLiuHu(gram):
    posWord = lambda word: word in posWords
    negWord = lambda word: word in negWords
    countPosNeg = lambda pn, word: (pn[0] + 1,
                                    pn[1],
                                    pn[2]) if posWord(word) else (pn[0],
                                                                  pn[1] + 1,
                                                                  pn[2]) if negWord(word) else (pn[0],
                                                                                                pn[1],
                                                                                                pn[2] + 1)
    p, n, u = reduce(countPosNeg, gram, [0, 0, 0])
    l = p + n + u
    return {"compound": round((p - n) / l, 3),
            "neg": round(n / l, 4),
            "neu": round(u / l, 4),
            "pos": round(p / l, 4)} if l > 0 else {"compound": 0,
                                                   "neg": 0,
                                                   "neu": 0,
                                                   "pos": 0}

def sentimentVader(gram):
    return vader.polarity_scores(" ".join(gram))

def tokenSuffixes(tokens):
    return [max(s2, key=len) for s2 in
            [s1 for s1 in
             [
                [s0 for s0 in suffixes if word.endswith(s0)]
                for word in tokens]
             if s1]]

def treebankToWordnetPOS(treebankPosTag):
    return {'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.ADV}.get(treebankPosTag[0], wordnet.NOUN)

def upperLowerLen(tokensOriginalCase):
    return [
        (
            sum([1 if letter.isupper() else 0 for letter in token]),
            sum([1 if letter.islower() else 0 for letter in token]),
            len(token),
            token
        )for token in tokensOriginalCase]

def wordCase(upper, lower, length, token):
    if upper == 0:
        return "NC"  # No Caps
    elif upper == length:
        return "AC"  # All Caps
    elif upper == 1 and token[0].isupper():
        return "FC"  # First Cap
    else:
        return "MC"  # Mixed Caps

def numSyllables(word):
    return [len(list(y for y in x if y[-1].isdigit()))for x in d[word.lower()]][0]

def hasVowel(word):
    for char in word.lower():
        if char in ['a','e','i','o','u']:
            return 1
    return 0

def hasPunctuation(word):
    return search(PUNCTUATION_RE, word) is not None

def inDict(word):
    try:
        d[word.lower()]
        return True
    except KeyError:
        return False


def punctuationFeatures(s):
    """
    s: input string
    returns {punctuation_mark: (raw #, % of length of s, % of total # of punctuation marks found in s)}
    example:
    punctuation_features("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed consequat magna eu facilisis!!?")
    {'!': (2, 0.0217, 0.4),
     ',': (1, 0.0109, 0.2),
     '.': (1, 0.0109, 0.2),
     '?': (1, 0.0109, 0.2)}
     """

    punctuation_found_list = findall(PUNCTUATION_RE, s)
    punctuation_dict = {p: (punctuation_found_list.count(p),
                            round(punctuation_found_list.count(p) / len(s), 4),
                            round(punctuation_found_list.count(p) / len(punctuation_found_list), 4)) for p in
                        punctuation_found_list}
    punctuation_dict.update({punct: (0, 0, 0) for punct in punctuation if punct not in punctuation_dict.keys()})

    return punctuation_dict
