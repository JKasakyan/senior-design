from functools import reduce
from itertools import zip_longest, filterfalse
from nltk import ngrams, word_tokenize, pos_tag, FreqDist, ne_chunk, Tree
from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordTokenizer


# Ignore twython library missing, we aren't using it's functionality
# Must use nltk.download() and get the Opinion Lexicon and Vader Lexicon

PUNCTUATION_RE = "[\'\!\"\#\$\%\&\/\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\}\|\~\\u2026]"
TWEET_LINK_RE = "https://t.co/(\w)+"
TWEET_HANDLE_RE = "@(\w)+"

class nlp:
    lemma = WordNetLemmatizer()
    negWords = frozenset(opinion_lexicon.negative())
    posWords = frozenset(opinion_lexicon.positive())
    suffixes = list(line.rstrip() for line in open("suffixes.txt"))
    vader = SentimentIntensityAnalyzer()

    chunk = lambda self, posTagged: ne_chunk(posTagged, binary=True)
    freq = lambda self, grams: FreqDist(grams)
    lemmatize = lambda self, posTokens: [self.processPosTagsAndLemmatize(*wordPos) for wordPos in posTokens]
    ngrams = lambda self, tokens, n: list(ngrams(tokens, n))
    pos = lambda self, tokens: pos_tag(tokens)
    posTagOnly = lambda self, tagged: [tag for word, tag in tagged]
    tokenize = lambda self, text: word_tokenize(text)
    wordCases = lambda self, ulls: [self.wordCase(*ull) for ull in ulls]

    def capLetterFreq(self, ull):
        return reduce(lambda i, u: i + u[0], ull, 0) / reduce(lambda i, l: i + l[1], ull, 0)

    def processPosTagsAndLemmatize(self, word, pos):
        return self.lemma.lemmatize(word, self.treebankToWordnetPOS(pos))

    def removeNamedEntities(self, chunked, removeNumbers=True):
        def rec(t, r):
            if type(t) == Tree:
                if t.label() == 'NE':
                    r.append(('[NE]', 'NE'))
                else:
                    for child in t:
                        r.extend(rec(child, []))
            else:
                if removeNumbers:
                    r.append(('[CD]', 'CD')) if t[1] == "CD" else r.append(t)
                else:
                    r.append(t)
            return r
        return rec(chunked, [])

    def sentimentGrams(self, grams):
        return [{"LiuHu": self.sentimentLiuHu(gram), "Vader": self.sentimentVader(gram)} for gram in grams]

    def sentimentLiuHu(self, gram):
        posWord = lambda word: word in self.posWords
        negWord = lambda word: word in self.negWords
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

    def sentimentVader(self, gram):
        return self.vader.polarity_scores(" ".join(gram))

    def tokenSuffixes(self, tokens):
        return [max(s2, key=len) for s2 in
                [s1 for s1 in
                 [
                     [s0 for s0 in self.suffixes if word.endswith(s0)]
                     for word in tokens]
                 if s1]]

    def treebankToWordnetPOS(self, treebankPosTag):
        return {'J': wordnet.ADJ,
                'V': wordnet.VERB,
                'N': wordnet.NOUN,
                'R': wordnet.ADV}.get(treebankPosTag[0], wordnet.NOUN)

    def upperLowerLen(self, tokensOriginalCase):
        return [
            (
                sum([1 if letter.isupper() else 0 for letter in token]),
                sum([1 if letter.islower() else 0 for letter in token]),
                len(token),
                token
            )for token in tokensOriginalCase]

    def wordCase(self, upper, lower, length, token):
        if upper == 0:
            return "NC"  # No Caps
        elif upper == length:
            return "AC"  # All Caps
        elif upper == 1 and token[0].isupper():
            return "FC"  # First Cap
        else:
            return "MC"  # Mixed Caps

