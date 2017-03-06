from nltk import ngrams, word_tokenize, pos_tag, FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import zip_longest, filterfalse
from functools import reduce
from enum import Enum

class caps(Enum):
    NoCaps = 0
    FirstCap = 1
    MixedCap = 2
    AllCap = 3

class nlp:
    defaultSuffixes =  list(line.rstrip() for line in open("suffixes.txt"))
    defaultSuffixTable = dict([suffix, 0] for suffix in defaultSuffixes)
    lemma = WordNetLemmatizer()
    
    def treebankToWordnetPOS(self, treebankPosTag):
        return{'J':wordnet.ADJ,
               'V':wordnet.VERB,
               'N':wordnet.NOUN,
               'R':wordnet.ADV}.get(treebankPosTag[0], wordnet.NOUN)
                
    tokenize = lambda self, text: (word_tokenize(text.lower()), word_tokenize(text))
    
    pos = lambda self, tokens: pos_tag(tokens)
    
    posTagOnly = lambda self, tagged: [tag for word,tag in tagged]
    
    def processPosTagsAndLemmatize(self, word, pos):
        return self.lemma.lemmatize(word, self.treebankToWordnetPOS(pos))
    
    lemmatize = lambda  self, posTokens: [self.processPosTagsAndLemmatize(*wordPos) for wordPos in tagged]
    
    ngram = lambda self, tokens, n: ngrams(tokens, n)
    
    ngramFreq = lambda self, grams: FreqDist(grams)
    
    def longestSuffixAWordEndsWith(self, tokens):
        return[max(s2, key=len) for s2 in 
                    [s1 for s1 in 
                        [
                            [s0 for s0 in self.defaultSuffixes if word.endswith(s0)]
                        for word in tokens]
                    if s1]]
                    
    def upperLowerLen(self, tokens):
        return[
                    (
                    sum([1 if letter.isupper() else 0 for letter in token]),
                    sum([1 if letter.islower() else 0 for letter in token]),
                    len(token),
                    token
                    )
                for token in tokensOriginalCase]
                
    def capLetterFreq(self, ull):
        return reduce(lambda i, u: i +u[0] , ull, 0)/reduce(lambda i, l: i + l[1] , ull, 0)
    
    def wordCase(self, upper, lower, length, token):
        if upper == 0:
            return caps.NoCaps
        elif upper == length:
            return caps.AllCap
        elif upper == 1 and token[0].isupper():
            return caps.FirstCap
        else:
            return caps.MixedCap 