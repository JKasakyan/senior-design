from nltk import ngrams, word_tokenize, pos_tag, FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import zip_longest
from functools import reduce

class caps(enum):
	NoCaps = 0
	FirstCap = 1
	MixedCap = 2
	AllCap = 3

class nlp:
	defaultSuffixes =  list(line.rstrip() for line in open("suffixes.txt"))
	defaultSuffixTable = dict([suffix, 0] for suffix in defaultSuffixes)
	lemma = WordNetLemmatizer()
	
	def treebankToWordnetPOS(treebankPosTag)
	return  {'J':wordnet.ADJ,
				'V':wordnet.VERB,
				'N':wordnet.NOUN,
				'R':wordnet.ADV}.get(treebankPosTag[0], wordnet.NOUN)
				
	tokenize = lambda text: (word_tokenize(text.lower()), word_tokenize(text))
	
	pos = lambda tokens: pos_tag(tokens)
	
	posTagOnly = lambda tagged: [tag for word,tag in tagged]
	
	processPosTagsAndLemmatize = lambda (word, pos): lemma.lemmatize(word, treebankToWordnetPOS(pos))
	
	lemmatize = lambda  posTokens: [processPosTagsAndLemmatize(wordPos) for wordPos in tagged]
	
	ngram = lambda tokens, n: ngrams(tokens, n)
	
	ngramFreq = lambda grams: FreqDist(grams)
	
	def longestSuffixAWordEndsWith(tokens):
		return[max(s2, key=len) for s2 in 
					[s1 for s1 in 
						[
							[s0 for s0 in defaultSuffixes if word.endswith(s0)]
						for word in tokens]
					if s1]]
					
	def upperLowerLen(tokensOriginalCase):
		return[
					(
					sum(1 if letter.isupper() else 0 for letter in token]),
					sum([1 if letter.islower() else 0 for letter in token]),
					len(token),
					token
					)
				for token in tokensOriginalCase]
				
	capLetterFreq = lambda (upper, lower, length, token): reduce(lambda i, u: i +u , upper)/reduce(lambda i, l: i + l , lower)
	
	def wordCase((upper, lower, length, token)):
		if upper == 0:
			return caps.NoCaps
		elif upper == length:
			return caps.AllCap
		elif upper == 1 and token[0].isupper():
			return caps.FirstCap
		else:
			return caps.MixedCap 
	
		