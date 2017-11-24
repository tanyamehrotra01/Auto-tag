from __future__ import division, print_function, absolute_import, unicode_literals

import nltk
import math
import re
import csv
import pandas as pd
import pickle
import sys
import string
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
from textblob import TextBlob as tb
from nltk.classify.scikitlearn import SklearnClassifier


abbreviationsDict = {}
WORDS = None
def CleanStringAndExtractHashtags(string):
	print("CleanStringAndExtractHashtags: Starting. String received:", string)

	def WordTokenizeString(string):
		return word_tokenize(string)

	def TweetTokenizeString(string):
		tweetTokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)
		return " ".join(tweetTokenizer.tokenize(string))

	def RemoveApostropheWords(wordList):
		apostropheWords = {
							"n't":"not",
						  	"'s": ["is", "us"],
						  	"'ll": "will",
						  	"'ve": "have",
						  	"'re": "are",
						  	"'d": "would",
						  	"'m": "am",
						  	"c'mon": ["come", "on"],
						  	"y'all": ["you", "all"]
						  }

		removedApostropheWords = []

		previousWord = None
		for word in wordList:
			if (word in apostropheWords):
				if (word == "'s"):
					index = int(previousWord != "it")
					removedApostropheWords.append(apostropheWords[word][index])
				elif (len(apostropheWords[word]) > 1):
					removedApostropheWords.extend(apostropheWords[word])
				else:
					removedApostropheWords.append(apostropheWords[word])
			else:
				removedApostropheWords.append(word)
			previousWord = word

		return removedApostropheWords

	def LemmatizeWords(wordList):
		POSTaggedWordList = pos_tag(wordList)

		lemmatizedWords = []

		wordNetLemmatizer = WordNetLemmatizer()

		for word, partOfSpeech in POSTaggedWordList:

			try:
				lemmatizedWord = wordnet.morphy(word, pos=partOfSpeech[0].lower())
			except:
				lemmatizedWord = wordnet.morphy(word)
						
			if (lemmatizedWord == None):
				try:
					lemmatizedWord = wordNetLemmatizer.lemmatize(word, pos=partOfSpeech[0].lower())
				except:
					lemmatizedWord = wordNetLemmatizer.lemmatize(word)

			lemmatizedWords.append(lemmatizedWord)

		return lemmatizedWords

	# http://norvig.com/spell-correct.html
	def CorrectWords(wordList):
		
		global WORDS

		def words(text):
			return re.findall(r'\w+', text.lower())

		if(WORDS == None):
			WORDS = Counter(words(open('Glove_BagOfWords.txt').read()))

		def P(word, N=sum(WORDS.values())): 
			"Probability of `word`."
			return WORDS[word] / N

		def correction(word):
			"Most probable spelling correction for word."
			return max(candidates(word), key=P)

		def candidates(word): 
			"Generate possible spelling corrections for word."
			return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

		def known(words): 
			"The subset of `words` that appear in the dictionary of WORDS."
			return set(w for w in words if w in WORDS)

		def edits1(word):
			"All edits that are one edit away from `word`."
			letters    = 'abcdefghijklmnopqrstuvwxyz'
			splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
			deletes    = [L + R[1:]               for L, R in splits if R]
			transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
			replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
			inserts    = [L + c + R               for L, R in splits for c in letters]
			return set(deletes + transposes + replaces + inserts)

		def edits2(word): 
			"All edits that are two edits away from `word`."
			return (e2 for e1 in edits1(word) for e2 in edits1(e1))

		correctedWords = []

		for word in wordList:
			correctedWord = correction(word)
			if (len(correctedWord) > 0):
				correctedWords.append(correctedWord)
			else:
				correctedWords.append(word)

		return correctedWords

	def RemoveStopwords(wordList):
		stopWords = set(stopwords.words("english"))
		return [word for word in wordList if word not in stopWords]

	def RemoveURLs(string):
		result = re.sub(r"http\S+", "", string)
		return result

	def ReplaceAbbreviatedWords(wordList):
		global abbreviationsDict

		if (len(abbreviationsDict) == 0):
			with open('abbreviations.csv', mode='r') as infile:
				reader = csv.reader(infile)
				for rows in reader:
					rows[0] = rows[0].split("~")
					abbreviationsDict[rows[0][0].lower()] = rows[0][1].lower()

		newWordList = []

		for word in wordList:
			if word in abbreviationsDict:
				newWordList.extend(word_tokenize(abbreviationsDict[word]))
			else:
				newWordList.append(word)

		return newWordList

	def RemovePunctuations(wordList):
		string = " ".join(wordList)
		string = "".join(character for character in string.strip() if character not in punctuation)

		return word_tokenize(string)

	lowercaseString = string.lower()
	print("\tConverting string to lower case:", lowercaseString)

	removedURLString = RemoveURLs(lowercaseString)
	print("\tAfter removing URLs:", removedURLString)

	tweetTokenizedString = TweetTokenizeString(removedURLString)
	print("\tAfter tweet tokenizing:", tweetTokenizedString)

	hashtags = [word for word in tweetTokenizedString.split(' ') if '#' in word]
	print("\tExtracting hashtags:", hashtags)

	wordTokenizedString = WordTokenizeString(tweetTokenizedString)
	print("\tAfter word tokenizing:", wordTokenizedString)
	
	removedApostropheWords = RemoveApostropheWords(wordTokenizedString)
	print("\tAfter removing apostrophe words:", removedApostropheWords)

	replacedAbbreviatedWords = ReplaceAbbreviatedWords(removedApostropheWords)
	print("\tAfter replacing abbreviations:", replacedAbbreviatedWords)

	removedPunctuations = RemovePunctuations(replacedAbbreviatedWords)
	print("\tAfter removing punctuations:", removedPunctuations)

	lemmatizedWords = LemmatizeWords(removedPunctuations)
	print("\tAfter lemmatizing words:", lemmatizedWords)

	correctedWords = CorrectWords(lemmatizedWords)
	print("\tAfter correcting words to their correct spelling:", correctedWords)

	cleanedString = " ".join(correctedWords)
	print("\tCleaned string:", cleanedString)

	return cleanedString, hashtags

string = input("Enter string:")
cleanedString, extractedHashtag = CleanStringAndExtractHashtags(string)

print("Cleaned String:", cleanedString)
print()
print("Extracted Hashtag:", extractedHashtag)
print()