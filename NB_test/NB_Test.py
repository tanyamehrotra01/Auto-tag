
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random



tweetsF = open("clean-strings-0.pickle" , 'rb')
hashtagsF = open("extracted-hashtags-0.pickle", 'rb')

wordsAllInOne = {}
hashtagAllInOne = {}
dct = {}

def MakeDictionary(tweetsF, hashtagsF):
	tweets = pickle.load(tweetsF)[:10]
	hashtags = pickle.load(hashtagsF)[:10]
	global wordsAllInOne
	global hashtagAllInOne
	wordsAllInOne = set([word for tweet in tweets for word in word_tokenize(tweet)])
	hashtagAllInOne = set([hashtag for hashtagList in hashtags for hashtag in hashtagList])
	
	dct = {hashtag:[] for hashtag in hashtagAllInOne}
	for hashtag in hashtagAllInOne:
		indexes = [i for i in range (len(hashtags)) if hashtag in hashtags[i]]
		for index in indexes:
			filtered_sentence = []
			words = word_tokenize(tweets[index])
			for w in words:
				filtered_sentence.append(w)
			dct[hashtag] += filtered_sentence
	return dct 

dct = MakeDictionary(tweetsF,hashtagsF)
print(dct)

with open('filename.pickle', 'wb') as handle:
    pickle.dump(dct , handle)

dctF = open("filename.pickle", 'rb')
dct = pickle.load(dctF)

documents = [
				(wordList, hashtag) 
				for hashtag in hashtagAllInOne 
				for wordList in dct[hashtag]
			]

 
random.shuffle(documents)

all_words = []

for w in wordsAllInOne:
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())

def findFeatures(document):
	wordsD = set(document)
	features = {w: (w in wordsD) for w in word_features}
	return features

featureSet = [(findFeatures(rev), category) for (rev, category) in documents]

trainingSet = featureSet
testingSet = featureSet

classifier = nltk.NaiveBayesClassifier.train(trainingSet)
print ("NaiveBayesClassifier accuracy percent: ", (nltk.classify.accuracy(classifier, testingSet)) * 100)
classifier.show_most_informative_features(15)

#Testing
def NB_Test(cleanedString):
	doc = word_tokenize(cleanedString.lower())
	featurized_doc = {i:(i in doc) for i in word_features}
	tagged_label = classifier.classify(featurized_doc)
	return tagged_label

hashtagObtained = NB_Test("big challenge to vote this election ge15")
print(hashtagObtained)







