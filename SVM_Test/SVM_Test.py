import nltk
import random
import pickle
from nltk import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import validation_curve

tweetsF = open("clean-strings-0.pickle" , 'rb')
hashtagsF = open("extracted-hashtags-0.pickle", 'rb')

wordsAllInOne = {}
hashtagAllInOne = {}
dct = {}

def MakeDictionary(tweetsF, hashtagsF):
	tweets = pickle.load(tweetsF)[:200]
	hashtags = pickle.load(hashtagsF)[:200]
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

trainingSet = featureSet[:180]
testingSet = featureSet[180:]

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(trainingSet)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testingSet))*100)

featurized_doc = []
def SVM_Test(cleanedString):
	doc = word_tokenize(cleanedString.lower())
	global featurized_doc
	featurized_doc = {i:(i in doc) for i in word_features}
	tagged_label = LinearSVC_classifier.classify(featurized_doc)
	return tagged_label

hashtagObtained = SVM_Test("big challenge to vote this election ge15")
print(hashtagObtained)



