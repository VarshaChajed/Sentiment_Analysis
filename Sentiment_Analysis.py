#!/usr/bin/env python
# coding: utf-8

# Author: Varsha Chajed


import nltk
nltk.download()

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json

twitter_samples.fileids()

def extract_words(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


pos_tweets = twitter_samples.strings('positive_tweets.json')
print (len(pos_tweets)) # Output: 5000
 
neg_tweets = twitter_samples.strings('negative_tweets.json')
print (len(neg_tweets)) # Output: 5000
 
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
print (len(all_tweets)) # Output: 20000
 
for tweet in pos_tweets[:5]:
    print (tweet)

neg_tweet = []
for line in open('/Data/negative_tweets.json', 'r'):
    tweet_json = json.loads(line)
    tweet_text = (tweet_json["text"])
    tweet_text = tweet_text.replace(":", "").replace(")", "").replace("(", "")
    print (tweet_text)
    neg_tokens = word_tokenize(tweet_text)
    neg_tweet.append((extract_words(neg_tokens), "negative"))
print (len(neg_tweet))

pos_tweet = []
for line in open('/Data/positive_tweets.json', 'r'):
    tweet_json1 = json.loads(line)
    tweet_text1 = (tweet_json1["text"])
    tweet_text1 = tweet_text1.replace(":", "").replace(")", "").replace("(", "")
    print (tweet_text1)
    pos_tokens = word_tokenize(tweet_text1)
    pos_tweet.append((extract_words(pos_tokens), "positive"))
print (len(pos_tweet))
print (pos_tweet[0])

train_data = neg_tweet[:4000] + pos_tweet[:4000]
test_data =  neg_tweet[4000:] + pos_tweet[4000:]
print(len(train_data),  len(test_data))

classifier = NaiveBayesClassifier.train(train_data)

accuracy = nltk.classify.util.accuracy(classifier, test_data)
print(accuracy * 100)

review = 'Sentiment Analysis is cool'
words_text = word_tokenize(review)
words_text = extract_words(words_text)
classifier.classify(words_text)



