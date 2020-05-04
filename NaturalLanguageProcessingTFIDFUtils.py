# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:01:34 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importNaturalLanguageProcessingTFIDFDataset(naturalLanguageProcessingTFIDFDatasetFileName):
    
    naturalLanguageProcessingTFIDFDataset = pd.read_csv(naturalLanguageProcessingTFIDFDatasetFileName, delimiter = '\t', quoting = 3)
    
    return naturalLanguageProcessingTFIDFDataset

def cleanNaturalLanguageProcessingBagOfWordsDataset(naturalLanguageProcessingTFIDFDataset, nltkStopwords):
    
    bagOfWordsCorpus = []
    
    for i in range(0, 1000):
        
        bagOfWordsReview = re.sub('[^a-zA-Z]', ' ', naturalLanguageProcessingTFIDFDataset['Review'][i])
        
        bagOfWordsReview = bagOfWordsReview.lower()
        
        bagOfWordsReview = bagOfWordsReview.split()
        
        bagOfWordsPorterStemmer = PorterStemmer()
        
        bagOfWordsReview = [bagOfWordsPorterStemmer.stem(word) for word in bagOfWordsReview if not word in set(nltkStopwords.words('english'))]
        
        bagOfWordsReview = ' '.join(bagOfWordsReview)
        
        bagOfWordsCorpus.append(bagOfWordsReview)
    
    return bagOfWordsCorpus


def createNaturalLanguageProcessingBagOfWordsModel(naturalLanguageProcessingTFIDFDataset, bagOfWordsCorpus):
    
    bagOfWordsCountVectorizer = CountVectorizer(max_features = 1500)
    bagOfWordsCountVectorizer.fit(bagOfWordsCorpus)
    
    X = bagOfWordsCountVectorizer.transform(bagOfWordsCorpus).toarray()
    y = naturalLanguageProcessingTFIDFDataset.iloc[:, 1].values
    
    return bagOfWordsCountVectorizer, X, y

def createNaturalLanguageProcessingTFIDFModel(X):
    
    tfidfTransformer = TfidfTransformer()
    tfidfTransformer.fit(X)
    
    X = tfidfTransformer.transform(X).toarray()
    
    return tfidfTransformer, X

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save NaiveByesClassificationModelForTFIDF as a pickle file.
"""
def saveNaiveByesClassificationModelForTFIDF(naiveByesClassificationModelForTFIDF):
    
    #Write NaiveByesClassificationModelForTFIDF as a picke file
    with open("NaiveByesClassificationModelForTFIDF.pkl",'wb') as NaiveByesClassificationModelForTFIDF_Pickle:
        pickle.dump(naiveByesClassificationModelForTFIDF, NaiveByesClassificationModelForTFIDF_Pickle, protocol = 2)

"""
read NaiveByesClassificationModelForTFIDF from pickle file
"""
def readNaiveByesClassificationModelForTFIDF():
    
    #load NaiveByesClassificationModelForTFIDF model
    with open("NaiveByesClassificationModelForTFIDF.pkl","rb") as NaiveByesClassificationModelForTFIDF:
        naiveByesClassificationModelForTFIDF = pickle.load(NaiveByesClassificationModelForTFIDF)
    
    return naiveByesClassificationModelForTFIDF

"""
read X_train from pickle file
"""
def readNaturalLanguageProcessingTFIDFXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readNaturalLanguageProcessingTFIDFXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readNaturalLanguageProcessingTFIDFYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readNaturalLanguageProcessingTFIDFYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveNaturalLanguageProcessingTFIDFYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readNaturalLanguageProcessingTFIDFYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred

"""
save bagOfWordsCountVectorizer as a pickle file
"""

def saveNaturalLanguageProcessingBagOfWordsCountVectorizer(countVectorizer):
    
    #Write CountVectorizer in a picke file
    with open("CountVectorizer.pkl",'wb') as countVectorizer_Pickle:
        pickle.dump(countVectorizer, countVectorizer_Pickle, protocol = 2)

"""
read bagOfWordsCountVectorizer from pickle file
"""
def readNaturalLanguageProcessingBagOfWordsCountVectorizer():
    
    #load bagOfWordsCountVectorizer
    with open("CountVectorizer.pkl","rb") as CountVectorizer_pickle:
        countVectorizer = pickle.load(CountVectorizer_pickle)
    
    return countVectorizer

"""
save TfidfTransformer as a pickle file
"""

def saveNaturalLanguageProcessingTfidfTransformer(tfidfTransformer):
    
    #Write TfidfTransformer in a picke file
    with open("TfidfTransformer.pkl",'wb') as tfidfTransformer_Pickle:
        pickle.dump(tfidfTransformer, tfidfTransformer_Pickle, protocol = 2)

"""
read TfidfTransformer from pickle file
"""
def readNaturalLanguageProcessingTfidfTransformer():
    
    #load TfidfTransformer
    with open("TfidfTransformer.pkl","rb") as TfidfTransformer_pickle:
        tfidfTransformer = pickle.load(TfidfTransformer_pickle)
    
    return tfidfTransformer