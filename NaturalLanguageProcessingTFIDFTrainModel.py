# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:17:32 2020

@author: Santosh Sah
"""

from sklearn.naive_bayes import GaussianNB
from NaturalLanguageProcessingTFIDFUtils import (saveNaiveByesClassificationModelForTFIDF, readNaturalLanguageProcessingTFIDFXTrain, 
                                                      readNaturalLanguageProcessingTFIDFYTrain)

"""
Train NaiveByesClassification model 
"""
def trainNaiveByesClassificationModelForTFIDF():
    
    X_train = readNaturalLanguageProcessingTFIDFXTrain()
    y_train = readNaturalLanguageProcessingTFIDFYTrain()
    
    naiveByesClassification = GaussianNB()
    naiveByesClassification.fit(X_train, y_train)
    
    saveNaiveByesClassificationModelForTFIDF(naiveByesClassification)

if __name__ == "__main__":
    trainNaiveByesClassificationModelForTFIDF()