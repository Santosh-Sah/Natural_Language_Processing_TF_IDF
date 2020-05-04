# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:47:27 2020

@author: Santosh Sah
"""

from NaturalLanguageProcessingTFIDFUtils import (readNaturalLanguageProcessingTFIDFXTest, readNaiveByesClassificationModelForTFIDF,
                                     saveNaturalLanguageProcessingTFIDFYPred)

"""
test the model on testing dataset
"""
def testNaiveByesClassificationModel():
    
    X_test = readNaturalLanguageProcessingTFIDFXTest()
    
    naiveByesClassificationModel = readNaiveByesClassificationModelForTFIDF()
    
    y_pred = naiveByesClassificationModel.predict(X_test)
    
    saveNaturalLanguageProcessingTFIDFYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testNaiveByesClassificationModel()