# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:27:01 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from NaturalLanguageProcessingTFIDFUtils import (readNaturalLanguageProcessingTFIDFYTest, readNaturalLanguageProcessingTFIDFYPred)

"""

calculating NaturalLanguageProcessingTFIDF confussion matrix

"""
def testNaturalLanguageProcessingTFIDFConfussionMatrix():
    
    y_test = readNaturalLanguageProcessingTFIDFYTest()
    y_pred = readNaturalLanguageProcessingTFIDFYPred()
    
    naturalLanguageProcessingTFIDFConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(naturalLanguageProcessingTFIDFConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[57 40]
    [16 87]]
    
    """
"""
calculating accuracy score

"""

def testNaturalLanguageProcessingTFIDFAccuracy():
    
    y_test = readNaturalLanguageProcessingTFIDFYTest()
    y_pred = readNaturalLanguageProcessingTFIDFYPred()
    
    naturalLanguageProcessingTFIDFAccuracy = accuracy_score(y_test, y_pred)
    
    print(naturalLanguageProcessingTFIDFAccuracy) #.72%

"""
calculating classification report

"""

def testNaturalLanguageProcessingTFIDFClassificationReport():
    
    y_test = readNaturalLanguageProcessingTFIDFYTest()
    y_pred = readNaturalLanguageProcessingTFIDFYPred()
    
    naturalLanguageProcessingTFIDFClassificationReport = classification_report(y_test, y_pred)
    
    print(naturalLanguageProcessingTFIDFClassificationReport)
    
    """
             precision    recall  f1-score   support

          0       0.78      0.59      0.67        97
          1       0.69      0.84      0.76       103

avg / total       0.73      0.72      0.71       200
    """
    
if __name__ == "__main__":
    #testNaturalLanguageProcessingTFIDFConfussionMatrix()
    #testNaturalLanguageProcessingTFIDFAccuracy()
    testNaturalLanguageProcessingTFIDFClassificationReport()