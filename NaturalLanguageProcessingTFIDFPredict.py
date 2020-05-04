# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:15:53 2020

@author: Santosh Sah
"""
import re
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from NaturalLanguageProcessingTFIDFUtils import (readNaiveByesClassificationModelForTFIDF, readNaturalLanguageProcessingBagOfWordsCountVectorizer,
                                                 readNaturalLanguageProcessingTfidfTransformer)

def predict():
    
    naiveByesClassificationModelForTFIDF = readNaiveByesClassificationModelForTFIDF()
    
    naturalLanguageProcessingBagOfWordsCountVectorizer = readNaturalLanguageProcessingBagOfWordsCountVectorizer()
    
    naturalLanguageProcessingTfidfTransformer = readNaturalLanguageProcessingTfidfTransformer()
    
    #bagOfWordsReview = "This is a good restaurant. Food is best in the city"
    bagOfWordsReview = "This is a bad restaurant. Food is bad in the city"
    
    bagOfWordsReview = re.sub('[^a-zA-Z]', ' ', bagOfWordsReview)
        
    bagOfWordsReview = bagOfWordsReview.lower()
        
    bagOfWordsReview = bagOfWordsReview.split()
        
    bagOfWordsPorterStemmer = PorterStemmer()
        
    bagOfWordsReview = [bagOfWordsPorterStemmer.stem(word) for word in bagOfWordsReview if not word in set(stopwords.words('english'))]
        
    bagOfWordsReview = ' '.join(bagOfWordsReview)
    
    bagOfWordsCorpus = []
    
    bagOfWordsCorpus.append(bagOfWordsReview)
    
    newObservation = naturalLanguageProcessingBagOfWordsCountVectorizer.transform(bagOfWordsCorpus).toarray()
    
    newObservation = naturalLanguageProcessingTfidfTransformer.transform(newObservation).toarray()
        
    predictedValue = naiveByesClassificationModelForTFIDF.predict(newObservation)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()