from sklearn.feature_extraction import DictVectorizer
from datetime import datetime
import pickle
import os
import json
import re

def createFeatVector(feats):
    dv = DictVectorizer()
    sparse = dv.fit_transform(feats)
    return (dv,sparse)

class ml:
    def __init__(self, classifiers={}, pickledir=''):
        self.classifiers = classifiers
        self.pickledir = pickledir
        if pickledir:
                for file in os.listdir(pickledir):
                    if file.endswith(".pickle"):
                            self.classifiers[file[:-6]] = pickle.load(open(pickledir + "/" + file, 'rb'))
    
    def save(self,pickledir='', prependStr=''):
        if not prependStr:
            prependStr = str(datetime.datetime.now())
        if not pickledir:
            pickledir = self.pickledir
        for name, classifier in self.classifiers.items():
            name = pickledir + "/" + re.sub(r'[ :<>".*,|\/]+', "", ' '.join([prependStr, name])) + '.pickle'
            pickle.dump(classifier ,open(name, 'wb'))
        
    def trainListTup(self, listTup):
        (listOfDictFeatures, listOfSarcasmBool) = zip(*listTup)
        return self.train(list(listOfDictFeatures), list(listOfSarcasmBool))
        
    def train(self, listOfDictFeatures, listOfSarcasmBool):
        if len(listOfDictFeatures) != len(listOfSarcasmBool):
            raise TypeError("Length of listOfDictOfFeatures and listOfSarcasmBool must match!")
        vectorizedFeatures = createFeatVector(listOfDictFeatures)[1]
        return self.trainVectorizedFeatures(vectorizedFeatures, listOfSarcasmBool)
    
    def trainVectorizedFeatures(self, vectorizedFeatures, listOfSarcasmBool):
        d = {}
        for name, classifier in self.classifiers.items():
            s = datetime.datetime.now()
            classifier.fit(vectorizedFeatures, listOfSarcasmBool)
            e = datetime.datetime.now()
            t=e-s
            d[name] = (classifier, t)
        return d

    def accuracySingle(self, dictOfFeatures, sarcasmBool):
        return self.accuracy([dictOfFeatures], [sarcasmBool])
    
    def accuracyListTup(self, listTup):
        (listOfDictFeatures, listOfSarcasmBool) = zip(*listTup)
        return self.accuracy(list(listOfDictFeatures), list(listOfSarcasmBool))
            
    def accuracy(self, listOfDictFeatures, listOfSarcasmBool):
        if len(listOfDictFeatures) != len(listOfSarcasmBool):
            raise TypeError("Length of listOfDictOfFeatures and listOfSarcasmBool must match!")
        vectorizedFeatures = createFeatVector(listOfDictFeatures)[1]
        return self.accuracyVectorizedFeatures(vectorizedFeatures, listOfSarcasmBool)

    def accuracyVectorizedFeatures(self, vectorizedFeatures, listOfSarcasmBool):
        scores = {}
        for name, classifier in self.classifiers.items():
            start = datetime.datetime.now()
            scores[name] = (classifier.score(vectorizedFeatures.toarray(), listOfSarcasmBool), (datetime.datetime.now()-start).total_seconds())
        return scores