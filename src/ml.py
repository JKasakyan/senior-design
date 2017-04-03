from sklearn.feature_extraction import DictVectorizer
from datetime import datetime
import pickle
import os
import json

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
                            self.classifiers[file[:-6]] = pickle.load(open(file, 'rb'))
    
    def save(self,pickledir='', prependStr=''):
        if not prependStr:
            prependStr = str(datetime.now())
        if not pickledir:
            pickledir = self.pickledir
        for name, classifier in self.classifiers:
            name = ' '.join([prependStr, name]) + '.pickle'
            pickle.dump = pickle.dump(classifier ,open(name, 'wb'))
            open(pd+"/index.txt", 'a').write('\t'.join(name, type(classifier), classifier.get_params()) + '\n')
        
    def trainListTup(self, listTup):
        (listOfDictFeatures, listOfSarcasmBool) = zip(*listTup)
        return self.train(list(listOfDictFeatures), list(listOfSarcasmBool))
        
    def train(self, listOfDictFeatures, listOfSarcasmBool):
        if len(listOfDictFeatures) != len(listOfSarcasmBool):
            raise TypeError("Length of listOfDictOfFeatures and listOfSarcasmBool must match!")
        vectorizedFeatures = createFeatVector(listOfDictFeatures)[1]
        return self.trainVectorizedFeatures(vectorizedFeatures, listOfSarcasmBool)
    
    def trainVectorizedFeatures(self, vectorizedFeatures, listOfSarcasmBool):
        for classifier in self.classifiers.values():
            classifier.fit(vectorizedFeatures.toarray(), listOfSarcasmBool)
        return self.classifiers

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
            scores[name] = classifier.score(vectorizedFeatures.toarray(), listOfSarcasmBool)
        return scores