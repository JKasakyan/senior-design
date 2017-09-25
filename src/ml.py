import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import islice, tee, chain
from os import listdir, remove
from random import shuffle
from collections import defaultdict

import numpy as np
import scipy as sp
import json_io
from nlp import *
from dvs import DictVectorizerPartial
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit

DEFAULT_CLASSIFIERS = [
    LogisticRegression(n_jobs=-1)
    # LogisticRegression(solver='sag', max_iter=1000, n_jobs=-1, warm_start=True),
    # SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1),
    # BernoulliNB(alpha=0.2, binarize=0.4),
    # MultinomialNB(alpha=0),
]
DEFAULT_CLASSIFIERS_ARGS = [
    # (SGDClassifier(penalty='elasticnet', n_jobs=-1), {'loss':['log','modified_huber','perceptron'], 'penalty':['none','l1','elasticnet','l2']}),
    # (BernoulliNB(),{'alpha':list(np.arange(0,20,0.1)), 'binarize':list(np.arange(0.1,0.9,0.1))}),
    # (MultinomialNB(),{'alpha':list(np.arange(0,20,0.1))})
]
FILENAME_REGEX = r'[ :<>".*,|\/]+'
PICKLED_FEATS_DIR = 'pickledfeatures/'
JSON_DIR = '../json/'
BEST_TRAINED_CLASSIFIERS = './pickled/best.pickle'

def trainTest(X, y, classifiers=DEFAULT_CLASSIFIERS, reduce=0, splits=10, trainsize=0.8, testsize=0.2):
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=testsize, train_size=trainsize)
    results = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        if reduce > 0:
            print("Samples, Features before reduction: " + str(X_train.shape))
            reducer = SelectKBest(score_func=f_classif, k=reduce)
            X_train = reducer.fit_transform(X_train, y_train)
            X_test = reducer.transform(X_test)
            print("Samples, Features after reduction: " + str(str(X_train.shape)))
            support = reducer.get_support()

        for classifier in classifiers:
            print("Starting to train %s"%str(type(classifier)))
            s = datetime.now()
            classifier.fit(X_train, y_train)
            traintime = (datetime.now() - s).total_seconds()
            score = classifier.score(X_test, y_test)
            if reduce > 0:
                results.append((classifier, traintime, score, support))
            else:
                results.append((classifier, traintime, score))
            print("%s\tTime: %d\tScore:\t%f" %(str(type(classifier)), traintime, score))
    return results

def trainTestMultiple(X, y, reduce_amounts, train_sizes, classifiers=DEFAULT_CLASSIFIERS, splits=10, save=False, results_fn=None):
    """
    Train and report/save results on multiple combinations of reduction amounts, train sizes, classifiers, and splits.

    X: scipy.sparse.csr_matrix, shape M x N
    y: numpy.ndarray, shape M
    reduce_amounts: list-like, restrict to top k features when predicting
    train_sizes: list-like, determines the train-test split. ex: [0.8, 0.6] will perform a 80-20 and 60-40 train test split
    classifiers: list-like, sklearn Classifiers
    splits: integer: (Optional, default=10) number of times to cross-validate
    save: bool: (Optional, default=False) save results as pickle
    results_fn: (Optional, default=None) path to save results pickle
    """

    results = []
    for reduceamount in reduce_amounts:
        print("\n\t\tReduction: "+str(reduceamount))
        for trainsize in train_sizes:
            print("\n\t\tTraining size: "+str(trainsize))
            results.append((reduceamount,
                           trainsize,
                           trainTest(X,
                                    y,
                                    classifiers=[clone(c) for c in classifiers],
                                    reduce=reduceamount,
                                    splits=splits,
                                    trainsize=trainsize,
                                    testsize=1-trainsize)))
    if save:
        assert results_fn != None, "Provide path to save pickle"
        pickle.dump(results, open(results_fn, 'wb'))

    return results

def flattenDict(feature):
    d = {}
    for key, value in feature.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                d[subkey] = subvalue
        else:
            d[key] = value
    return d

def flatten(X,y=None):
    if y:
        return (flattenDict(x) for x in X), y
    else:
        return (flattenDict(x) for x in X)

def saveVectorizer(dv, X=None, y=None, extra=''):
    if X is not None and y is not None:
        pickle.dump((dv, X, y), open(PICKLED_FEATS_DIR + 'Xydv' + extra  + '.pickle', 'wb'))
        pickle.dump(dv, open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'wb'))
    else:
        pickle.dump(dv, open(PICKLED_FEATS_DIR + 'dv' + extra  + '.pickle', 'wb'))

def split_feat(gen, n):
    def create_generator(it, n):
        return (item[n] for item in it)
    G = tee(gen, n)
    return [create_generator(g, n) for n, g in enumerate(G)]

def predict(listOfString, classifier, dvp, cleanTokens, support=None):
    listOfFeats = np.array([flattenDict(feature(s, cleanTokens)) for s in listOfString])
    if support is not None:
        dvp = deepcopy(dvp).restrict(support)
    X = dvp.transform(listOfFeats)
    prediction = classifier.predict(X)
    invert_op = getattr(classifier, "predict_proba", None)
    if callable(invert_op):
        preProb = classifier.predict_proba(X)
        return {'classifier':classifier, 'prediction': prediction, 'prediction_probabilities':preProb}
    else:
        return {'classifier':classifier, 'prediction': prediction}
    print(r)
    return r

def process_X_y_dvp_from_json(sarcastic_path, serious_path, source='-twitter', save=False, X_fn=None, y_fn=None, dvp_fn=None):
    """
    Process features from tweets saved as jsons at sarcastic_path and serious_path
    and fit transform a DictVectorizerPartial to data

    sarcastic_path - path to sarcastic tweets json
    serious_path - path to serious tweets json
    save (optional) - save X, y and dvp as pickles
    X_fn - (optional) full path to save X pickle
    y_fn - (optional) full path to save y pickle
    dvp_fn - (optional) full path to save dvp pickle

    returns X, y, dvp
    """

    json_io.processRandomizeJson(sarcastic=True,
                     json_path=sarcastic_path,
                     features_path="./",
                     source=source,
                     n=1,
                     cleanTokens=cleanTokensTwitter)
    json_io.processRandomizeJson(sarcastic=False,
                     json_path=serious_path,
                     features_path="./",
                     source=source,
                     n=1,
                     cleanTokens=cleanTokensTwitter)
    sarcasticFeats = json_io.loadProcessedFeatures("./",
                                       source,
                                       sarcastic=True,
                                       n=1,
                                       random=False)
    seriousFeats = json_io.loadProcessedFeatures("./",
                                         source,
                                         sarcastic=False,
                                         n=1,
                                         random=False)
    features = chain(sarcasticFeats, seriousFeats)
    dvp = DictVectorizerPartial()
    (X,y) = split_feat(features, 2)
    (X,y) = flatten(X,y)
    (X,y) = (dvp.partial_fit_transform(X), np.array(list(y)))

    if save:
        assert X_fn != None and y_fn != None and dvp_fn != None, "Provide paths for X, y, dvp"
        try:
            for payload, fn in zip([X, y, dvp], [X_fn, y_fn, dvp_fn]):
                pickle.dump(payload, open(fn, 'wb'))
            remove(json_io.fileName("./", source, True, i=0))
            remove(json_io.fileName("./", source, False, i=0))
        except:
            raise

    return X, y, dvp

def even_samples(X, y, n):
    """
    Create a balanced set where n random positive samples and n random negative samples are taken from X and y. y must be binary (two class)

    X: scipy.sparse.csr_matrix, shape M x N
    y: numpy.ndarray, shape M
    n: integer, number of samples to take from each class

    returns X_bal (shape 2n * N), y_bal (shape 2n) with n samples of positive class and n samples of negative class
    """

    assert n <= len(y[y==True]), "Cannot create balanced sample with {} of each class when class positive only has {} samples".format(n, len(y[y==True]))

    sarc_start_indx = 0
    sarc_end_indx = len(y[y==True] - 1)
    ser_start_indx = sarc_end_indx + 1
    ser_end_indx = len(y) - 1
    # Select sarcastic samples
    try:
        start_indx = np.random.choice(np.arange(sarc_start_indx, sarc_end_indx - n))
    except ValueError:
        # using all samples
        start_indx = 0
    new_y = list(y[start_indx:start_indx + n])
    new_X = sp.sparse.csr_matrix(X[start_indx:start_indx + n, :])

    # Append serious samples
    start_indx = np.random.choice(np.arange(ser_start_indx, ser_end_indx - n))
    new_y = new_y + list(y[start_indx:start_indx + n])
    new_X = sp.sparse.vstack([new_X, X[start_indx:start_indx + n, :]])

    return new_X, np.array(new_y)

def best_classifiers(results, names, dvp, save=False, fn=None):
    """
    Determine the highest scoring classifiers from results output by ml.trainTestMultiple

    results: list-like (shape M), list of results in format output by ml.trainTestMultiple
    names: list-like (shape M), names to reference each result by
    dvp: DictVectorizerPartial, the dvp the full dataset was fit to
    save: boolean, (Optional, default=False) save the results to path specified by fn
    fn: string, the path to save results to
    """

    # define helper function
    def find_best_classifier(results):
        highest_acc = 0.0
        best_results = None
        for r in results:
            classifiers = r[2]
            for c in classifiers:
                if c[2] > highest_acc:
                    highest_acc = c[2]
                    best_results = r
                    best_classifier = c

        reduction = best_results[0]
        size = best_results[1]
        classifier = best_classifier[0]
        score = best_classifier[2]
        train_time = best_classifier[1]
        if len(best_classifier) == 3:
            return classifier, score, reduction, size, train_time
        else:
            support = best_classifier[3]
            return classifier, score, reduction, size, train_time, support

    payload = defaultdict(dict)
    for result, name in zip(results, names):
        best_results = find_best_classifier(result)
        if len(best_results) == 5:
            best_classifier, score, reduction, size, train_time = best_results
            payload[name]['dvp'] = deepcopy(dvp)
        else:
            best_classifier, score, reduction, size, train_time, support = best_results
            payload[name]['dvp'] = deepcopy(dvp).restrict(support)

        payload[name]['classifier'] = best_classifier
        payload[name]['score'] = score
        payload[name]['reduction'] = reduction
        payload[name]['size'] = size
        payload[name]['train_time'] = train_time

    if save:
        assert fn != None, "Provide path to save pickle"
        pickle.dump(payload, open(fn, 'wb'))
    return payload

def top_n_features(classifier, dvp, classBool, posClassLabel, negClassLabel, n=10, support=None):
    """
    Determine the top n most informative features of a binary classifier

    classifier: sklearn classifier
    dvp: DictVectorizerPartial the data was fit-transformed on
    classBool: boolean, top k features of which class (True=positive, False=negative)
    posClassLabel: string, name of positive class
    negClassLabel: string, name of negative class
    """

    if support is not None:
        dvp = deepcopy(dvp).restrict(support)
    feature_names = dvp.get_feature_names()

    if (classBool):
        topn = sorted(zip(classifier.coef_[0], feature_names))[:-n-1:-1]
    else:
        topn = sorted(zip(classifier.coef_[0], feature_names))[:n]

    return [(posClassLabel if classBool else negClassLabel, feat, coef) for coef, feat in topn]
    for coef, feat in topn:
        print (posClassLabel if classBool else negClassLabel, feat, coef)

def predictMultiple(tweets, classifiers, names, dvps, y, graph=False):
    """
    Test multiple classifiers on a set of tweets and return results

    tweets: list-like (shape N), text of tweets to test on
    classifiers: list-like (shape M), sklearn classifiers to test
    names: list-like (shape M), label to give to each classifier in results dictionary
    dvps: list-like (shape M), DictVectorizer used to fit features for each classifier
    y: list-like (shape N), boolean representing actual classification label for each tweet
    graph: boolean,
    """
    results = {}
    for classifier, name, dvp in zip(classifiers, names, dvps):
        results[name] = {}
        pre = predict(tweets, classifier, dvp, cleanTokensTwitter)
        if y:
            tn, fp, fn, tp = confusion_matrix(y, pre['prediction']).ravel()
            results[name]['trueSar'] = tp
            results[name]['falseSar'] = fp
            results[name]['trueSer'] = tn
            results[name]['falseSer'] = fn
            results[name]['score'] = (tp + tn) / len(y)

    if graph:
        import plotly.offline as py
        import plotly.graph_objs as go

        data = [
            go.Bar(
                x=list(results.keys()),
                y=[results[k]["score"] for k in results.keys()],
                text=["TP: {} FP: {} TN: {} FN: {}".format(results[k]["trueSar"], results[k]["falseSar"], results[k]["trueSer"], results[k]["falseSer"]) for k in results.keys()],
            )
        ]
        layout = go.Layout(
            title='Generalization score (n={})'.format(len(tweets))
        )

        fig = go.Figure(data=data, layout=layout)
        py.plot(fig)

    return results
