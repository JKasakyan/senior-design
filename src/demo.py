import ml
import nlp
import pickle
import re

if __name__ == "__main__":
    results = pickle.load(open('-twitter-trained-log87.pickle', 'rb'))
    classifier = results[0][2][1][0] # best logistic
    dvp = pickle.load(open('-twitter-dvp87.pickle', 'rb'))

    while (True):
        print("Enter some text:")
        s = input()
        pre = ml.predict([s],
               classifier,
               dvp,
               nlp.cleanTokensTwitter)
        for p,pp in zip(pre['prediction'], pre['prediction_probabilities']):
            print('\tSarcastic' if p else '\tNon-sarcastic')
            print('\t'+str(pp[1]*100)+'%' if p else '\t'+str(pp[0]*100)+'%')
            print()
