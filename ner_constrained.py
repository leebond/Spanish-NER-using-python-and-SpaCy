from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
import re
import sklearn_crfsuite
import eli5
from sklearn.model_selection import GridSearchCV


# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

def is_nonalphanum(word):
    return re.match("^\w+", word) is None

def is_nonalpha(word):
    return re.match("[^A-Za-z]", word) is None

def is_lenshort(word, l = 2):
    return len(word) <= l

def contain_dot(word):
    return re.match("\w+[.]\w+", word) is not None

def contain_hyphen(word):
    return re.match("\w+[-]\w+", word) is not None

def gettokenfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    token_features = [
        (o + 'word', word)
        ,('word.isupper()', word.isupper())
#        ,('word.istitle()', word.istitle())
#        ,('word.islower()', word.islower())
#        ,('word.isdigit()', word.isdigit())
#        ,('word.is_nonalphanum', is_nonalphanum(word))
        ,('word.is_nonalpha', is_nonalpha(word))
        ,('word_lenshort', is_lenshort(word, 2))
#        ,('word_contains_dot', contain_dot(word))
#        ,('word_contains_hyphen', contain_hyphen(word))
    ]
#    print(token_features)
    return token_features
    
def getposfeats(pos, o):
    """ This takes the pos in question and
    the offset with respect to the instance
    pos """
    o = str(o)
    pos_features = [
        (o + 'pos', pos)
    ]
#    print(pos_features)
    return pos_features

def getiobfeats(iob, o):
    """ This takes the pos in question and
    the offset with respect to the instance
    pos """
    o = str(o)
    iob_features = [
        (o + 'iob', iob)
    ]
#    print(pos_features)
    return iob_features

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the target
    b = 3 # token window size best = 3
    for o in list(range(-b, b+1)):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist_token = gettokenfeats(word, o)
            features.extend(featlist_token)
            
    m = 1 # pos window size best = 1
    for o in list(range(-m, m+1)):
        if i+o >= 0 and i+o < len(sent):
            pos = sent[i+o][1]
            featlist_pos = getposfeats(pos, o)
            features.extend(featlist_pos)  

    n = 2 # iob window size best = 2
    for o in [i for i in range(-n, n+1) if i != 0]:
        if i+o >= 0 and i+o < len(sent):
            iob = sent[i+o][2]
            featlist_iob = getiobfeats(iob, o)
            features.extend(featlist_iob)  
    
    return dict(features)

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []

    for sent in train_sents + dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    ## switch these on when training on perceptron model
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    model = Perceptron(max_iter = 100, early_stopping = True, tol = 0.001, verbose=0)
#    model = sklearn_crfsuite.CRF(
#        algorithm='lbfgs',
#        c1=0.1,
#        c2=0.1,
#        max_iterations=20,
#        all_possible_transitions=False,
#    )
    
    ## switch these on when training on perceptron model
    model.fit(X_train, train_labels)
    
    ## switch these on when training on crf model
#    model.fit([[x] for x in train_feats], [[x] for x in train_labels])

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    ## switch dev_sents to test_sents
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    ## switch these on when training on perceptron model
    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)
    ## switch these on when training on crf model
#    y_pred = model.predict(test_feats)


    j = 0
    print("Writing to constrained_results.txt")
    # format is: word gold pred
    with open("constrained_results.txt", "w") as out:
        for sent in test_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py constrained_results.txt")