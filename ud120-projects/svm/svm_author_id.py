#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
def trainer(kernel, c):
    clf = SVC(kernel=kernel, C=c)
    clf.fit(features_train, labels_train)
    return clf
#########################################################
### your code goes here ###
#clf = SVC(kernel="linear")

# clf = trainer("rbf", 10)
# t0 = time()
# print "training time:", round(time()-t0, 3), "s"
# print clf.score(features_test, labels_test)
# clf = trainer("rbf", 100)
# t0 = time()
# print "training time:", round(time()-t0, 3), "s"
# print clf.score(features_test, labels_test)
# clf = trainer("rbf", 1000)
# t0 = time()
# print "training time:", round(time()-t0, 3), "s"
# print clf.score(features_test, labels_test)
clf = trainer("rbf", 10000)
t0 = time()
print "training time:", round(time()-t0, 3), "s"
print clf.score(features_test, labels_test)
predictions = clf.predict(features_test)
print predictions[10]
acum = 0
for elem in predictions:
    if elem == 1:
        acum += 1

print acum
print len(filter(lambda x:x==1,predictions))
#########################################################


