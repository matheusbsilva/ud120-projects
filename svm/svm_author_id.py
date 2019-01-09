#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from sklearn.svm import SVC

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf_linear = SVC(kernel='linear')
clf_linear.fit(features_train, labels_train)

acc_linear = clf_linear.score(features_test, labels_test)

clf_rbf = SVC(kernel='rbf', C=10000)
clf_rbf.fit(features_train, labels_train)

acc_rbf = clf_rbf.score(features_test, labels_test)

print('Accurary of SVM with linear kernel:', acc_linear)
print('Accurary of SVM with RBF kernel:', acc_rbf)
