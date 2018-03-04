#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:10:49 2017

@author: Flore
"""

from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

#####################################################################################
#####################################################################################
#    FUNCTIONS 
#####################################################################################
#Cross validation score
def compute_score(classifier,X,y):
    xval = cross_val_score(classifier,X,y,cv=10)
    return mean(xval)
#####################################################################################
#Data visualisation
def plot_hist(feature, train, bins = 20):
    churned = train[train['Churn?'] == 1]
    not_churned = train[train['Churn?'] == 0]
    print churned
    print not_churned
    x1 = array(churned[feature].dropna())
    x2 = array(not_churned[feature].dropna())
    plt.hist([x1,x2],label=['Churned','Not Churned'], bins=bins)
    plt.legend(loc='upper left')
    plt.title('Relative Distribution of %s' %feature)
    plt.show()

#####################################################################################
#Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#####################################################################################
#####################################################################################
#     BEGINNING OF THE CODE
#####################################################################################
#Load the training data set
data = pd.read_csv('base.csv', delimiter=',')
print(data.shape)
y = data['Churn?']
#print data['Churn?']

#Data visualisation
#plot_hist('State',data)

#####################################################################################
#Feature engineering
X = data
del X['Churn?']
del X['Phone']
intplan_split = pd.get_dummies(X["Int'l Plan"],prefix='intplan')
X = X.join(intplan_split)
mailplan_split = pd.get_dummies(X["VMail Plan"],prefix='mailplan')
X = X.join(mailplan_split)
state_split = pd.get_dummies(X["State"],prefix='state')
X = X.join(state_split)
areacode_split = pd.get_dummies(X["Area Code"],prefix='areacode')
X = X.join(areacode_split)

to_del = ["Int'l Plan", 'VMail Plan', 'State', 'Area Code']
for col in to_del : del X[col]
print X.shape

#####################################################################################
#Cross validation 
X, X_test, y, y_test = cross_validation.train_test_split(X,y, test_size=0.2, random_state=42)

#####################################################################################
#Recherche des meilleurs paramètres pour random forest
#random = []
#parameters = [75, 85, 95, 100, 105, 115, 125]
#for k in parameters:
#    rf = RandomForestClassifier(max_depth = 17, min_samples_split = 12, n_estimators = k, random_state = 1)
#    rf = rf.fit(X,y)
#    random.append(compute_score(rf,X,y))
#plt.plot(parameters,random)

#Random Forest optimisé
rf = RandomForestClassifier(max_depth=17, min_samples_split=12, n_estimators=115, random_state=1)
rf.fit(X, y)
print('Random Forest Classifier' , compute_score(rf,X,y))
y_pred1 = rf.predict(X)

#####################################################################################
#Recherche des meilleurs paramètres pour gradient boosting classifier
#random = []
#parameters = [10,25,50,75,100,125,150,175,200]
#for k in parameters:
#    gb = GradientBoostingClassifier(random_state=1, n_estimators=k)
#    gb.fit(X, y)
#    random.append(compute_score(gb,X,y))
#plt.plot(parameters,random)

#Gradient Boosting Classifier optimisé
gb = GradientBoostingClassifier(n_estimators = 100, random_state=1)
gb.fit(X, y)
print('Gradient Boosting Classifier' , compute_score(gb,X,y))
y_pred2 = gb.predict(X)

#####################################################################################
#CONFUSION MATRIX
cnf_matrix = confusion_matrix(y, y_pred1)
set_printoptions(precision=2)

## Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')

## Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()






