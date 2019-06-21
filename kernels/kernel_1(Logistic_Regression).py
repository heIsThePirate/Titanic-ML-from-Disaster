# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:27:12 2019

@author: mohit
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')

dataset_1 = dataset.iloc[:, [0,1,2,3,4,5,6,7,9,11]]

dataset_1 = dataset_1.dropna()

X_1 = dataset_1.iloc[:, [2, 4, 5, 6, 7, 8, 9]].values
y_1 = dataset_1.iloc[:, 1].values

# Handling missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputer = imputer.fit(X[:, 2:3])
#X[:, 2:3] = imputer.transform(X[:, 2:3])

# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X_1[:, 1] = labelencoder_1.fit_transform(X_1[:, 1])
labelencoder_2 = LabelEncoder()
X_1[:, 6] = labelencoder_2.fit_transform(X_1[:, 6])
onehotencoder_1 = OneHotEncoder(categorical_features=[6])
X_1 = onehotencoder_1.fit_transform(X_1).toarray()

X_1 = X_1[:, 1:]

onehotencoder_2 = OneHotEncoder(categorical_features=[2])
X_1 = onehotencoder_2.fit_transform(X_1).toarray()

X_1 = X_1[:, 1:]

# Scaling the values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_1 = sc_X.fit_transform(X_1)

# Using Linear Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_1, y_1)

# Predicting the results
test_set = pd.read_csv('test.csv')

test_set_1 = test_set.iloc[:, [0,1,2,3,4,5,6,8,10]]

X_test_1 = test_set_1.iloc[:, [1,3,4,5,6,7,8]].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X_test_1[:, [2, 5]])
X_test_1[:, [2, 5]] = imputer.transform(X_test_1[:, [2, 5]])


X_test_1[:, 1] = labelencoder_1.transform(X_test_1[:, 1])
X_test_1[:, 6] = labelencoder_2.transform(X_test_1[:, 6])
X_test_1 = onehotencoder_1.transform(X_test_1).toarray()

X_test_1 = X_test_1[:, 1:]

X_test_1 = onehotencoder_2.transform(X_test_1).toarray()

X_test_1 = X_test_1[:, 1:]

X_test_1 = sc_X.transform(X_test_1)

y_pred = classifier.predict(X_test_1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_1, classifier.predict(X_1))

pred = test_set_1.iloc[:, 0:2].values
pred[:, 1] = y_pred

pred = pd.DataFrame(data = pred, columns = ['PassengerId', 'Survived'])
pred.to_csv('Passengers_Survived.csv')

# accuracy (on training set) = 80.2%