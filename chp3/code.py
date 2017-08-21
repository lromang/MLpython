##################################################
##
## ML in Python: chapter 3
##
##################################################

## ----------------------------------------
## Libraries
## ----------------------------------------
from sklearn                 import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import Perceptron
from sklearn                 import preprocessing
from sklearn.metrics         import accuracy_score

import numpy as np

## ----------------------------------------
## Read in dataset
## ----------------------------------------
iris = datasets.load_iris()
X    = iris.data[:, [2, 3]]
y    = iris.target

## ----------------------------------------
## Split data into train and test datasets
## ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 0
)

## ----------------------------------------
## Preprocessing
## ----------------------------------------

## Scale the data
sc = preprocessing.StandardScaler()

## 'Fit' scale parameters
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

## ----------------------------------------
## Model fitting
## ----------------------------------------

## Perceptron parameters
ppn = Perceptron(n_iter       = 40,
                 eta0         = 0.1,
                 random_state = 0)
## Fitting preceptron
ppn.fit(X_train_std, y_train)

## ----------------------------------------
## Prediction
## ----------------------------------------
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d ' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
