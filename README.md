# SigOpt + scikit-learn Interfacing
This package implements useful interfaces and wrappers for using [SigOpt](https://sigopt.com) and [scikit-learn](http://scikit-learn.org/stable/) together

## Getting Started

Install the sigopt_sklearn python modules with `pip install sigopt_sklearn`.

Sign up for an account at [https://sigopt.com](https://sigopt.com).
To use the interfaces, you'll need your API token from your [user profile](https://sigopt.com/user/profile).

### SigOptSearchCV

The simplest use case for SigOpt in conjunction with scikit-learn is optimizing
estimator hyperparameters using cross validation.  A short example that tunes the 
parameters of an SVM on a small dataset is provided below

```python
from sklearn import svm, datasets
from sigopt_sklearn.search import SigOptSearchCV

# find your SigOpt client token here : https://sigopt.com/user/profile
client_token = "<YOUR_SIGOPT_CLIENT_TOKEN>"

iris = datasets.load_iris()

# define parameter domains
svc_parameters  = {'kernel': ['linear', 'rbf'], 'C': [0.5, 100]}

# define sklearn estimator
svr = svm.SVC()

# define SigOptCV search strategy
clf = SigOptSearchCV(svr, svc_parameters, cv=5, 
	client_token=client_token, n_jobs=5, n_iter=20)

# perform CV search for best parameters and fits estimator
# on all data using best found configuration
clf.fit(iris.data, iris.target)

# clf.predict() now uses best found estimator 
# clf.best_score_ contains CV score for best found estimator
# clf.best_params_ contains best found param configuration
```

The objective optimized by default is is the default score associated with the estimator
, however users can provide a different metric by passing the `scoring` option to the SigOptSearchCV constructor.
Shown below is an example that uses the f1_score already implemented in sklearn

```python
from sklearn.metrics import f1_score, make_scorer
f1_scorer = make_scorer(f1_score)

# define SigOptCV search strategy
clf = SigOptSearchCV(svr, svc_parameters, cv=5, scoring=f1_scorer,
    client_token=client_token, n_jobs=5, n_iter=50)

# perform CV search for best parameters
clf.fit(X, y)
```

### XGBoostClassifier

SigOptSearchCV also works with XGBoost's XGBClassifier wrapper.  A
hyperparameter search over XGBClassifier models can be done using the same interface

```python
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import datasets
from sigopt_sklearn.search import SigOptSearchCV

# find your SigOpt client token here : https://sigopt.com/user/profile
client_token = "<YOUR_SIGOPT_CLIENT_TOKEN>"
iris = datasets.load_iris()

xgb_params = {
 'learning_rate' : [0.01, 0.5],
 'n_estimators' :  [10, 50],
 'max_depth':[3, 10],
 'min_child_weight':[6, 12],
 'gamma':[0, 0.5],
 'subsample':[0.6, 1.0],
 'colsample_bytree':[0.6, 1.0]
}

xgbc = XGBClassifier()

clf = SigOptSearchCV(xgbc, xgb_params, cv=5,
    client_token=client_token, n_jobs=5, n_iter=70, verbose=1)

clf.fit(iris.data, iris.target)
```

### SigOptEnsembleClassifier

This class concurrently trains and tunes several classification models within sklearn to facilitate model selection
efforts when investigating new datasets.  A short example, using an activity recognition dataset is provided below

We also have a video tutorial outlining how to run this example here :

[![SigOpt scikit-learn Tutorial](http://img.youtube.com/vi/9XZ3ihE7OjM/0.jpg)](http://www.youtube.com/watch?v=9XZ3ihE7OjM "SigOpt scikit-learn Hyperparameter Optimization Tutorial")

```
# Human Activity Recognition Using Smartphone
# https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip UCI\ HAR\ Dataset.zip
cd UCI\ HAR\ Dataset
```

```python
import numpy as np
import pandas as pd
from sigopt_sklearn.ensemble import SigOptEnsembleClassifier

def load_datafile(filename):
  X = []
  with open(filename,"r") as f:
    for l in f:
      X.append(np.array(map(float,l.split())))
  X = np.vstack(X)
  return X
X_train = load_datafile("train/X_train.txt")
y_train = load_datafile("train/y_train.txt").ravel()
X_test = load_datafile("test/X_test.txt")
y_test = load_datafile("test/y_test.txt").ravel()

# fit and tune several classification models concurrently
# find your SigOpt client token here : https://sigopt.com/user/profile
sigopt_clf = SigOptEnsembleClassifier()
sigopt_clf.parallel_fit(X_train, y_train, est_timeout=(40 * 60),
               client_token='<YOUR_CLIENT_TOKEN>')

# compare model performance on hold out set
ensemble_train_scores = [est.score(X_train,y_train) for est in sigopt_clf.estimator_ensemble]
ensemble_test_scores = [est.score(X_test,y_test) for est in sigopt_clf.estimator_ensemble]
data = sorted(zip([est.__class__.__name__
                        for est in sigopt_clf.estimator_ensemble], ensemble_train_scores, ensemble_test_scores),
                        reverse=True, key= lambda x : (x[2], x[1]))
pd.DataFrame(data, columns=['Classifier ALGO.', 'Train ACC.', 'Test ACC.'])
```

### CV Fold Timeouts

SigOptSearchCV performs evaluations on cv folds in parallel using
joblib.  Timeouts are now supported in the master branch of joblib and
SigOpt can use this timeout information to learn to avoid hyperparameter 
configurations that are too slow. 

You'll need to install joblib from source for this example to work.
```
pip uninstall joblib
git clone https://github.com/joblib/joblib.git`
cd joblib; python setup.py install
```
Installation flow also explained on the [joblib github page](https://github.com/joblib/joblib#installing)


```python
from sklearn import svm, datasets
from sigopt_sklearn.search import SigOptSearchCV

# find your SigOpt client token here : https://sigopt.com/user/profile
client_token = "<YOUR_SIGOPT_CLIENT_TOKEN>"
dataset = datasets.fetch_20newsgroups_vectorized()
X = dataset.data
y = dataset.target

# define parameter domains
svc_parameters  = {'kernel': ['linear', 'rbf'], 'C': [0.5, 100], 
                   'max_iter': [10, 200], 'tol': [1e-2,1e-6]}
svr = svm.SVC()

# SVM fitting can be quite slow, so we set timeout = 180 seconds 
# for each fit.  SigOpt will then avoid configurations that are too slow
clf = SigOptSearchCV(svr, svc_parameters, cv=5, timeout=180,
	client_token=client_token, n_jobs=5, n_iter=40)

clf.fit(X, y)
```
