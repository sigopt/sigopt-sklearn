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

### CV Fold Timeouts

SigOptSearchCV performs evaluations on cv folds in parallel using
joblib.  Timeouts are now supported in the master branch of joblib and
SigOpt can use this timeout information to learn to avoid hyperparameter 
configurations that are too slow.  An example is shown below

You'll need to install joblib from source for this example to work.
```
pip uninstall joblib
git clone https://github.com/joblib/joblib.git`
cd joblib; pip setup.py install
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
# for each fit.  SigOpt will then avoid these slower configurations
clf = SigOptSearchCV(svr, svc_parameters, cv=5, timeout=180,
	client_token=client_token, n_jobs=5, n_iter=40)

clf.fit(X, y)
```
