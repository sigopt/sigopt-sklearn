# SigOpt and scikit-learn Interfacing
This package implements useful interfaces and wrappers for using SigOpt and [scikit-learn](http://scikit-learn.org/stable/) together

## Getting Started

Install the sigopt_sklearn python modules with `pip install sigopt_sklearn`.

Sign up for an account at [https://sigopt.com](https://sigopt.com).
In order to use the API, you'll need your API token from your [user profile](https://sigopt.com/user/profile).

### SigOptSearchCV

The simplest use case for SigOpt in conjunction with scikit-learn is optimizing
estimator hyperparameters using cross validation.  A short example that tunes the 
parameters of an SVM on a small dataset is provided below

```python
from sklearn import svm, datasets
from sigopt_sklearn import SigOptSearchCV

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
