import numpy as np
from sklearn.base import ClassifierMixin
from search import SigOptSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import Parallel, delayed
from sklearn.utils.validation import check_array

def joblib_estimator_fit(X, y, estimator, param_space, cv, client_token):
  clf = estimator
  if param_space is not None:
    clf = SigOptSearchCV(estimator, param_space,
                         cv=cv, client_token=client_token,
                         n_jobs=5, n_iter=len(param_space)*10)
  clf.fit(X, y)
  if hasattr(clf, 'best_estimator_'):
    clf = clf.best_estimator_
  return clf


class SigOptEnsembleClassifier(ClassifierMixin):
  def __init__(self, verbose=0):
    self.estimator_ensemble = []
    self.estimator_bayes_avg_coefs = []
    self.estimator_build_args = []
    # create build args for all models in ensemble
    self.construct_ensemble_build_args()

  def construct_ensemble_build_args(self):
    # Add Gaussian NB Classifier
    self.estimator_build_args.append((GaussianNB(), None))

    # Add Random Forest Classifier
    rf_param = {
      'max_features': ['sqrt', 'log2'],
      'max_depth': [3, 20],
      'criterion': ['gini', 'entropy'],
      'n_estimators': [10, 100],
    }
    self.estimator_build_args.append((RandomForestClassifier(), rf_param))

    # Add KNN Classifier
    knn_params = {
      'n_neighbors': [2, 10],
      'algorithm': ['ball_tree', 'kd_tree'],
      'leaf_size': [10, 50],
      'p': [1, 3]
    }
    self.estimator_build_args.append((KNeighborsClassifier(), knn_params))

    # Add SGD Classifier
    sgd_params = {
      'alpha': [0.001, 10.0],
      'l1_ratio': [0.0, 1.0],
      'loss': ['log', 'modified_huber']
    }
    self.estimator_build_args.append((SGDClassifier(penalty='elasticnet'),
                                      sgd_params))

    # Add XGBoost Classifier
    xgb_params = {
     'learning_rate': [0.01, 0.5],
     'n_estimators':  [10, 100],
     'max_depth': [3, 10],
     'min_child_weight': [6, 12],
     'gamma': [0, 0.5],
     'subsample': [0.6, 1.0],
     'colsample_bytree': [0.6, 1.0]
    }
    #self.estimator_build_args.append((XGBClassifier(), xgb_params))

  def find_bayes_avg_coefs(self, X, y):
    log_likelihoods = []
    eps = 1e-4
    for est in self.estimator_ensemble:
      proba = est.predict_proba(X)
      proba = np.nan_to_num(proba)
      log_lik = sum(np.log(proba[np.arange(len(y)),y] + eps))
      log_likelihoods.append(log_lik)
    z = np.array(log_likelihoods)
    z /= min(-z)
    z = 1.0 / z
    z = z / np.sum(z)
    return z

  def fit(self, X, y, client_token=None, cv=5,
          cv_timeout=None, n_iter=25, n_jobs=1, n_cv_jobs=5):
    self.n_outputs_ = 1
    self.classes_ = np.unique(check_array(y, ensure_2d=False,
                                              allow_nd=True))
    estimators = Parallel(n_jobs=-2, backend='multiprocessing')(
      delayed(joblib_estimator_fit)(X, y, args[0], args[1], cv, client_token)
      for args in self.estimator_build_args
    )
    # TODO : Remove timed out estimators
    self.estimator_ensemble = estimators
    self.estimator_bayes_avg_coefs = self.find_bayes_avg_coefs(X, y)

  def predict_proba(self, X):
    # validate X
    res_proba = np.zeros((X.shape[0], len(self.classes_)))
    for idx, est in enumerate(self.estimator_ensemble):
      w = self.estimator_bayes_avg_coefs[idx]
      res_proba += w * np.nan_to_num(est.predict_proba(X))
    return res_proba

  def predict(self, X):
    proba = self.predict_proba(X)
    if self.n_outputs_ == 1:
      return self.classes_.take(np.argmax(proba, axis=1), axis=0)
    else:
      n_samples = proba[0].shape[0]
      predictions = np.zeros((n_samples, self.n_outputs_))

      for k in range(self.n_outputs_):
        predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                            axis=1),
                                                  axis=0)
      return predictions

