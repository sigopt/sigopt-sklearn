from __future__ import absolute_import, print_function

import argparse
import math
try:
  import cPickle as pickle
except ImportError:
  import pickle

import scipy.sparse
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sigopt_sklearn.search import SigOptSearchCV


ESTIMATOR_NAMES = [
    "SVMClassifier",
    "GaussianNBClassifier",
    "RandomForestClassifier",
    "SGDClassifier",
    "XGBClassifier",
    "KNNClassifier",
    "LDAClassifier",
]

def parse_args():
  parser = argparse.ArgumentParser(
    description='SigOpt sklearn estimator fit script',
  )

  parser.add_argument(
    '--estimator',
    type=str,
    required=True,
    help='name of sklearn estimator',
  )

  parser.add_argument(
    '--opt_timeout',
    type=int,
    help="max time alloted for optimizing",
    required=False,
    default=None,
  )

  parser.add_argument(
   '--X_file',
   type=str,
   required=True,
   help='path of training data matrix X',
  )

  parser.add_argument(
   '--y_file',
   type=str,
   required=True,
   help='path of label array y',
  )

  parser.add_argument(
   '--output_file',
   type=str,
   required=True,
   help='path of file to store classifier',
  )

  parser.add_argument(
   '--client_token',
   type=str,
   required=True,
   help='SigOpt client token',
  )

  args = parser.parse_args()
  return args

def main():
  # convert arg structure to regular dict
  args = vars(parse_args())
  X_path = args['X_file']
  y_path = args['y_file']
  client_token = args['client_token']
  estimator_name = args['estimator']
  output_path = args['output_file']
  opt_timeout = args['opt_timeout']
  with open(X_path, 'rb') as infile:
    X = pickle.load(infile)
  with open(y_path, 'rb') as infile:
    y = pickle.load(infile)

  # define param doimains for all esimators
  rf_params = {
    'max_features': ['sqrt', 'log2'],
    'max_depth': [3, 20],
    'criterion': ['gini', 'entropy'],
    'n_estimators': [10, 100],
  }

  svm_params = {
    'degree': [2, 4],
    '__log__C': [math.log(0.00001), math.log(1.0)],
    'gamma': [0.0, 1.0]
  }

  knn_params = {
    'n_neighbors': [2, 10],
    'algorithm': ['ball_tree', 'kd_tree'],
    'leaf_size': [10, 50],
    'p': [1, 3]
  }

  sgd_params = {
    '__log__alpha': [math.log(0.00001), math.log(10.0)],
    'l1_ratio': [0.0, 1.0],
    'loss': ['log', 'modified_huber']
  }

  xgb_params = {
    '__log__learning_rate': [math.log(0.0001),math.log(0.5)],
    'n_estimators':  [10, 100],
    'max_depth': [3, 10],
    'min_child_weight': [6, 12],
    'gamma': [0, 0.5],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
  }

  lda_params = { "__log__tol": [math.log(0.00001), math.log(0.5)] }
  qda_params = { "__log__tol": [math.log(0.00001), math.log(0.5)] }

  # mapping from classifier name to estimaitor object and domain
  # dict stores : (estimator, hyperparams, sparse_support)
  estname_2_args = {
    "GaussianNBClassifier": (GaussianNB(), None, False),
    "SVMClassifier": (SVC(probability=True), svm_params, True),
    "RandomForestClassifier": (RandomForestClassifier(n_jobs=2),
                               rf_params, True),
    "SGDClassifier": (SGDClassifier(penalty='elasticnet'),
                      sgd_params, True),
    "XGBClassifier": (XGBClassifier(nthread=2), xgb_params, True),
    "KNNClassifier": (KNeighborsClassifier(n_jobs=2), knn_params, False),
    "LDAClassifier": (LinearDiscriminantAnalysis(), lda_params, False),
    "QDAClassifier": (QuadraticDiscriminantAnalysis(), qda_params, False),
  }
  est, est_params, est_handle_sparse = estname_2_args[estimator_name]

  # check that estimator can handle sparse matrices
  if scipy.sparse.issparse(X) and not est_handle_sparse:
    raise Exception('{} does not support sparse matrices.'.format(estimator_name))
  elif est_params is not None:
    # fit the estimator if it has params to tune
    n_iter = max(10 * len(est_params), 20)
    clf = SigOptSearchCV(
      est,
      est_params,
      cv=3,
      opt_timeout=opt_timeout,
      client_token=client_token,
      n_jobs=3,
      n_iter=n_iter,
    )
  else:
    clf = est

  clf.fit(X, y)
  if hasattr(clf, 'best_estimator_'):
    clf = clf.best_estimator_
  # store classifier in specified output file
  with open(output_path, 'wb') as outfile:
    pickle.dump(clf, outfile, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  main()
