from __future__ import absolute_import, print_function

import cPickle as pickle
from subprocess import Popen
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array

import sigopt_sklearn.sklearn_fit as sklearn_fit
from sigopt_sklearn.sklearn_fit import ESTIMATOR_NAMES


class SigOptEnsembleClassifier(ClassifierMixin):
  def __init__(self):
    self.estimator_ensemble = []
    self.estimator_bayes_avg_coefs = []
    self.estimator_build_args = []

    self.n_outputs_ = None
    self.classes_ = None

    # create build args for all models in ensemble
    self._construct_ensemble_build_args()

  def _construct_ensemble_build_args(self):
    # Generate names for input and output files
    self.X_file = NamedTemporaryFile()
    self.y_file = NamedTemporaryFile()
    for est_name in ESTIMATOR_NAMES:
       results_file = NamedTemporaryFile()
       arg_dict = {'X_file': self.X_file.name, 'y_file': self.y_file.name,
                   'output_file': results_file.name, 'estimator': est_name}
       self.estimator_build_args.append(arg_dict)

  def _find_bayes_avg_coefs(self, X, y):
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

  def fit(self, X, y, client_token=None, est_timeout=None):
    self.n_outputs_ = 1
    self.classes_ = np.array(np.unique(check_array(y, ensure_2d=False,
                                                   allow_nd=True, dtype=None)))
    # Store X and y data for workers to use
    with open(self.X_file.name, 'wb') as outfile:
      pickle.dump(X, outfile, pickle.HIGHEST_PROTOCOL)
    with open(self.y_file.name, 'wb') as outfile:
      pickle.dump(y, outfile, pickle.HIGHEST_PROTOCOL)

    sigopt_procs = []
    for build_args in self.estimator_build_args:
      # run separate python process for each estiamtor with timeout
      p = Popen(["timeout", str(est_timeout), "python", sklearn_fit.__file__,
           "--estimator", build_args['estimator'],
           "--X_file", build_args['X_file'], "--y_file", build_args['y_file'],
           "--client_token", client_token,
           "--output_file", build_args['output_file']])
      sigopt_procs.append(p)
    exit_codes = [p.wait() for _ in sigopt_procs]
    return_codes_args = zip(exit_codes, self.estimator_build_args)

    # remove estimators that errored or timed out
    valid_est_args = [rc_args[1] for rc_args in return_codes_args
                      if rc_args[0] == 0]

    # load valid estimators back into memory
    for est_arg in valid_est_args:
      with open(est_arg['output_file'], 'rb') as infile:
        clf = pickle.load(infile)
        self.estimator_ensemble.append(clf)

    # find weights for ensemble
    self.estimator_bayes_avg_coefs = self._find_bayes_avg_coefs(X, y)

  def predict_proba(self, X):
    # validate X
    # TODO : dim wrong when using 1d array
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
