from __future__ import absolute_import, print_function

try:
  import cPickle as pickle
except ImportError:
  import pickle

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

  def parallel_fit(self, X, y, client_token=None, est_timeout=None):
    self.n_outputs_ = 1
    self.classes_ = np.array(np.unique(check_array(y, ensure_2d=False,
                                                   allow_nd=True, dtype=None)))

    if est_timeout is None:
      est_timeout = int(1e6)

    # Store X and y data for workers to use
    with open(self.X_file.name, 'wb') as outfile:
      pickle.dump(X, outfile, pickle.HIGHEST_PROTOCOL)
    with open(self.y_file.name, 'wb') as outfile:
      pickle.dump(y, outfile, pickle.HIGHEST_PROTOCOL)

    sigopt_procs = []
    for build_args in self.estimator_build_args:
      # run separaete python process for each estimator with timeout
      # these processes are wrapped in timeout command to capture case
      # where a single observation never completes
      sigopt_procs.append(Popen([
        "timeout", str(est_timeout + 10), "python", sklearn_fit.__file__,
        "--opt_timeout", str(est_timeout),
        "--estimator", build_args['estimator'],
        "--X_file", build_args['X_file'], "--y_file", build_args['y_file'],
        "--client_token", client_token,
        "--output_file", build_args['output_file']
      ]))
    exit_codes = [p.wait() for p in sigopt_procs]
    return_codes_args = zip(exit_codes, self.estimator_build_args)

    # remove estimators that errored or timed out
    valid_est_args = [rc_args[1] for rc_args in return_codes_args
                      if rc_args[0] == 0]

    # load valid estimators back into memory
    for est_arg in valid_est_args:
      with open(est_arg['output_file'], 'rb') as infile:
        clf = pickle.load(infile)
        self.estimator_ensemble.append(clf)

