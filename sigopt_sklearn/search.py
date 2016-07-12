from __future__ import absolute_import, print_function

import math
from multiprocessing import TimeoutError
import time

import numpy
import sigopt
from joblib import Parallel, delayed
from joblib.func_inspect import getfullargspec
from sklearn.grid_search import BaseSearchCV
from sklearn.cross_validation import check_cv
from sklearn.cross_validation import _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, indexable
from sklearn.base import is_classifier, clone


class SigOptSearchCV(BaseSearchCV):
    """SigOpt powered search on hyper parameters.
    SigOptSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.
    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.
    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is chosen from the specified
    domains. The number of parameter settings that are tried is
    given by n_iter.
    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_domains : dict
        Dictionary with parameters names (string) as keys and domains
        as lists of parameter ranges to try. Domains are either lists of categorial
        (string) values or 2 element lists specifying a min and max for
        integer or float parameters
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    client_token : string
        SigOpt API client token, find yours here : https://sigopt.com/user/profile
    opt_timeout : float, optional
        Max time for entire optimization process
    cv_timeout : float, optional
        Max time each CV fold objective evaluation can take
    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.
    fit_params : dict, optional
        Parameters to pass to the fit method.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this RandomizedSearchCV instance after fitting.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.
    best_score_ : float
        Score of best_estimator on the left out data.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    """
    def __init__(self, estimator, param_domains, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 cv_timeout=None, opt_timeout=None, client_token=None):

        self.param_domains = param_domains
        self.n_iter = n_iter
        if not client_token:
            raise ValueError("Please find your client token here : https://sigopt.com/user/profile")
        self.client_token = client_token
        self.cv_timeout = cv_timeout
        self.opt_timeout = opt_timeout
        self.verbose = verbose

        self.scorer_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.experiment = None

        super(SigOptSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score)

    def _create_sigopt_exp(self, conn):
        est_name = self.estimator.__class__.__name__
        exp_name = est_name + " (sklearn)"
        if len(exp_name) > 50:
            exp_name = est_name

        if self.verbose > 0:
            print("Creating SigOpt experiment: ", exp_name)

        # generate sigopt experiment parameters
        parameters = []
        for param_name in self.param_domains:
            bounds = self.param_domains[param_name]
            all_ints = all(isinstance(x, int) for x in bounds)
            all_floats = all(isinstance(x, float) for x in bounds)
            any_floats = any(isinstance(x, float) for x in bounds)
            all_strings = all(isinstance(x, str) for x in bounds)

            # convert entire bounds to floats if one is present
            if any_floats:
                bounds = [float(b) for b in bounds]
                all_floats = True

            types = ['double', 'int', 'categorical']
            bound_types = [all_floats, all_ints, all_strings]
            param_type = types[bound_types.index(True)]

            param_dict = {}
            param_dict['name'] = param_name
            param_dict['type'] = param_type

            if param_type == 'categorical':
                cat_vals = []
                for str_name in bounds:
                    cat_vals.append({"name": str_name})
                param_dict['categorical_values'] = cat_vals
            else:
                bmin = min(bounds)
                bmax = max(bounds)
                param_dict['bounds'] = {"min": bmin, "max": bmax}

            # add parameter definition to list
            parameters.append(param_dict)

        # create sigopt experiment
        self.experiment = conn.experiments().create(
            name=exp_name,
            parameters=parameters)

        if self.verbose > 0:
            exp_url = "https://sigopt.com/experiment/{0}".format(self.experiment.id)
            print("Experiment progress available at :", exp_url)

    def _convert_log_params(self, param_dict):
      # searches through names for params and converts params with __log__ names
      log_converted_dict = {}
      for pname in param_dict:
        pval = param_dict[pname]
        if "__log__" in pname:
          pval = math.exp(pval)
          pname = pname.replace("__log__", "")
        log_converted_dict[pname] = pval
      return log_converted_dict

    def _fit(self, X, y, parameter_iterable=None):
        if parameter_iterable is not None:
            raise NotImplementedError('The parameter_iterable argument is not supported.')

        # Actual fitting,  performing the search over parameters.
        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))

        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        # setup SigOpt experiment and run optimization
        conn = sigopt.Connection(client_token=self.client_token)
        self._create_sigopt_exp(conn)

        # start tracking time to optimize estimator
        opt_start_time = time.time()
        for jk in range(self.n_iter):
            # check for opt timeout, ensuring at least 1 observation
            # TODO : handling failure observations
            if (
                self.opt_timeout is not None and
                time.time() - opt_start_time > self.opt_timeout and
                jk >= 1
            ):
                # break out of loop and refit model with best params so far
                break
            suggestion = conn.experiments(self.experiment.id).suggestions().create()
            parameters = suggestion.assignments.to_json()

            # automatically convert __log__ params
            parameters = self._convert_log_params(parameters)

            if self.verbose > 0:
                print("Evaluating params : ", parameters)

            # do CV folds in parallel using joblib
            # returns scores on test set
            obs_timed_out = False
            try:
                par_kwargs = {"n_jobs": self.n_jobs, "verbose": self.verbose,
                              "pre_dispatch": pre_dispatch}
                # add timeout kwarg if version of joblib supports it
                if 'timeout' in getfullargspec(Parallel.__init__).args:
                    par_kwargs['timeout'] = self.cv_timeout
                out = Parallel(
                    **par_kwargs
                )(
                    delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                            train, test, self.verbose, parameters,
                                            self.fit_params, return_parameters=True,
                                            error_score=self.error_score)
                        for train, test in cv)
            except TimeoutError:
                 obs_timed_out = True

            if not obs_timed_out:
                # grab scores from results
                scores = [o[0] for o in out]
                conn.experiments(self.experiment.id).observations().create(
                    suggestion=suggestion.id,
                    value=numpy.mean(scores),
                    value_stddev=numpy.std(scores)
                )

            else:
                # obsevation timed out so report a failure
                conn.experiments(self.experiment.id).observations().create(
                    suggestion=suggestion.id,
                    failed=True,
                )

        # return best SigOpt observation so far
        best_obs = conn.experiments(self.experiment.id).fetch().progress.best_observation
        self.best_params_ = best_obs.assignments.to_json()
        # convert all unicode names and values to plain strings
        self.best_params_ = self._convert_log_params(self.best_params_)
        self.best_score_ = best_obs.value

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

    def fit(self, X, y=None):
        """
        Run fit on the estimator with parameters chosen sequentially by SigOpt.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        """
        return self._fit(X, y)
