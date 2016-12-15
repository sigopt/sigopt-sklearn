from mock import MagicMock, patch
import pytest

import sklearn.datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import sigopt

from sigopt_sklearn.search import SigOptSearchCV

from test_util import random_assignments


GradientBoostingClassifier_PARAM_DOMAIN = {
  'n_estimators': (20, 500),
  'min_samples_split': (2, 5),
  'min_samples_leaf': (1, 3),
  'learning_rate': (0.01, 1.0)
}
GradientBoostingClassifier_EXPERIMENT_DEF = {
  'name': 'GradientBoostingClassifier (sklearn)',
  'parameters': [
    {
      'type': 'int',
      'name': 'n_estimators',
      'bounds': {
        'min': 20,
        'max': 500,
      },
    },
    {
      'type': 'int',
      'name': 'min_samples_split',
      'bounds': {
        'min': 2,
        'max': 5,
      },
    },
    {
      'type': 'int',
      'name': 'min_samples_leaf',
      'bounds': {
        'min': 1,
        'max': 3,
      },
    },
    {
      'type': 'double',
      'name': 'learning_rate',
      'bounds': {
        'min': 0.01,
        'max': 1.0,
      },
    },
  ],
}

SVC_PARAM_DOMAIN = {
  'C': {'little': 1e-3, 'some': 1, 'lots': 1e3}
}
SVC_EXPERIMENT_DEF = {
  'name': 'SVC with fancy categoricals',
  'parameters': [
    {
      'type': 'categorical',
      'name': 'C',
      'categorical_values': [
        {'name': 'little'},
        {'name': 'some'},
        {'name': 'lots'}
      ]
    }
  ]
}

def zero_corner(experiment_definition):
  """Take the parameters corresponding to the zero corner. All of the minimums and the first categories."""
  return {p['name']: (p['bounds']['min'] if p['type'] in ['int', 'double'] else p['categorical_values'][0]['name'])
          for p in experiment_definition['parameters']}

def mock_connection(experiment_definition):
  return MagicMock(return_value=MagicMock(
    experiments=MagicMock(return_value=MagicMock(
      create=MagicMock(return_value=MagicMock(
        id="exp_id",
      )),
      fetch=MagicMock(return_value=MagicMock(
        progress=MagicMock(
          best_observation=MagicMock(
            assignments=MagicMock(
              to_json=MagicMock(return_value=zero_corner(experiment_definition)),
        ))),
      )),
      suggestions=MagicMock(return_value=MagicMock(
        create=MagicMock(return_value=MagicMock(
          assignments=MagicMock(
            to_json=MagicMock(side_effect=lambda: random_assignments(experiment_definition))),
          id="sugg_id",
        )),
      )),
      observations=MagicMock(return_value=MagicMock(
        create=MagicMock(return_value=MagicMock(
          id="obs_id",
          value=52,
        )),
      )),
      best_assignments=MagicMock(return_value=MagicMock(
        fetch=MagicMock(return_value=MagicMock(
          data=[MagicMock(
            assignments=MagicMock(
              to_json=MagicMock(return_value=zero_corner(experiment_definition)),
            )
          )],
        )),
      )),
    ))
  ))

class TestSearch(object):
  def test_create(self):
    SigOptSearchCV(
      estimator=GradientBoostingClassifier,
      param_domains=GradientBoostingClassifier_PARAM_DOMAIN,
      client_token='client_token'
    )

  def test_no_token(self):
    with pytest.raises(ValueError):
      SigOptSearchCV(estimator=GradientBoostingClassifier, param_domains=GradientBoostingClassifier_PARAM_DOMAIN)

  @patch('sigopt.Connection', new=mock_connection(GradientBoostingClassifier_EXPERIMENT_DEF))
  def test_search(self):
    conn = sigopt.Connection()

    n_iter = 5
    folds = 3
    cv = SigOptSearchCV(
      estimator=GradientBoostingClassifier(),
      param_domains=GradientBoostingClassifier_PARAM_DOMAIN,
      client_token='client_token',
      n_iter=n_iter,
      cv=folds
    )
    assert len(conn.experiments().create.mock_calls) == 0
    assert len(conn.experiments().fetch.mock_calls) == 0
    assert len(conn.experiments().suggestions.create.mock_calls) == 0
    assert len(conn.experiments().observations.create.mock_calls) == 0

    data = sklearn.datasets.load_iris()
    cv.fit(data['data'], data['target'])
    assert len(conn.experiments().create.mock_calls) == 1
    create_definition = conn.experiments().create.call_args[1]
    assert create_definition['name'] == GradientBoostingClassifier_EXPERIMENT_DEF['name']

    assert len(create_definition['parameters']) == len(GradientBoostingClassifier_EXPERIMENT_DEF['parameters'])
    for p in GradientBoostingClassifier_EXPERIMENT_DEF['parameters']:
      assert p in create_definition['parameters']
    assert len(conn.experiments().best_assignments().fetch.mock_calls) == 1
    assert len(conn.experiments().suggestions().create.mock_calls) == n_iter * folds
    assert len(conn.experiments().observations().create.mock_calls) == n_iter * folds

    assert cv.best_params_ == zero_corner(GradientBoostingClassifier_EXPERIMENT_DEF)

  @patch('sigopt.Connection', new=mock_connection(SVC_EXPERIMENT_DEF))
  def test_non_string_categorical(self):
    data = sklearn.datasets.load_iris()
    clf = SigOptSearchCV(SVC(), SVC_PARAM_DOMAIN, client_token='client_token', n_iter=5)
    clf.fit(data['data'], data['target'])

  def test_bad_param_range1(self):
    with pytest.raises(Exception):
      clf = SigOptSearchCV(
        SVC(),
        {
          'bad_param_range': (1,),
          'hidden_layer_sizes': {'5': (5,), '5,4,3': (5, 4, 3)}
        },
        client_token='client_token',
        n_iter=5
      )
      clf._transform_param_domains(clf.param_domains)

  def test_bad_param_range2(self):
    with pytest.raises(Exception):
      clf = SigOptSearchCV(
        SVC(),
        {
          'bad_param_range': (1, 2, 3),
          'hidden_layer_sizes': {'5': (5,), '5,4,3': (5, 4, 3)}
        },
        client_token='client_token',
        n_iter=5
      )
      clf._transform_param_domains(clf.param_domains)

  def test_warn_param_range_list(self):
    with pytest.warns(UserWarning):
      clf = SigOptSearchCV(
        SVC(),
        {'max_iter': [5, 10]},
        client_token='client_token',
        n_iter=5
      )
      clf._transform_param_domains(clf.param_domains)

  def test_bad_param_range_not_iterable(self):
    with pytest.raises(Exception):
      clf = SigOptSearchCV(
        SVC(),
        {'max_iter': 15},
        client_token='client_token',
        n_iter=5
      )
      clf._transform_param_domains(clf.param_domains)
