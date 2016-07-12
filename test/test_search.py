from mock import MagicMock, patch
import pytest

from sklearn.ensemble import GradientBoostingClassifier
import sklearn.datasets
import sigopt

from sigopt_sklearn.search import SigOptSearchCV


class TestSearch(object):
  @pytest.fixture(params=[
    (GradientBoostingClassifier, {
      'n_estimators': [20, 500],
      'min_samples_split': [1, 4],
      'min_samples_leaf': [1, 3],
      'learning_rate': [0.01, 1.0],
    }, {
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
            'min': 1,
            'max': 4,
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
    }),
  ])
  def estimator_params_experiment(self, request):
    return request.param

  def to_sample_params(self, param_domains):
    return dict(((p[0], p[1][0]) for p in param_domains.items()))

  @pytest.fixture
  def param_domains(self, estimator_params_experiment):
    return estimator_params_experiment[1]

  @pytest.fixture
  def estimator(self, estimator_params_experiment, param_domains):
    estimator_cls = estimator_params_experiment[0]
    return estimator_cls(**self.to_sample_params(param_domains))

  @pytest.fixture
  def experiment_definition(self, estimator_params_experiment):
    return estimator_params_experiment[2]

  def test_create(self, estimator, param_domains):
    SigOptSearchCV(estimator=estimator, param_domains=param_domains, client_token='client_token')

  def test_no_token(self, estimator, param_domains):
    with pytest.raises(ValueError):
      SigOptSearchCV(estimator=estimator, param_domains=param_domains)

  @patch('sigopt.Connection')
  def test_search(self, Connection, estimator, param_domains, experiment_definition):
    BEST_PARAMS = self.to_sample_params(param_domains)
    conn = sigopt.Connection()
    conn.experiments = MagicMock(return_value=MagicMock(
      create=MagicMock(return_value=MagicMock(
        id="exp_id",
      )),
      fetch=MagicMock(return_value=MagicMock(
        progress=MagicMock(best_observation=MagicMock(assignments=MagicMock(
          to_json=MagicMock(return_value=BEST_PARAMS),
        ))),
      )),
      suggestions=MagicMock(return_value=MagicMock(
        create=MagicMock(return_value=MagicMock(
          id="sugg_id",
        )),
      )),
      observations=MagicMock(return_value=MagicMock(
        create=MagicMock(return_value=MagicMock(
          id="obs_id",
          value=52,
        )),
      )),
    ))
    n_iter = 5
    cv = SigOptSearchCV(estimator=estimator, param_domains=param_domains, client_token='client_token', n_iter=n_iter)
    assert len(conn.experiments().create.mock_calls) == 0
    assert len(conn.experiments().fetch.mock_calls) == 0
    assert len(conn.experiments().suggestions.create.mock_calls) == 0
    assert len(conn.experiments().observations.create.mock_calls) == 0

    data = sklearn.datasets.load_digits()
    cv.fit(data['data'], data['target'])
    assert len(conn.experiments().create.mock_calls) == 1
    create_definition = conn.experiments().create.call_args[1]
    assert create_definition['name'] == experiment_definition['name']

    assert len(create_definition['parameters']) == len(experiment_definition['parameters'])
    for p in experiment_definition['parameters']:
      assert p in create_definition['parameters']
    assert len(conn.experiments().fetch.mock_calls) == 1
    assert len(conn.experiments().suggestions().create.mock_calls) == n_iter
    assert len(conn.experiments().observations().create.mock_calls) == n_iter

    assert cv.best_params_ == BEST_PARAMS
