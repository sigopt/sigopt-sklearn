import pytest

from sigopt_sklearn.search import SigOptSearchCV

class TestSearch(object):
  @pytest.fixture
  def estimator(self):
    return None

  @pytest.fixture
  def param_domains(self):
    return None

  def test_create(self, estimator, param_domains):
    SigOptSearchCV(estimator=estimator, param_domains=param_domains, client_token='client_token')

  def test_no_token(self, estimator, param_domains):
    with pytest.raises(ValueError):
      SigOptSearchCV(estimator=estimator, param_domains=param_domains)
