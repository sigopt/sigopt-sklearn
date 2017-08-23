import pytest
import warnings

from sigopt_sklearn.ensemble import SigOptEnsembleClassifier

warnings.simplefilter("error", append=True)

class TestEnsemble(object):
  def test_create(self):
    SigOptEnsembleClassifier()
