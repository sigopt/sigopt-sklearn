import pytest

from sigopt_sklearn.ensemble import SigOptEnsembleClassifier

class TestEnsemble(object):
  def test_create(self):
    SigOptEnsembleClassifier()
