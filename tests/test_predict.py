import pytest

from iris_classifier.predict import predict_species


def test_predict_species_requires_four_features():
    """Check validation of the number of input features."""
    with pytest.raises(ValueError, match="Exactly four features are required"):
        predict_species([5.1, 3.5, 1.4])
