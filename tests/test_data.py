from iris_classifier.data import load_data


def test_load_data():
    """Check that the Iris dataset has the expected structure."""
    X, y = load_data()

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert set(y) == {0, 1, 2}
