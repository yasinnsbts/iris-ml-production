from sklearn.datasets import load_iris


def load_data():
    """Load the Iris dataset and return features and target."""
    iris = load_iris()

    X = iris.data
    y = iris.target

    return X, y
