from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.joblib"

SPECIES = (
    "setosa",
    "versicolor",
    "virginica",
)


def predict_species(features: list[float]) -> str:
    """Predict Iris species for four flower measurements."""
    if len(features) != 4:
        raise ValueError("Exactly four features are required.")

    model = joblib.load(MODEL_PATH)

    prediction = model.predict([features])[0]

    return SPECIES[prediction]


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]
    prediction = predict_species(sample)

    print(f"Predicted species: {prediction}")
