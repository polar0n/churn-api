import pickle
import pandas as pd
import pytest
from src.main import predict


@pytest.fixture(scope="module")
def artifacts():
    """Load the model and preprocessing objects."""
    with open("models/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/preprocessing.pkl", "rb") as f:
        preprocessing = pickle.load(f)
    return model, preprocessing


@pytest.fixture(scope="module")
def sample_data():
    """Load sample data from the CSV file for testing."""
    return pd.read_csv("customer_churn_dataset.csv")


def test_artifacts_loading(artifacts):
    """Test if the model and preprocessing objects are loaded correctly."""
    model, preprocessing = artifacts
    assert model is not None, "Model should not be None"
    assert preprocessing is not None, "Preprocessing object should not be None"
    assert "label_encoders" in preprocessing
    assert "scaler" in preprocessing
    assert "categorical_cols" in preprocessing
    assert "numerical_cols" in preprocessing
    assert "feature_columns" in preprocessing


def test_prediction_output_type_and_shape(artifacts, sample_data):
    """Test the output type and shape of a single prediction."""
    model, preprocessing = artifacts
    single_record = sample_data.iloc[[0]].drop(columns=["customer_id", "churn"])
    result = predict(single_record, preprocessing, model)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"], list)
    assert isinstance(result["probability"], list)
    assert len(result["prediction"]) == 1
    assert len(result["probability"]) == 1
    assert len(result["probability"][0]) == 2  # Probabilities for two classes (0 and 1)


def test_batch_prediction_output(artifacts, sample_data):
    """Test the output of a batch prediction."""
    model, preprocessing = artifacts
    batch_records = sample_data.head(3).drop(columns=["customer_id", "churn"])
    result = predict(batch_records, preprocessing, model)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"], list)
    assert isinstance(result["probability"], list)
    assert len(result["prediction"]) == 3
    assert len(result["probability"]) == 3


def test_prediction(artifacts, sample_data):
    """Test prediction accuracy with known data from the CSV."""
    model, preprocessing = artifacts

    non_churner_data = sample_data[sample_data["churn"] == 0].head(1)
    non_churner_record = non_churner_data.drop(columns=["customer_id", "churn"])

    result_non_churner = predict(non_churner_record, preprocessing, model)
    assert result_non_churner["prediction"][0] == non_churner_data["churn"].iloc[0]

    churner_data = sample_data[sample_data["churn"] == 1].head(1)
    churner_record = churner_data.drop(columns=["customer_id", "churn"])

    result_churner = predict(churner_record, preprocessing, model)
    assert result_churner["prediction"][0] == churner_data["churn"].iloc[0]
