from fastapi.testclient import TestClient
import pytest
import os


CATEGORICALS = [
    "gender",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "partner",
    "dependents",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies"
]

INTS = [
    "age",
    "tenure_months",
    "num_support_tickets",
    "num_logins_last_month",
    "late_payments"
]

FLOATS = [
    "monthly_charges",
    "total_charges",
    "feature_usage_score"
]

sample_data = {
    "gender": "Male",
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "paperless_billing": "Yes",
    "partner": "No",
    "dependents": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "age": 25,
    "tenure_months": 1,
    "monthly_charges": 29.85,
    "total_charges": 29.85,
    "num_support_tickets": 0,
    "num_logins_last_month": 10,
    "feature_usage_score": 50.5,
    "late_payments": 0,
}

incorrect_data: list[dict[str, str | int | float]] = []
for catg in CATEGORICALS:
    sample: str = sample_data[catg]     # type: ignore
    incorrect_data.extend(map(
        lambda v: {catg: v},            # type: ignore
        [
            sample[0],
            sample * 2,
            sample[:-1],
            "",
            "Q",
            0,
            1.0
        ]
    ))

    if not sample.isupper():
        incorrect_data.append({catg: sample.upper()})
    if not sample.islower():
        incorrect_data.append({catg: sample.lower()})
    if sample.isalpha():
        incorrect_data.append({catg: sample + "1"})

for intg in INTS:
    incorrect_data.extend(map(
        lambda v: {intg: v},            # type: ignore
        [
            -1,
            "",
            "Q",
            1.0
        ]
    ))

for fltg in FLOATS:
    incorrect_data.extend(map(
        lambda v: {fltg: v},            # type: ignore
        [
            -1.0,
            "",
            "Q",
            # 1      # StrictFloat can still receive int
        ]
    ))


def test_health(client: TestClient):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "version" in response.json()


def test_readiness(client: TestClient):
    """Test the readiness endpoint."""
    response = client.get("/readiness")
    assert response.status_code == 200


def test_predict_single(client: TestClient):
    """Test the single prediction endpoint."""
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], list)
    assert isinstance(data["probability"], list)
    assert len(data["prediction"]) == 1
    assert len(data["probability"]) == 2


def test_predict_batch(client: TestClient):
    """Test the batch prediction endpoint."""
    response = client.post("/batch-predict", json=[sample_data, sample_data])
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], list)
    assert isinstance(data["probability"], list)
    assert len(data["prediction"]) == 2
    assert len(data["probability"]) == 2
    assert len(data["probability"][0]) == 2


def test_readiness_error():
    """Test the readiness endpoint when the model is not loaded."""
    original_model_path = os.environ.get("MODEL_PATH")
    assert original_model_path
    os.environ["MODEL_PATH"] = ""

    # Create a new app instance
    from src.main import app, lifespan, context
    context.clear()
    app.router.lifespan_context = lifespan

    with TestClient(app) as c:
        response = c.get("/readiness")
        assert response.status_code == 503

    # Restore the environment variable
    os.environ["MODEL_PATH"] = original_model_path

    with TestClient(app):
        pass


@pytest.mark.parametrize("d", incorrect_data)
def test_predict_single_invalid_data(client: TestClient, d: dict):
    """Test the single prediction endpoint with invalid data."""
    invalid_data = sample_data.copy()
    invalid_data.update(**d)
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


@pytest.mark.parametrize("d", incorrect_data)
def test_predict_batch_invalid_data(client: TestClient, d: dict):
    """Test the batch prediction endpoint with invalid data."""
    invalid_data = sample_data.copy()
    invalid_data.update(**d)
    response = client.post("/batch-predict", json=[sample_data, invalid_data])
    assert response.status_code == 422
