import pytest
from fastapi.testclient import TestClient
import os
from src.main import app

# Set the model path for the tests
os.environ["MODEL_PATH"] = "models"


@pytest.fixture(scope="module")
def client():
    """
    Test client for the FastAPI application.
    """
    with TestClient(app) as c:
        yield c
