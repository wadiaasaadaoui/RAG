import pytest
import os

@pytest.fixture(scope="session")
def test_config():
    return {
        "model_config": {
            "model_name": "gpt2",
            "device": "cpu"  # Pour les tests
        },
        "vector_store": {
            "db_path": ":memory:"  # SQLite en mémoire pour les tests
        }
    }