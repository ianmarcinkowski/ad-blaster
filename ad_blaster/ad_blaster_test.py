import pytest
import sqlite3
from ad_blaster.ad_blaster import AdBlaster, State, decide
from unittest.mock import MagicMock

@pytest.fixture
def mock_db():
    return MagicMock(
        cursor=MagicMock()
    )

@pytest.fixture
def config():
    return {
        "ollama_uri": "http://localhost:8000",
        "webcam": {
            "width": 640,
            "height": 480,
        },
        "db_path": ":memory:"  # Use an in-memory database for testing
    }

@pytest.fixture
def ad_blaster(config):
    return AdBlaster(config, mock_db)

def test_decide(ad_blaster):
    decision = decide("sports")
    assert decision == State.UNMUTED

    decision = decide("advertisement")
    assert decision == State.MUTED

    decision = decide("unknown")
    assert decision is None

    decision = decide("")
    assert decision is None

    decision = decide(None)
    assert decision is None