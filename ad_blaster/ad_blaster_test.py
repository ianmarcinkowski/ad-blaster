import pytest
import sqlite3
from ad_blaster.ad_blaster import AdBlaster, State, decide
from unittest.mock import MagicMock, patch

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

def test_decide():
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


def test_state_transitions(ad_blaster):
    # Start at unmuted, stay at unmuted
    ad_blaster.state = State.UNMUTED
    ad_blaster.update_state(State.UNMUTED)
    assert ad_blaster.state == State.UNMUTED

    # Start at unmuted, change to muted
    ad_blaster.state = State.UNMUTED
    ad_blaster.update_state(State.MUTED)
    assert ad_blaster.state == State.MUTED

    # Start at muted, stay at muted
    ad_blaster.state = State.MUTED
    ad_blaster.update_state(State.MUTED)
    assert ad_blaster.state == State.MUTED

    # Start at muted, change to unmuted
    ad_blaster.state = State.MUTED
    ad_blaster.update_state(State.UNMUTED)
    assert ad_blaster.state == State.UNMUTED


def test_triggering_mute(ad_blaster):
    with patch.object(ad_blaster, "send_mute") as mock_send_mute:
        ad_blaster.state = State.UNMUTED
        ad_blaster.update_state(State.MUTED)
        mock_send_mute.assert_called_once()

    with patch.object(ad_blaster, "send_mute") as mock_send_mute:
        ad_blaster.state = State.MUTED
        ad_blaster.update_state(State.MUTED)
        mock_send_mute.assert_not_called()

def test_triggering_unmute(ad_blaster):
    with patch.object(ad_blaster, "send_unmute") as mock_send_unmute:
        ad_blaster.state = State.MUTED
        ad_blaster.update_state(State.UNMUTED)
        mock_send_unmute.assert_called_once()

    with patch.object(ad_blaster, "send_unmute") as mock_send_unmute:
        ad_blaster.state = State.UNMUTED
        ad_blaster.update_state(State.UNMUTED)
        mock_send_unmute.assert_not_called()
