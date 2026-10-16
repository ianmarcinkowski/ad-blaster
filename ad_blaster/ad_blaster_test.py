import asyncio
import pytest
from ad_blaster.ad_blaster import AdBlaster, State, find_tool_use
from ad_blaster.mute_providers.log import LogMuteProvider
from ollama import SystemOneResponse
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_db():
    return MagicMock(cursor=MagicMock())

@pytest.fixture
def config():
    return {
        "ollama_uri": "http://localhost:11434",
        "model": "clef-flash",
        "ad_detection_threshold": 0.5,
        "webcam": {
            "width": 640,
            "height": 480,
        },
        "db_path": ":memory:",
    }

@pytest.fixture
def ad_blaster(config, mock_db):
    return AdBlaster(config, mock_db, LogMuteProvider())

def test_default_model_is_clef_flash(mock_db):
    blaster = AdBlaster({"webcam": {"width": 640, "height": 480}}, mock_db, LogMuteProvider())
    assert blaster.model == "clef-flash"
    assert blaster.llm_hostname == "http://localhost:11434"

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
    with patch.object(ad_blaster.mute_provider, "mute") as mock_mute:
        ad_blaster.state = State.UNMUTED
        ad_blaster.update_state(State.MUTED)
        mock_mute.assert_called_once()

    with patch.object(ad_blaster.mute_provider, "mute") as mock_mute:
        ad_blaster.state = State.MUTED
        ad_blaster.update_state(State.MUTED)
        mock_mute.assert_not_called()

def test_triggering_unmute(ad_blaster):
    with patch.object(ad_blaster.mute_provider, "unmute") as mock_unmute:
        ad_blaster.state = State.MUTED
        ad_blaster.update_state(State.UNMUTED)
        mock_unmute.assert_called_once()

    with patch.object(ad_blaster.mute_provider, "unmute") as mock_unmute:
        ad_blaster.state = State.UNMUTED
        ad_blaster.update_state(State.UNMUTED)
        mock_unmute.assert_not_called()

def test_find_tool_use_function_call():
    raw_string = """```python
        advertising_detected(True)
    ```
    """
    function_name, function_args = find_tool_use(raw_string)
    assert function_name == "advertising_detected"
    assert function_args == "True"

def test_detect_ads_with_clef_flash_positive(ad_blaster):
    mock_response = SystemOneResponse(
        model="clef-flash",
        answers={
            "advertising_detected": {"type": "noul", "noul": 0.92},
            "category": {
                "type": "choice",
                "choice": "advertisement",
                "probabilities": {"advertisement": 0.91, "program": 0.09},
                "confidence": 0.85,
            },
        },
        usage={"input_tokens": 300, "output_tokens": 0},
    )
    with patch.object(ad_blaster.ollama, "_request", new=AsyncMock(return_value=mock_response)) as mock_req:
        detected, reason, metrics = asyncio.run(ad_blaster.detect_ads(["dGVzdA=="]))
        assert detected is True
        assert "p(ad)=0.920" in reason
        assert "category=advertisement" in reason
        assert metrics["prompt_eval_duration"] == "300"
        mock_req.assert_awaited_once()
        _, method, path = mock_req.call_args.args
        payload = mock_req.call_args.kwargs["json"]
        assert method == "POST"
        assert path == "/v1/systemone"
        assert payload["model"] == "clef-flash"
        assert payload["images"] == ["dGVzdA=="]

def test_detect_ads_with_clef_flash_negative(ad_blaster):
    mock_response = SystemOneResponse(
        model="clef-flash",
        answers={
            "advertising_detected": {"type": "noul", "noul": 0.08},
            "category": {
                "type": "choice",
                "choice": "program",
                "probabilities": {"advertisement": 0.08, "program": 0.92},
                "confidence": 0.88,
            },
        },
        usage={"input_tokens": 300, "output_tokens": 0},
    )
    with patch.object(ad_blaster.ollama, "_request", new=AsyncMock(return_value=mock_response)):
        detected, reason, _ = asyncio.run(ad_blaster.detect_ads([b"dGVzdA=="]))
        assert detected is False
        assert "p(ad)=0.080" in reason
        assert "category=program" in reason