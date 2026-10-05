from .base import MuteProvider
from .arduino import ArduinoMuteProvider
from .log import LogMuteProvider


def create_mute_provider(config: dict) -> MuteProvider:
    """Instantiate the configured mute provider.

    Config shape:
        mute_provider:
          type: arduino        # or "log"
          tty_device: /dev/tty.usbserial-...
          baudrate: 9600       # optional, defaults to 9600
    """
    provider_config = config.get("mute_provider", {})
    provider_type = provider_config.get("type", "log")

    if provider_type == "arduino":
        tty_device = provider_config["tty_device"]
        baudrate = provider_config.get("baudrate", 9600)
        return ArduinoMuteProvider(tty_device=tty_device, baudrate=baudrate)

    if provider_type == "log":
        return LogMuteProvider()

    raise ValueError(f"Unknown mute_provider type: {provider_type!r}")
