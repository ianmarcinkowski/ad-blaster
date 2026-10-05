"""No-op mute provider that logs actions instead of controlling hardware.

Useful for development, testing, or running without any mute hardware attached.
"""

from .base import MuteProvider


class LogMuteProvider(MuteProvider):
    """Logs mute/unmute calls; takes no real action."""

    def mute(self) -> None:
        print("[LogMuteProvider] MUTE")

    def unmute(self) -> None:
        print("[LogMuteProvider] UNMUTE")
