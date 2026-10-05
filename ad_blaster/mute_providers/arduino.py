"""Mute provider that sends IR commands via an Arduino over serial."""

import serial
from .base import MuteProvider


class ArduinoMuteProvider(MuteProvider):
    """Sends b'1' (mute) and b'2' (unmute) over serial to an Arduino IR blaster."""

    def __init__(self, tty_device: str, baudrate: int = 9600):
        self._serial = serial.Serial(port=tty_device, baudrate=baudrate, timeout=0.1)

    def mute(self) -> None:
        self._serial.write(b"1")

    def unmute(self) -> None:
        self._serial.write(b"2")

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
