"""Abstract base class for mute providers."""

from abc import ABC, abstractmethod


class MuteProvider(ABC):
    """Interface for anything that can mute/unmute the TV when ads are detected."""

    @abstractmethod
    def mute(self) -> None:
        """Mute the TV."""
        ...

    @abstractmethod
    def unmute(self) -> None:
        """Unmute the TV."""
        ...

    def close(self) -> None:
        """Release any resources held by the provider. Override if needed."""
        pass
