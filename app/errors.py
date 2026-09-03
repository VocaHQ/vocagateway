from __future__ import annotations


class APIProblem(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.recoverable = recoverable


class InvalidAudioError(Exception):
    """The uploaded audio cannot be processed."""


class SilentAudioError(InvalidAudioError):
    """The uploaded audio contains no usable speech."""


class EngineUnavailableError(Exception):
    """The selected transcription engine is unavailable."""


class TranscriptionProcessError(Exception):
    """The transcription engine failed while processing audio."""


class LanguageUnsupportedError(TranscriptionProcessError):
    """The request asked for a language the loaded model cannot transcribe.

    A subclass so any handler that only knows about `TranscriptionProcessError`
    still catches it, but it gets its own error code because retrying is futile:
    the fix is to choose a different language or load a different model.
    """
