"""Domain exceptions mapped to HTTP responses."""


class ReqGuardError(Exception):
    status_code = 500
    error = "An unexpected error occurred."

    def __init__(self, error: str | None = None) -> None:
        super().__init__(error or self.error)
        if error:
            self.error = error


class InvalidRequirementError(ReqGuardError):
    status_code = 400
    error = "Requirement cannot be empty."


class NLPProcessingError(ReqGuardError):
    status_code = 500
    error = "NLP processing failed."


class LLMUnavailableError(ReqGuardError):
    status_code = 503
    error = "LLM service is temporarily unavailable."


class LLMOutputError(ReqGuardError):
    status_code = 502
    error = "LLM response could not be validated."
