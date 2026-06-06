"""Custom exception classes for the customer feedback analysis project."""

class CustomerFeedbackError(Exception):
    """Base class for all project-specific exceptions."""


class DataSchemaError(CustomerFeedbackError):
    """Raised when an expected dataset schema cannot be detected."""


class DataLoadError(CustomerFeedbackError):
    """Raised when dataset loading fails."""


class ModelTrainingError(CustomerFeedbackError):
    """Raised when model training or evaluation fails."""
