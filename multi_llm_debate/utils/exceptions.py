class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass

class DatasetValidationError(DatasetError):
    """Raised when dataset format validation fails."""
    pass

class DatasetDownloadError(DatasetError):
    """Raised when dataset download fails."""
    pass
