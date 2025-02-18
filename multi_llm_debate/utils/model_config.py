from typing import TypedDict


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    provider: str
    name: str
    quantity: int
