import json
from pathlib import Path
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config():
    if not CONFIG_FILE.exists():
        return {
            "api_key": "",
            "base_url": "https://api2.aigcbest.top/v1",
            "models": [
                {
                    "provider": "api",
                    "name": "claude-3-5-sonnet-20241022",
                    "quantity": 1,
                },
                {"provider": "api", "name": "gpt-4o-2024-11-20", "quantity": 1},
                {"provider": "ollama", "name": "llama3.2-vision:11b", "quantity": 1},
                {"provider": "ollama", "name": "llava:13b", "quantity": 1},
                {"provider": "ollama", "name": "llama3.2-vision:90b", "quantity": 1},
            ],
        }
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def get_api_key() -> str:
    config = load_config()
    return config.get("api_key", "")


def get_base_url() -> str:
    config = load_config()
    return config.get("base_url")  # Remove redundant default value


def get_models() -> List[Tuple[str, str, int]]:
    config = load_config()
    return [
        (model["provider"], model["name"], model["quantity"])
        for model in config.get("models", [])
    ]


def save_api_key(key: str) -> None:
    config = load_config()
    config["api_key"] = key
    save_config(config)


def get_vllm_url() -> str:
    """Get the vLLM server URL from the configuration.

    Returns:
        str: The URL of the vLLM server. Default is http://localhost:8000.
    """
    config = load_config()
    return config.get("vllm_url", "http://localhost:8000")


def save_vllm_url(url: str) -> None:
    """Save the vLLM server URL to the configuration.

    Args:
        url: The URL of the vLLM server.
    """
    config = load_config()
    config["vllm_url"] = url
    save_config(config)


def get_vllm_model_path() -> Dict[str, str]:
    """Get the vLLM model paths from the configuration.
    
    This function returns a dictionary mapping model names to their local paths.

    Returns:
        Dict[str, str]: Dictionary of model name to local path mappings.
                       Default includes common open models.
    """
    config = load_config()
    default_paths = {
        "llama3": "/path/to/llama3",
        "mistral": "/path/to/mistral",
        # Add other default model paths as needed
    }
    return config.get("vllm_model_paths", default_paths)


def save_vllm_model_path(model_name: str, model_path: str) -> None:
    """Save a vLLM model path to the configuration.

    Args:
        model_name: The name to reference the model by
        model_path: The local path to the model files
    """
    config = load_config()
    if "vllm_model_paths" not in config:
        config["vllm_model_paths"] = {}
    config["vllm_model_paths"][model_name] = model_path
    save_config(config)
