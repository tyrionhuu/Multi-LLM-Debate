import json
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config():
    if not CONFIG_FILE.exists():
        return {
            "api_key": "",
            "base_url": "https://api2.aigcbest.top/v1",
            "models": [
                {"provider": "api", "name": "claude-3-5-sonnet-20241022"},
                {"provider": "api", "name": "gpt-4o-2024-11-20"},
                {"provider": "ollama", "name": "llama3.2-vision:11b"},
                {"provider": "ollama", "name": "llava:13b"},
                {"provider": "ollama", "name": "llama3.2-vision:90b"}
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


def get_models() -> List[Tuple[str, str]]:
    config = load_config()
    return [(model["provider"], model["name"]) for model in config.get("models", [])]


def save_api_key(key: str) -> None:
    config = load_config()
    config["api_key"] = key
    save_config(config)
