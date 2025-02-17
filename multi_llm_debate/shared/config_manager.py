import json
from pathlib import Path

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
CONFIG_FILE = CONFIG_DIR / 'config.json'

def load_config():
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_config(config):
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def get_api_key():
    config = load_config()
    return config.get('api_key', '')

def save_api_key(key):
    config = load_config()
    config['api_key'] = key
    save_config(config)
