from http import HTTPStatus
import pandas as pd
import requests
from pathlib import Path
resp = requests.get("https://www.levels.fyi/js/salaryData.json")
print(f"Response: {resp.status_code}")
if resp.status_code != HTTPStatus.OK:
    # AFAIK this endpoint only responds with HTTP 200 and not any other 2xx
    # status for successful queries
    raise RuntimeError(f"Possible failure in fetching salaryData.json: {resp.status_code}")

data = resp.json()
df = pd.DataFrame(data)
download_dir = Path("data/comp_analysis")
download_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(download_dir / "salaryData.csv", index=False)
