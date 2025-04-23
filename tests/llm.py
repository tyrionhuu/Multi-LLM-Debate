import subprocess
from multi_llm_debate.llm.llm import call_model

def get_gcloud_access_token() -> str:
    """Get the current gcloud access token.

    Returns:
        str: The access token from gcloud.
    """
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

response = call_model(
    model_name="gemini-2.0-flash-001",
    base_url="https://us-central1-aiplatform.googleapis.com/v1/projects/multi-llm-debate/locations/us-central1/publishers/google/models/gemini-2.0-flash-001:generateContent",
    prompt="What is the capital of France?",
    api_key=get_gcloud_access_token()
)