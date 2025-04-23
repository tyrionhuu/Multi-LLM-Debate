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
        check=True,
    )
    return result.stdout.strip()


response = call_model(
    model_name="google/gemini-2.0-flash-001",
    prompt="What is the capital of France?",
    api_key=get_gcloud_access_token(),
)
print(response)
