"""
Shared connection to the Azure OpenAI model.

Every agent imports `ask_json` from here, so the endpoint, key and
deployment name are read from .env in one place only.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None


def get_client():
    """Create the Azure OpenAI client once and reuse it."""
    global _client
    if _client is None:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint = endpoint.split("/openai")[0] + "/openai/v1"
        _client = OpenAI(base_url=endpoint + "/", api_key=os.environ["AZURE_OPENAI_API_KEY"])
    return _client


def ask_json(system_prompt, user_content):
    """Send a prompt to the model and return its reply parsed as JSON."""
    response = get_client().chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
