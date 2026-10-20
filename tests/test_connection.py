import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
if not endpoint.endswith("/openai/v1"):
    endpoint = endpoint.split("/openai")[0] + "/openai/v1"

client = OpenAI(base_url=endpoint + "/", api_key=os.environ["AZURE_OPENAI_API_KEY"])

response = client.chat.completions.create(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    messages=[{"role": "user", "content": "Give me one line of excited football commentary for a stoppage-time equaliser."}],
)
print(response.choices[0].message.content)