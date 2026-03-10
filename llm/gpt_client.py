from bootstrap import *
from openai import AzureOpenAI
from config import MODEL
from utils.io_utils import save_text

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://docexpresearch-api.azure-api.net/gpt4-docexpresearch",
    api_key="acb0aac95da54559be250d96f270a297"
)

def call_llm(prompt, name=None):

    if name:
        save_text(name + "_prompt", prompt)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    out = resp.choices[0].message.content

    if name:
        save_text(name + "_response", out)

    return out