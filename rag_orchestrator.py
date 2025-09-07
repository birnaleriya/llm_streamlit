import requests
import streamlit as st

# Replace with actual hosted API endpoints
MODEL_ENDPOINTS = {
    "Mistral": "https://api.mistral.ai/v1/generate",
    "Phi3": "https://api.phi3.ai/v1/generate"
}

# API keys from Streamlit Secrets
MODEL_KEYS = {
    "Mistral": st.secrets["MISTRAL_API_KEY"],
    "Phi3": st.secrets["PHI3_API_KEY"]
}

def query_model(model_name: str, prompt: str) -> str:
    """
    Query a hosted model API using the API key.
    """
    try:
        url = MODEL_ENDPOINTS[model_name]
        headers = {"Authorization": f"Bearer {MODEL_KEYS[model_name]}"}
        payload = {"prompt": prompt, "max_tokens": 300}  # customize per API

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", data.get("text", "No response received"))

    except Exception as e:
        return f"⚠️ Error querying {model_name}: {str(e)}"

def process_query(prompt: str):
    """
    Multi-model query (MCP) function for Streamlit.
    """
    responses = {}
    for model_name in MODEL_ENDPOINTS.keys():
        responses[model_name] = query_model(model_name, prompt)
    return responses
