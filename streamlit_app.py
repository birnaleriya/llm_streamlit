import streamlit as st
from rag_orchestrator import process_query

st.set_page_config(
    page_title="MCP Hosted Models",
    page_icon="🤖",
    layout="wide"
)

st.title("Multi-Choice Prompting with Hosted Models")

prompt = st.text_area("Enter your prompt here:")

if st.button("Generate Responses"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        st.info("Querying hosted models...")
        responses = process_query(prompt)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mistral Response")
            st.write(responses["Mistral"])
        with col2:
            st.subheader("Phi3 Response")
            st.write(responses["Phi3"])
