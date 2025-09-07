import streamlit as st
from rag_orchestrator import load_retriever, process_query

# --- MUST be the first Streamlit command ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Load retriever once ---
@st.cache_resource
def get_retriever():
    try:
        return load_retriever()
    except Exception as e:
        st.error("⚠️ FATAL: Could not load the vector database. Please ensure 'ingest.py' has been run successfully.")
        st.stop()
        return None

retriever = get_retriever()

# Now you can use other Streamlit commands
st.title("RAG Chatbot")
st.write("Welcome to the chatbot!")
# Input text box
user_prompt = st.text_input("Enter your question:", placeholder="Type your query here...")

if st.button("Submit") or user_prompt:
    if retriever is None:
        st.error("❌ Knowledge base not loaded.")
    elif not user_prompt.strip():
        st.warning("⚠️ Please enter a valid prompt.")
    else:
        try:
            response, model_used = process_query(user_prompt, retriever)
            st.success(f"**Response:** {response}")
            st.caption(f"Model used: `{model_used}`")
        except Exception as e:
            st.error(f"🚨 Unexpected error: {e}")

# Optional: Keep a chat history
if "history" not in st.session_state:
    st.session_state.history = []

if user_prompt:
    st.session_state.history.append(("You", user_prompt))
    if retriever:
        try:
            response, model_used = process_query(user_prompt, retriever)
            st.session_state.history.append(("Bot", f"[{model_used}] {response}"))
        except Exception as e:
            st.session_state.history.append(("Bot", f"Unexpected error: {e}"))

# Display history
if st.session_state.history:
    st.subheader("💬 Chat History")
    for sender, msg in st.session_state.history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")
