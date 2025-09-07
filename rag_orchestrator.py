import re
import ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
MISTRAL_MODEL = "mistral"
PHI_MODEL = "phi3"
DB_FAISS_PATH = "vectorstore/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PHI_KEYWORDS = [
    'code', 'script', 'technical', 'malware', 'forensics', 'vulnerability', 
    'exploit', 'packet', 'network', 'binary', 'debug', 'reverse engineer', 
    'threat hunting', 'ransomware', 'server', 'attack', 'breach', 'section', 'act'
]
# New keywords specifically for the log analysis task
LOG_ANALYSIS_KEYWORDS = ['analyze this log', 'examine this log', 'check this log', 'analyze log']

def load_retriever():
    """Loads the FAISS vector store and initializes the retriever."""
    print("--- Loading Vector Database and Initializing Retriever ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 4})
    print("--- Retriever ready. ---")
    return retriever

def route_query(query: str) -> str:
    """Decides which model (Mistral or Phi) is best suited to answer standard RAG queries."""
    query_lower = query.lower()
    if any(re.search(r'\b' + keyword + r'\b', query_lower) for keyword in PHI_KEYWORDS):
        return PHI_MODEL
    return MISTRAL_MODEL

def process_query(query: str, retriever):
    """
    Main processing function. Includes a special path for log analysis and a standard RAG path for others.
    """
    print(f"\nProcessing new query: '{query}'")
    query_lower = query.lower()

    # --- NEW: Special Path for Log Analysis ---
    # Check if the query is a log analysis request.
    is_log_analysis = any(keyword in query_lower for keyword in LOG_ANALYSIS_KEYWORDS)

    if is_log_analysis:
        print("Log analysis task detected. Bypassing RAG and using Phi-3's inherent skill.")
        
        system_prompt = """
        You are an expert security analyst. Your task is to analyze the provided log data.
        Identify any anomalies, errors, warnings, or potential security threats.
        Provide a concise summary of your findings in a clear, structured format.
        Focus only on the data provided by the user.
        """
        
        # The user's query IS the log data in this case. We pass it directly.
        full_prompt = f"{system_prompt}\n\nLOG DATA:\n{query}"
        chosen_model = PHI_MODEL
        
        print(f"Sending log data directly to '{chosen_model}' for analysis...")

    # --- Standard RAG Path for all other queries ---
    else:
        # 1. Retrieve context from the vector database
        relevant_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 2. Route the query to the best model for the task
        chosen_model = route_query(query)
        
        print(f"Standard RAG query. Routing to '{chosen_model}'.")

        # 3. Construct the prompt with the retrieved context
        if chosen_model == PHI_MODEL:
            system_prompt = """
            You are a highly skilled digital forensics analyst and cybersecurity expert named 'Phi'.
            You will be given a user's query and a context of relevant technical documents.
            Your task is to provide a precise, technical, and factual answer based ONLY on the provided context.
            If the context does not contain the answer, state clearly that the information is not available in the provided documents.
            Do not use your general knowledge.
            """
        else: # Mistral
            system_prompt = """
            You are a helpful and articulate assistant specializing in cyber law named 'Mistral'.
            Your primary goal is to provide a clear, easy-to-understand answer based ONLY on the provided context.
            If the context does not contain enough information, state that you cannot provide an answer based on the available documents.
            You are not a lawyer; do not provide legal advice. Frame your answers as informational summaries of the text provided.
            """
            
        prompt_template = f"CONTEXT:\n{context}\n\nQUERY:\n{query}\n\nBased on the context provided, answer the query."
        full_prompt = f"{system_prompt}\n\n{prompt_template}"

    # --- Common final step: Call the chosen Ollama model ---
    try:
        response = ollama.chat(
            model=chosen_model,
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content'], chosen_model
    except Exception as e:
        print(f"An error occurred while calling the Ollama model: {e}")
        error_message = "Sorry, I encountered an error. Please ensure the Ollama server is running and the models are available."
        return error_message, "system_error"

