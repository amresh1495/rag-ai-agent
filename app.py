import streamlit as st
import os
from rag_core import (
    load_text_from_documents,
    split_documents,
    get_embeddings,
    create_vector_store,
    load_retriever, # This function in rag_core.py loads the FAISS vector_store
    initialize_llm,
    answer_query,
    VECTOR_STORE_PATH # Import the constant for checking existence
)

# --- Page Configuration ---
st.set_page_config(page_title="Customer Support Chatbot", layout="wide")

# --- Initialization Function ---
@st.cache_resource # Caches the resource across reruns for performance
def initialize_rag_components():
    """
    Loads documents, sets up embeddings, vector store, and LLM.
    This function is cached to run only once.
    """
    with st.spinner("Initializing chatbot components... This may take a moment."):
        # 1. Initialize Embeddings
        st.write("Initializing embeddings...")
        embeddings = get_embeddings()
        if embeddings is None:
            st.error("Failed to initialize embeddings. Chatbot cannot start.")
            return None, None

        # 2. Load or Create Vector Store
        st.write(f"Looking for existing vector store at '{VECTOR_STORE_PATH}'...")
        if os.path.exists(VECTOR_STORE_PATH):
            st.write("Existing vector store found. Loading...")
            vector_store = load_retriever(embeddings) # In rag_core, load_retriever loads the FAISS store
            if vector_store is None:
                st.warning("Failed to load existing vector store, will attempt to recreate.")
        else:
            vector_store = None # Explicitly set to None if not found

        if vector_store is None:
            st.write("No existing vector store or failed to load. Building a new one...")
            st.write("Loading documents...")
            docs = load_text_from_documents()
            if not docs:
                st.error("No documents found to build the vector store. Chatbot cannot start.")
                return None, None
            
            st.write("Splitting documents...")
            split_docs = split_documents(docs)
            if not split_docs:
                st.error("Failed to split documents. Chatbot cannot start.")
                return None, None
            
            st.write("Creating vector store... (This might take a while on first run)")
            vector_store = create_vector_store(split_docs, embeddings)
            if vector_store is None:
                st.error("Failed to create vector store. Chatbot cannot start.")
                return None, None
            st.success("New vector store created and saved successfully!")

        # 3. Initialize LLM
        st.write("Initializing Language Model...")
        try:
            llm = initialize_llm()
            if llm is None:
                st.error("Failed to initialize Language Model. Chatbot cannot start.")
                return None, None
        except Exception as e:
            st.error(f"Error initializing LLM: {e}")
            st.info("This might be due to missing dependencies like 'sentencepiece'. Try: pip install sentencepiece, then restart the app.")
            return None, None
            
        st.success("Chatbot components initialized successfully!")
        return llm, vector_store

# --- Load components ---
# Try to get components from session state first to preserve them across reruns
if 'llm' not in st.session_state or 'vector_store' not in st.session_state:
    st.session_state.llm, st.session_state.vector_store = initialize_rag_components()

# --- Chatbot UI ---
st.title("📄 Customer Support Chatbot")
st.markdown("""
Welcome to the Customer Support Chatbot. 
Ask questions about your benefit plans, and the chatbot will try to find relevant information from the documents.
""")

if st.session_state.llm and st.session_state.vector_store:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, doc in enumerate(message["sources"]):
                        st.write(f"**Source {i+1}: {doc.metadata.get('source', 'Unknown')}** (Score: {message['debug_info']['used_scores'][i]:.4f} - L2 Distance)")
                        st.caption(doc.page_content)


    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response_dict = answer_query(prompt, st.session_state.llm, st.session_state.vector_store)
            answer = response_dict["answer"]
            source_documents = response_dict["source_documents"]
            debug_info = response_dict["debug_info"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
            if source_documents and answer != "I Don't know. The most relevant information found is not strong enough." and "Error:" not in answer :
                 with st.expander("View Sources that contributed to the answer"):
                    for i, doc in enumerate(source_documents):
                        doc_score = debug_info.get("used_scores", [])[i] if i < len(debug_info.get("used_scores", [])) else "N/A"
                        score_display = f"{doc_score:.4f}" if isinstance(doc_score, float) else doc_score
                        st.write(f"**Source {i+1}: {doc.metadata.get('source', 'Unknown')}** (L2 Score: {score_display})")
                        st.caption(doc.page_content)
            elif debug_info.get("notes") == "Top document L2 score above threshold." or debug_info.get("notes") == "No documents retrieved at all.":
                 # Optionally show all initially retrieved documents if IDK was due to low scores
                 if source_documents: # These are the low_scoring_docs passed back
                    with st.expander("View Retrieved Documents (low relevance)"):
                        for i, doc in enumerate(source_documents):
                            doc_score = debug_info.get("retrieved_scores (L2 distance)", [])[i] if i < len(debug_info.get("retrieved_scores (L2 distance)", [])) else "N/A"
                            score_display = f"{doc_score:.4f}" if isinstance(doc_score, float) else doc_score
                            st.write(f"**Retrieved Document {i+1}: {doc.metadata.get('source', 'Unknown')}** (L2 Score: {score_display})")
                            st.caption(doc.page_content)


        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": source_documents, # Store for re-display if needed
            "debug_info": debug_info
            })
else:
    st.error("Chatbot components could not be initialized. Please check the logs and ensure all dependencies are correctly installed and configurations are set.")
    st.markdown("You might need to run `pip install -r requirements.txt` again or check for errors during the initial setup.")

st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses a Retrieval Augmented Generation (RAG) model "
    "to answer questions based on a set of provided documents. "
    "It leverages FAISS for efficient similarity search and "
    "a Hugging Face language model (google/flan-t5-small) for answer generation."
)
st.sidebar.header("Instructions to Run Locally")
st.sidebar.code("streamlit run app.py")
