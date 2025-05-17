import os
import glob
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain # Corrected import
from langchain.docstore.document import Document

# --- Configuration ---
TEXT_FILES_PATTERN = "*.txt"
EXCLUDE_FILES = ["requirements.txt"] # Files to exclude from loading
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_ID = 'google/flan-t5-small'
VECTOR_STORE_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Data Loading ---
def load_text_from_documents(directory="."):
    """
    Loads text from all .txt files in the specified directory, excluding certain files.
    Returns a list of Langchain Document objects.
    """
    all_documents = []
    txt_files = glob.glob(os.path.join(directory, TEXT_FILES_PATTERN))
    
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        if file_name in EXCLUDE_FILES:
            print(f"Skipping excluded file: {file_name}")
            continue
        try:
            # Using TextLoader to get Langchain Document objects directly
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Loaded document: {file_name}")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Add an empty document or handle as per requirement
            all_documents.append(Document(page_content="", metadata={"source": file_name, "error": str(e)}))
            
    if not all_documents:
        print("No documents were loaded. Please check the directory and file patterns.")
    return all_documents

# --- Document Processing ---
def split_documents(documents):
    """
    Splits the loaded documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
    return split_docs

# --- Embeddings and Vector Store ---
def get_embeddings():
    """Initializes and returns HuggingFaceEmbeddings."""
    # Specify device explicitly if needed, e.g., model_kwargs={'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Ensure CPU is used
    )
    return embeddings

def create_vector_store(documents, embeddings):
    """
    Creates a FAISS vector store from document chunks and embeddings, and saves it locally.
    """
    if not documents:
        print("No documents to create vector store from.")
        return None
    try:
        print("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store saved locally at: {VECTOR_STORE_PATH}")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# --- Retriever ---
def load_retriever(embeddings):
    """
    Loads the FAISS index from local storage and returns the vector store itself.
    The vector store can then be used to call methods like similarity_search_with_score.
    """
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store not found at {VECTOR_STORE_PATH}. Please create it first.")
        return None
    try:
        print(f"Loading FAISS vector store from: {VECTOR_STORE_PATH}")
        # Set allow_dangerous_deserialization=True if you trust the source of the FAISS index.
        # For HuggingFaceEmbeddings, it's generally safe.
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
        return vector_store  # Return the whole store, not just the retriever
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

# --- Language Model ---
def initialize_llm():
    """
    Initializes and returns a Hugging Face language model pipeline.
    """
    print(f"Initializing LLM: {LLM_MODEL_ID}")
    # Using HuggingFacePipeline for local model execution
    # You might need to install additional dependencies like `torch-sentencepiece` or `sentencepiece`
    # depending on the tokenizer used by the model.
    # For flan-t5, `transformers` should handle it if `sentencepiece` is installed.
    # If `torch` is installed for CPU, this should run on CPU.
    llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_ID,
        task="text2text-generation", # Appropriate task for Flan-T5
        pipeline_kwargs={"max_new_tokens": 200}, # Adjust as needed
        device=-1 # Explicitly set to CPU (-1 for CPU, 0, 1, etc. for GPU)
    )
    print("LLM initialized successfully.")
    return llm

# --- RAG Chain ---
def setup_rag_chain(llm, retriever):
    """
    Creates and returns a RetrievalQA chain.
    """
    if llm is None or retriever is None:
        print("LLM or Retriever not initialized. Cannot set up RAG chain.")
        return None
    if llm is None or retriever is None: # This function might be deprecated or adapted
        print("LLM or Retriever not initialized. Cannot set up RAG chain.")
        return None
    
    # This specific RetrievalQA chain might not be used directly by answer_query anymore,
    # but it's good to have a reference or for other potential uses.
    # The answer_query function will likely use load_qa_chain with filtered docs.
    # To ensure it can still be called if needed, let's check retriever type.
    # If retriever is a FAISS object, convert it to a retriever.
    from langchain_core.vectorstores import VectorStore
    if isinstance(retriever, VectorStore):
        retriever_for_chain = retriever.as_retriever()
    else:
        retriever_for_chain = retriever

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever_for_chain,
        return_source_documents=True
    )
    print("RAG chain (standard) set up successfully.")
    return rag_chain

# --- Query Function with IDK Logic ---
# Similarity score threshold (0 to 1, higher is better).
# FAISS.similarity_search_with_relevance_scores returns normalized scores.
# Using L2 distance with similarity_search_with_score. Lower scores are better.
# For normalized embeddings (like all-MiniLM-L6-v2), L2 distance `d` is related to cosine similarity `s`
# by d^2 = 2 - 2s.
# If desired cosine similarity `s_target = 0.7`, then d_target = sqrt(2 - 2*0.7) = sqrt(0.6) approx 0.7746.
# Documents with L2 distance > L2_DISTANCE_THRESHOLD will be considered not relevant enough.
L2_DISTANCE_THRESHOLD = 0.7746

def answer_query(query_text, llm, vector_store, threshold=L2_DISTANCE_THRESHOLD):
    """
    Takes a user query, LLM, and vector store.
    Retrieves documents with scores, applies a threshold for IDK,
    and then generates an answer using only high-scoring documents.
    Returns a structured dictionary with answer, sources, and debug info.
    """
    if llm is None or vector_store is None:
        return {
            "answer": "Error: LLM or vector store not initialized.",
            "source_documents": [],
            "debug_info": {"retrieved_scores": [], "used_scores": [], "threshold": threshold, "error": "LLM or vector_store missing"}
        }

    print(f"\nProcessing query: '{query_text}' with threshold: {threshold}")

    try:
        # Retrieve documents with L2 distance scores (lower is better).
        # k is the number of documents to retrieve initially.
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(query_text, k=3)
        
        retrieved_scores = [score for doc, score in retrieved_docs_with_scores]
        print(f"Retrieved {len(retrieved_docs_with_scores)} docs with L2 scores: {retrieved_scores}")

        if not retrieved_docs_with_scores:
            print("No documents retrieved.")
            return {
                "answer": "I Don't know. No relevant documents found.",
                "source_documents": [],
                "debug_info": {"retrieved_scores (L2 distance)": [], "used_scores (L2 distance)": [], "L2_distance_threshold": threshold, "notes": "No documents retrieved at all."}
            }

        # Filter documents based on the L2 distance threshold (lower is better)
        top_doc_score = retrieved_docs_with_scores[0][1]
        
        if top_doc_score > threshold: # If L2 distance of top doc is greater than threshold
            print(f"Top document L2 score ({top_doc_score:.4f}) is above threshold ({threshold:.4f}), indicating low relevance.")
            low_scoring_docs = [doc for doc, score in retrieved_docs_with_scores]
            return {
                "answer": "I Don't know. The most relevant information found is not strong enough.",
                "source_documents": low_scoring_docs,
                "debug_info": {"retrieved_scores (L2 distance)": retrieved_scores, "used_scores (L2 distance)": [], "L2_distance_threshold": threshold, "notes": "Top document L2 score above threshold."}
            }

        # Use documents with scores <= threshold
        high_scoring_docs = [doc for doc, score in retrieved_docs_with_scores if score <= threshold]
        used_scores = [score for doc, score in retrieved_docs_with_scores if score <= threshold]

        if not high_scoring_docs: # Should not happen if top_doc_score <= threshold, but as a safeguard
             print("No documents met the L2 distance threshold after filtering (safeguard).")
             return {
                "answer": "I Don't know. No documents met the relevance threshold.",
                "source_documents": [doc for doc, score in retrieved_docs_with_scores], # All initially retrieved for debugging
                "debug_info": {"retrieved_scores (L2 distance)": retrieved_scores, "used_scores (L2 distance)": [], "L2_distance_threshold": threshold, "notes": "Safeguard: No docs met L2 threshold after filtering."}
            }

        print(f"Using {len(high_scoring_docs)} documents (L2 score <= {threshold:.4f}) for QA.")

        # Use load_qa_chain with the filtered documents
        # The "stuff" chain type is simple and effective for a small number of documents.
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        result = qa_chain.invoke({"input_documents": high_scoring_docs, "question": query_text})
        
        answer = result.get("output_text", "No answer could be generated by the LLM.")
        
        return {
            "answer": answer,
            "source_documents": high_scoring_docs,
            "debug_info": {"retrieved_scores (L2 distance)": retrieved_scores, "used_scores (L2 distance)": used_scores, "L2_distance_threshold": threshold}
        }

    except Exception as e:
        print(f"Error during query processing: {e}")
        return {
            "answer": f"Error processing your query: {e}",
            "source_documents": [],
            "debug_info": {"retrieved_scores (L2 distance)": [], "used_scores (L2 distance)": [], "L2_distance_threshold": threshold, "error": str(e)}
        }

# --- Main execution (for testing and setup) ---
def main():
    """
    Main function to orchestrate the RAG setup and testing.
    """
    print("Starting RAG system setup...")
    
    # 1. Load text from documents
    print("\n--- Step 1: Loading Documents ---")
    documents = load_text_from_documents()
    if not documents:
        print("Halting execution as no documents were loaded.")
        return

    # 2. Split documents
    print("\n--- Step 2: Splitting Documents ---")
    split_docs = split_documents(documents)
    if not split_docs:
        print("Halting execution as no documents were split.")
        return
        
    # 3. Initialize Embeddings
    print("\n--- Step 3: Initializing Embeddings ---")
    embeddings = get_embeddings()

    # 4. Create or Load Vector Store
    # We need the vector_store object directly now, not just a retriever.
    print("\n--- Step 4: Loading/Creating Vector Store ---")
    embeddings = get_embeddings() # Ensure embeddings are initialized before loading

    # Attempt to load existing vector store
    vector_store = load_retriever(embeddings) # Renamed function, but it loads the FAISS vector_store
    
    if vector_store is None:
        print("Failed to load vector store. Attempting to create a new one...")
        if not split_docs: # Should have been created in step 2
            print("Documents not split, cannot create vector store. Halting.")
            return
        vector_store = create_vector_store(split_docs, embeddings)
        if vector_store is None:
            print("Failed to create new vector store. Halting.")
            return
    
    # 5. Initialize LLM
    print("\n--- Step 5: Initializing LLM ---")
    try:
        llm = initialize_llm()
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        # Ensure sentencepiece is mentioned if common error occurs
        if "sentencepiece" in str(e).lower():
             print("This might be due to missing 'sentencepiece'. Try: pip install sentencepiece")
        return
        
    # 6. Test Queries
    print("\n--- Step 6: Testing Queries ---")
    
    queries = [
        {"text": "What are the benefits of America's Choice Gold plan?", "type": "Relevant"},
        {"text": "What is the out-of-pocket maximum for America's Choice 5000 Bronze?", "type": "Relevant"},
        {"text": "What is the capital of France?", "type": "Irrelevant"},
        {"text": "Tell me about pre-authorization for medical services.", "type": "Relevant"},
        {"text": "Who is the current president of the United States?", "type": "Irrelevant"}
    ]

    if not os.path.exists(VECTOR_STORE_PATH):
         print(f"Warning: Vector store at {VECTOR_STORE_PATH} seems to be missing, " \
               "but was supposedly loaded or created. Queries might fail or yield poor results.")

    for query_info in queries:
        query_text = query_info["text"]
        query_type = query_info["type"]
        print(f"\n--- Testing Query ({query_type}): '{query_text}' ---")
        
        # The rag_chain from setup_rag_chain is not used here anymore.
        # We call answer_query with llm and vector_store.
        result_dict = answer_query(query_text, llm, vector_store)
        
        print(f"Query: {query_text}")
        print(f"Answer: {result_dict['answer']}")
        print(f"Debug Info: {result_dict['debug_info']}")
        # Optionally print source document content if needed for detailed debugging
        # print("Source Documents:")
        # for i, doc in enumerate(result_dict['source_documents']):
        #     print(f"  Source {i+1}: {doc.metadata.get('source', 'Unknown')} (Score: {result_dict['debug_info'].get('used_scores', [])[i]:.4f} if available)")
        #     print(f"    Content: {doc.page_content[:200]}...") # Print snippet

    print("\n--- RAG System Query Tests Complete ---")

if __name__ == "__main__":
    # Simplified sentencepiece check to avoid syntax errors.
    # LLM initialization in main() should handle critical model loading issues.
    try:
        import sentencepiece
        print("SentencePiece is installed and accessible.")
    except ImportError:
        print("SentencePiece not found. If 'google/flan-t5-small' (or other LLM) fails to load, you might need to install it: 'pip install sentencepiece'")
    
    main()
