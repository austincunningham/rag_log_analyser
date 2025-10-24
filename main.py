print("Starting script...")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Chroma.*deprecated.*")
warnings.filterwarnings("ignore", message=".*langchain-chroma.*")

from utils.loaders import load_sop_files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import os

print("_______________________________________________________________________________")
print(" _ __ __ _  __ _    | | ___   __ _      __ _ _ __   __ _| |_   _ ___  ___ _ __")
print("| '__/ _` |/ _` |___| |/ _ \\ / _` |___ / _` | '_ \\ / _` | | | | / __|/ _ | '__|")
print("| | | (_| | (_| |___| | (_) | (_| |___| (_| | | | | (_| | | |_| \\__ |  __| |")
print("|_|  \\__,_|\\__, |   |_|\\___/ \\__, |    \\__,_|_| |_|\\__,_|_|\\__, |___/\\___|_|")
print("           |___/              |___/                         |___/")
print("_______________________________________________________________________________")
print("                                                                                          ")
print("                                                                                          ")
print("                                                                                          ") 

# Load and prepare documents
try:
    LOG_DIRECTORY = input("-> Please enter the full path to the log directory (e.g., /home/user/support/): ")
    if not os.path.isdir(LOG_DIRECTORY):
        print(f"Error: Directory not found at '{LOG_DIRECTORY}'. Please check the path.")
        sys.exit(1)
except KeyboardInterrupt:
    print("\n\nExiting...")
    sys.exit(0)

print(f"Loading log files from: {LOG_DIRECTORY}...")
try:
    # docs = load_sop_files("/home/austincunningham/support/")
    docs = load_sop_files(LOG_DIRECTORY)
    print(f"Loaded {len(docs)} log entries from files")
except KeyboardInterrupt:
    print("\n\nFile loading interrupted by user. Exiting ...")
    sys.exit(0)


chunks = docs # <-- If the loader returns Document objects for each line, use them directly.


print(f"Creating vector database with {len(chunks)} log entries...")
print("This may take several minutes for large files...")

# Check if user wants to rebuild database
db_path = "./chroma_db"
try:
    if os.path.exists(db_path):
        rebuild_choice = input("Vector database already exists. Rebuild? (y/n): ").lower().strip()
        if rebuild_choice in ('y', 'yes'):
            print("Removing existing database...")
            import shutil
            shutil.rmtree(db_path)
            db_exists = False
        else:
            db_exists = True
    else:
        db_exists = False
except KeyboardInterrupt:
    print("\n\nDatabase setup interrupted by user. Exiting...")
    sys.exit(0)

# Initialize embeddings with optimized settings
print("Using optimized embedding model for faster processing...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32  # Optimized batch size for speed
    }
)

print("Generating embeddings...")
print("This is the slowest step - generating vectors for each log line...")

# Process in batches to show progress and reduce memory usage
batch_size = 1000
total_chunks = len(chunks)
print(f"Processing {total_chunks:,} log entries in batches of {batch_size:,}...")

# Create or load vector database
try:
    if db_exists:
        print("Found existing vector database, loading...")
        # Suppress ChromaDB deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
        print("Vector database loaded successfully")
    else:
        print("Creating new vector database...")
        # Suppress ChromaDB deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Create database with batching
            db = Chroma.from_documents(
                chunks, 
                embeddings,
                persist_directory=db_path,
                collection_metadata={"hnsw:space": "cosine"}  # Optimize for cosine similarity
            )
        print("Vector database created and saved")
except KeyboardInterrupt:
    print("\n\nVector database creation interrupted by user. Exiting...")
    sys.exit(0)


retriever = db.as_retriever()

# You can improve the prompt by using a custom chain or a different chain type,
# but for minimal change, we will rely on the query itself.
llm = OllamaLLM(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


print("Log Analyser Assistant ready. Type your question below. Type 'exit' to quit.")



try:
    while True:
       query = input("\nYou (e.g., 'What errors occurred in the last hour?'): ")
       if query.lower() in ("exit", "quit"):
           break

       analysis_query = (
           "You are an expert log analyser. Review the provided log entries and answer the user's question. "
           "Provide a concise summary, highlight any potential issues, and mention the relevant source log files. "
           f"Question: {query}"
       )

       try:

           result = qa.invoke({"query": analysis_query})

           print("\nAssistant:\n", result["result"])

           print("\nSources:")
           for doc in result["source_documents"]:
               source = doc.metadata.get('source')
               line_number = doc.metadata.get('line_number')
               print(f" - {source}{f' (Line {line_number})' if line_number else ''}")
       except KeyboardInterrupt:
           print("\n\n Query processing interrupted. You can ask another question or type 'exit' to quit.")
           continue

except KeyboardInterrupt:
    print("\n\nExiting...")
    sys.exit(0)
