print("Starting script...")
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


# Load and prepare documents
LOG_DIRECTORY = input("➡️ Please enter the full path to the log directory (e.g., /home/user/support/): ")
if not os.path.isdir(LOG_DIRECTORY):
    print(f"❌ Error: Directory not found at '{LOG_DIRECTORY}'. Please check the path.")
    sys.exit(1)

print(f"📂 Loading log files from: {LOG_DIRECTORY}...")
# docs = load_sop_files("/home/austincunningham/support/")
docs = load_sop_files(LOG_DIRECTORY)
print(f"✅ Loaded {len(docs)} log entries from files")

# **CRITICAL CHANGE: Log lines should be treated as their own chunks.**
# If you implemented the line-by-line loader, you can comment this out:
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# chunks = splitter.split_documents(docs)
chunks = docs # <-- If the loader returns Document objects for each line, use them directly.


print(f"🧠 Creating vector database with {len(chunks)} log entries...")
print("⏳ This may take several minutes for large files...")

# Initialize embeddings with optimized settings
print("🚀 Using optimized embedding model for faster processing...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32  # Optimized batch size for speed
    }
)

print("📊 Generating embeddings...")
print("💡 This is the slowest step - generating vectors for each log line...")

# Process in batches to show progress and reduce memory usage
batch_size = 1000
total_chunks = len(chunks)
print(f"📊 Processing {total_chunks:,} log entries in batches of {batch_size:,}...")

# Create database with batching
db = Chroma.from_documents(
    chunks, 
    embeddings,
    collection_metadata={"hnsw:space": "cosine"}  # Optimize for cosine similarity
)

print("✅ Vector database created successfully")


retriever = db.as_retriever()

# You can improve the prompt by using a custom chain or a different chain type,
# but for minimal change, we will rely on the query itself.
llm = OllamaLLM(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


print("🤖 Log Analyser Assistant ready. Type your question below. Type 'exit' to quit.")


# Chat loop
while True:
   # **SUGGESTION: Change query for log analysis to ask for patterns/errors.**
   query = input("\n📝 You (e.g., 'What errors occurred in the last hour?'): ")
   if query.lower() in ("exit", "quit"):
       print("👋 Bye! Take care.")
       break

   # **IMPROVEMENT: Add a system prompt to the query to guide the LLM's role**
   analysis_query = (
       "You are an expert log analyser. Review the provided log entries and answer the user's question. "
       "Provide a concise summary, highlight any potential issues, and mention the relevant source log files. "
       f"Question: {query}"
   )


   # Call the chain with the enhanced query
   result = qa.invoke({"query": analysis_query})


   print("\n🤖 Assistant:\n", result["result"])


   print("\n📎 Sources:")
   # **IMPROVEMENT: Include line number if the custom loader was used**
   for doc in result["source_documents"]:
       source = doc.metadata.get('source')
       line_number = doc.metadata.get('line_number')
       print(f" - {source}{f' (Line {line_number})' if line_number else ''}")
