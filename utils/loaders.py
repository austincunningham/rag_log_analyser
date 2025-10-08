from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
import os


def load_sop_files(directory: str):
    # Expanded extensions to ensure it catches common log files
    allowed_exts = ('.log', '.txt', '.out', '.err', '.access', '.csv', '.json', '.yaml', '.yml', '.md', '.asciidoc')

    docs = []
    file_count = 0
    total_lines = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(allowed_exts):
                path = os.path.join(root, file)
                file_count += 1
                print(f"📄 Processing file {file_count}: {file}")
                
                try:
                    # **CRITICAL CHANGE: Use LineLoaders or process the file line-by-line.**
                    # TextLoader is fine, but we will adjust the splitter in main.py to handle lines.
                    # For simplicity, we keep TextLoader but are mindful of the splitter change.
                    # A more robust solution for logs is to use a custom LineLoader or a JSONLoader for structured logs.
                    # Sticking to the current setup for minimal change, but logs will be large "documents".

                    # Alternative for better log line isolation:
                    with open(path, 'r', encoding='utf-8') as f:
                        # Create a document for EACH log line (or small group of lines)
                        # This maintains the context of individual log entries.
                        line_count = 0
                        for i, line in enumerate(f):
                            if line.strip():  # Skip empty lines
                                docs.append({
                                    "page_content": line.strip(),
                                    "metadata": {"source": path, "line_number": i + 1}
                                })
                                line_count += 1
                                total_lines += 1
                                
                                # Show progress for large files
                                if line_count % 10000 == 0:
                                    print(f"   📊 Processed {line_count:,} lines...")
                        
                        print(f"   ✅ Completed: {line_count:,} lines processed")

                except Exception as e:
                    print(f"❌ Error loading {path}: {e}")
    
    print(f"📊 Summary: Processed {file_count} files, {total_lines:,} total log entries")

    # Since we are returning a list of dictionaries, we must convert them to LangChain Document objects

    return [Document(**d) for d in docs]

# You can remove the unused imports if you implement the custom line-by-line logic above
# from langchain_community.document_loaders import DirectoryLoader