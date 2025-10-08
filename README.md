# RAG Log Analyzer

Transform your massive log files into an intelligent, queryable knowledge base using Retrieval-Augmented Generation (RAG). Ask questions in plain English and get intelligent insights from your logs.

## 🚀 Features

- **Natural Language Queries**: Ask questions like "What errors occurred in the last hour?"
- **Line-by-Line Processing**: Each log entry becomes a searchable document
- **Source Attribution**: Always know which file and line number provided the information
- **Multiple Format Support**: Works with `.log`, `.txt`, `.out`, `.err`, `.access`, `.csv`, `.json`, `.yaml`, `.yml`, `.md`, `.asciidoc`
- **Progress Tracking**: Real-time processing updates for large files
- **Local Processing**: No data leaves your machine - complete privacy

## 🎯 Use Cases

- **Incident Response**: "What caused the system outage at 3 PM?"
- **Performance Monitoring**: "Are there any slow database queries?"
- **Security Analysis**: "Find all failed login attempts"
- **Error Tracking**: "Show me all authentication failures"
- **Pattern Detection**: "What happened around 2:30 PM?"

## 📋 Prerequisites

### Install Ollama (for local LLM)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Mistral model
ollama pull mistral
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag_log_analyser
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analyzer**
```bash
python main.py
```

4. **Enter your log directory path**
```
➡️ Please enter the full path to the log directory (e.g., /home/user/support/): /path/to/your/logs
```

5. **Wait for processing** (shows progress for large files)
```
📄 Processing file 1: application.log
   📊 Processed 10,000 lines...
   ✅ Completed: 45,123 lines processed
🧠 Creating vector database with 45,123 log entries...
📊 Generating embeddings...
✅ Vector database created successfully
```

6. **Start asking questions**
```
🤖 Log Analyser Assistant ready. Type your question below.

📝 You: What errors occurred in the last hour?

🤖 Assistant:
Based on the log entries, I found several errors in the last hour:

1. **Database Connection Error** (2 occurrences)
   - Time: 14:23:15, 14:45:22
   - Error: "Connection timeout to database server"
   - Severity: CRITICAL

📎 Sources:
 - /var/log/application.log (Line 1247)
 - /var/log/application.log (Line 1253)
```

## 🏗️ Architecture

```
📁 rag_log_analyser/
├── main.py              # Main application
├── utils/
│   └── loaders.py       # Custom log file loader
├── requirements.txt     # Dependencies
├── README.md           # This file
└── LICENSE             # MIT License
```

### How It Works

1. **Document Processing**: Each log line becomes a separate document
2. **Vector Embeddings**: Convert text to numerical representations using HuggingFace
3. **Vector Storage**: Store embeddings in ChromaDB for fast retrieval
4. **Semantic Search**: Find relevant log entries based on meaning
5. **AI Analysis**: Mistral LLM provides intelligent insights and summaries

## 🔧 Configuration

### Embedding Models
The system uses optimized settings by default:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Batch Size**: 32 (optimized for speed)
- **Device**: CPU (consistent performance)

### LLM Models
- **Default**: Mistral (via Ollama)
- **Alternative**: Llama2, CodeLlama, or any Ollama model

## 📊 Performance

### Processing Times (Approximate)
- **Small files** (< 1MB): 1-2 minutes
- **Medium files** (1-10MB): 5-10 minutes  
- **Large files** (10-50MB): 15-30 minutes
- **Very large files** (50MB+): 30+ minutes

### Optimization Tips
- Use SSD storage for better I/O performance
- Ensure 8GB+ RAM for large datasets
- Consider sampling for initial testing

## 🎨 Example Queries

Try these with your logs:

```
"What errors occurred in the last hour?"
"Show me all database connection issues"
"Are there any performance bottlenecks?"
"What authentication problems happened today?"
"Find all 500 status codes"
"Show me memory usage warnings"
"What happened around 2:30 PM?"
"Are there any security violations?"
"Show me all failed transactions"
```

## 🛠️ Development

### Project Structure
- `main.py`: Main application with RAG pipeline
- `utils/loaders.py`: Custom log file processor
- `requirements.txt`: Python dependencies

### Key Components
- **Line-by-line processing**: Each log entry is a separate document
- **Progress tracking**: Real-time updates for large files
- **Metadata preservation**: Source file and line number tracking
- **Optimized embeddings**: Fast processing with good accuracy

## 🚨 Troubleshooting

### Common Issues

**"Directory not found" error**
- Ensure the path exists and is accessible
- Use absolute paths: `/home/user/logs/` not `~/logs/`

**"Ollama model not found"**
- Run `ollama pull mistral` to download the model
- Check Ollama is running: `ollama list`

**Slow processing**
- Large files take time - this is normal
- Consider using smaller test files first
- Ensure sufficient RAM (8GB+ recommended)

**Memory issues**
- Reduce batch size in embedding settings
- Process smaller files first
- Close other applications to free RAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -m 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the RAG framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [HuggingFace](https://huggingface.co/) for embedding models
- [ChromaDB](https://docs.trychroma.com/) for vector storage

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join the community discussions
- **Documentation**: Check the [blog post](BLOG_POST.md) for detailed technical explanations

---

*Built with ❤️ using Python, LangChain, and the power of RAG*
