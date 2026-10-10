# 🛒 RAGCart AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Framework" />
  <img src="https://img.shields.io/badge/LangChain-Enabled-1C3C3A?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-0052CC?style=for-the-badge&logo=google-cloud&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.3-F50057?style=for-the-badge&logo=dependabot&logoColor=white" alt="Groq Llama 3.3" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Firecrawl-MCP_Search-FF6D00?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Firecrawl MCP" />
  <img src="https://img.shields.io/badge/ScrapeGraphAI-Smart_Extract-2E7D32?style=for-the-badge&logo=openai&logoColor=white" alt="ScrapeGraphAI" />
  <img src="https://img.shields.io/badge/Hugging_Face-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
</p>

<p align="center">
  <b>Built by <a href="https://github.com/CoderJaynt">Jayant Yadav</a></b>
</p>

---

## 🌟 Introduction

**RAGCart AI** is an autonomous, state-of-the-art **Agentic RAG Shopping & Market Intelligence Assistant**. Unlike standard static RAG systems that read pre-saved files, RAGCart actively searches the live web on demand, extracts raw technical specs and pricing using visual-layout LLM scrapers, indexes them into a local vector database, and generates highly accurate, side-by-side product comparison reports grounded in verified citations.

The core AI engine is built as an autonomous agent using **LangChain** and **Groq's Llama 3.3 70B** model, communicating with search and parsing endpoints via stdio **Model Context Protocol (MCP)** bridges.

---

## 📐 Pipeline & System Architecture

RAGCart runs a highly synchronized, multi-step pipeline to answer user queries with 100% factual accuracy:

```
          [ User Query ]
                │
                ▼
        [ LangChain Agent ]
                │
                ▼
     [ Firecrawl MCP Search ]
                │
                ▼
       [ Find Product URLs ]
                │
                ▼
     [ ScrapeGraph Extraction ]
                │
                ▼
           [ Chunking ] (RecursiveCharacterTextSplitter)
                │
                ▼
          [ Embeddings ] (all-MiniLM-L6-v2)
                │
                ▼
      [ ChromaDB Storage ] (Local Vector DB)
                │
                ▼
          [ Retriever ] (Vector Similarity Match)
                │
                ▼
  [ LLM Reasoning & Comparison ] (Llama 3.3 via Groq)
                │
                ▼
     [ Final Recommendation ] ("Based on my deep analysis...")
```

---

## 🚀 Key Engineering & Architecture Highlights

### 1. ⚙️ Robust MCP Stdio Registry (Windows Compatible)
RAGCart implements LangChain's `MultiServerMCPClient` to bridge standard stdio transport protocols. To ensure native compatibility with Windows systems, the Firecrawl MCP discovery server is spun up using a robust cmd shell wrapper:
```python
"command": "cmd",
"args": ["/c", "npx", "-y", "firecrawl-mcp"]
```

### 2. 🛡️ Dynamic LLM Schema Sanitization & Token Optimization
Many open-source tool schemas contain complex regular expression validation filters. However, Groq's tool-calling engine throws validation failures when encountering JSON schemas containing `pattern` parameters. RAGCart's `agent.py` includes a custom recursive sanitizer that strips these constraints programmatically, ensuring 100% agent stability:
```python
def remove_pattern(obj):
    if isinstance(obj, dict):
        if "pattern" in obj:
            del obj["pattern"]
        for key, value in obj.items():
            remove_pattern(value)
```

### 3. 🧠 Smart Visual Page Extraction (ScrapeGraphAI)
Instead of regex-based web scraping, RAGCart invokes ScrapeGraphAI's LLM-based layout-aware parsing pipeline. The system passes a structured data schema to extract clean specifications, pros, cons, ratings, and reviews directly into dynamic data structures:
* Automatically parses dynamic responses from the SDK checking for `result.data.json_data` vs. `result.data.results`.
* Formats list objects into Markdown tables and key-value blocks before embedding.

### 4. 🔗 Authentic Source Metadata Citation
Every scraped block is indexed in **ChromaDB** alongside its source URL. When retrieving facts:
* Context chunks are fed to Llama 3.3 prefixed with their explicit source, e.g., `--- Chunk 1 [Source: https://...] ---`.
* A strict system prompt protocol forces the LLM to cite only the authentic, verified review links directly from the retrieved context, completely eliminating hallucinated URLs.

---

## 🔧 Backend Tool Specifications

The LangChain Agent has access to the following tool registry:

| Tool Identifier | Call Logic | Description | Arguments |
| :--- | :--- | :--- | :--- |
| **`firecrawl_search`** | **MCP Bridge** | Executes live Google search sweeps for review URLs. | `query` (string), `limit` (integer) |
| **`scrape_tool`** | **Custom Logic** | Extracts structured JSON specs using ScrapeGraphAI and indexes into Chroma. | `query` (string), `url` (string) |
| **`retrieve_tool`** | **Custom Logic** | Queries local Chroma vectors and returns the top 5 comparative documents. | `query` (string) |

---

## 🖥️ Streamlit Frontend Features

RAGCart is packaged inside a premium dark-themed Streamlit user interface featuring:
* **🤖 Live LLM Dropdown Selector**: Hot-swap between Llama 3.3 (70B), Llama 3.1 (8B), Mixtral (8x7B), or Gemma 2 (9B) directly from the sidebar. If you hit a Groq API Daily Token limit (Rate Limit 429 TPD), simply switch models to resume queries instantly with a fresh daily quota pool!
* **🔑 Hot-Swappable API Credentials**: Enter your Groq or Smartscrape API keys directly in the sidebar. Clicking **Save Keys** binds them to `os.environ`, flushes the cached agent state, and hot-reloads the application instantly.
* **📊 Live Vector DB Inspector**: Displays the total document collection count inside ChromaDB in real-time.
* **🗑️ One-Click Database Purge**: Purge indexed chunks in a single click to start fresh queries.
* **🔍 Direct Similarity Tester**: Input any query directly into the sidebar inspector to instantly view matching chunks and click their verified source links.
* **📥 Markdown Report Downloader**: Export synthesized agent reports instantly as fully-formatted markdown files.

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/tushar80rt/RAG_Cart.git
cd RAG_Cart
```

### 2. Configure a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Keys (`.env`)
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
SCRAPEGRAPH_API_KEY=your_scrapegraph_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### 5. Configure Firecrawl MCP Server (`mcp.json`)
Verify `mcp.json` settings:
```json
{
  "mcpServers": {
    "firecrawl-mcp": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "your_firecrawl_api_key"
      }
    }
  }
}
```

---

## 🏃 Running the Application

Launch the local interactive Streamlit web application:
```bash
streamlit run main.py
```

---

## 💎 Project Structure

```text
├── assets/                 # SVGs, assets, and base64 brand assets
├── chroma_db/              # Local vector database persistent directories
├── agent.py                # LangChain Agent, custom tools, and MCP pipelines
├── main.py                 # Streamlit frontend, UI styles, and session state
├── mcp.json                # MCP Server registry configuration
├── requirements.txt        # Python package dependencies
└── README.md               # Dynamic documentation
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

Developed with passion by **[Jayant Yadav](https://github.com/CoderJaynt)**. 

Hey! If you like this project, feel free to give it a ⭐️, fork the repository, or open an issue! Connect with me on [GitHub](https://github.com/CoderJaynt) to follow my work on agentic AI and advanced RAG pipelines.

