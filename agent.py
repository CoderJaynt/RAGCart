import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import asyncio
import json
import uuid
import warnings
import logging

# Suppress high-noise warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from dotenv import load_dotenv

from scrapegraph_py import ScrapeGraphAI

from langchain_groq import ChatGroq

from langchain.agents import create_agent

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.documents import (
    Document
)

from langchain_core.tools import StructuredTool

from langchain_mcp_adapters.client import (
    MultiServerMCPClient
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv(".env")


class AgenticShoppingAI:

    def __init__(self, model_name="llama-3.3-70b-versatile"):

        # =================================================
        # API KEYS
        # =================================================

        self.GROQ_API_KEY = os.getenv(
            "GROQ_API_KEY"
        )

        self.SCRAPEGRAPH_API_KEY = os.getenv(
            "SCRAPEGRAPH_API_KEY"
        )

        # =================================================
        # LLM
        # =================================================

        self.llm = ChatGroq(
            model=model_name,
            temperature=0
        )

        # =================================================
        # SCRAPEGRAPH
        # =================================================

        self.scraper_tool = ScrapeGraphAI(
            api_key=self.SCRAPEGRAPH_API_KEY
        )

        # =================================================
        # EMBEDDINGS
        # =================================================

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # =================================================
        # VECTOR DATABASE
        # =================================================

        self.vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embedding_model
        )

        # =================================================
        # TEXT SPLITTER
        # =================================================

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
        )

        # =================================================
        # MCP TOOLS
        # =================================================
        
        # Check if running inside Streamlit to handle tool stability
        try:
            import streamlit as st
            if "mcp_tools" not in st.session_state:
                st.session_state["mcp_tools"] = asyncio.run(self.load_mcp_tools())
            self.mcp_tools = st.session_state["mcp_tools"]
        except ImportError:
            self.mcp_tools = asyncio.run(self.load_mcp_tools())

        print("\n===== MCP TOOLS LOADED =====\n")

        for tool_item in self.mcp_tools:
            print(tool_item.name)

        print("\n=====================\n")

        # =================================================
        # CUSTOM TOOLS
        # =================================================

        self.scrape_tool = (
            StructuredTool.from_function(
                name="scrape_tool",
                func=self._scrape_and_store_products_logic
            )
        )

        self.retrieve_tool = (
            StructuredTool.from_function(
                name="retrieve_tool",
                func=self._retrieve_product_knowledge_logic
            )
        )

        # =================================================
        # AGENT
        # =================================================

        self.agent = create_agent(
            model=self.llm,
            tools=self.mcp_tools + [
                self.scrape_tool,
                self.retrieve_tool
            ],
            system_prompt="""
            You are an autonomous shopping agent that MUST follow a strict multi-step RAG pipeline to answer the user's request. 
            
            STRICT PROTOCOL - DO NOT SKIP ANY STEPS:
            1. FIRST, you MUST use 'firecrawl_search' to search the web for the user's requested products and discover high-quality product URLs.
            2. SECOND, you MUST select at least 2-3 specific product URLs found from the search results, and call 'scrape_tool' for EACH of them (passing the product name as 'query' and the URL as 'url') to parse their details and save them to the local RAG database.
               * CRITICAL SCRAPING RULE: Avoid official manufacturer brand sites (like asus.com, msi.com, razer.com, dell.com, apple.com) because they aggressively block automated scrapers, resulting in HTTP 403 blocks. Instead, SELECT reviews, articles, or blog pages from sites like TechRadar, PCMag, Tom's Hardware, LaptopMag, Digital Trends, or CNET, which are open and scrape successfully!
            3. THIRD, you MUST call 'retrieve_tool' with the product keywords to fetch the successfully scraped details from the vector database.
            4. FINALLY, synthesize the retrieved information and provide a comprehensive comparison report to the user. You MUST cite and use the EXACT, real URLs provided in the retrieved chunks' metadata (labeled as '[Source: https://...]') for all product references. DO NOT invent, guess, or generalize any URLs under any circumstances.
            
            CRITICAL RULES:
            - You are FORBIDDEN from answering the user based on your pre-trained knowledge alone.
            - You are STRICTLY PROHIBITED from hallucinating, guessing, or generating any generic links. Only use the exact, full URLs supplied by the retrieved database sources.
            - If 'retrieve_tool' returns 0 documents, it means the scraping failed or was skipped! You must find alternative blog/review URLs from your search results, scrape them using 'scrape_tool', and then retrieve again.
            - When presenting your final answer, you MUST start with the exact phrase: "Based on my deep analysis of stored data..." to prove the RAG pipeline executed successfully.
            """
        )

    # =====================================================
    # LOAD MCP TOOLS
    # =====================================================

    async def load_mcp_tools(self):

        client = MultiServerMCPClient(

            {
                "firecrawl": {

                    "command": "cmd",

                    "args": [
                        "/c",
                        "npx",
                        "-y",
                        "firecrawl-mcp"
                    ],

                    "env": {
                        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
                    },

                    "transport": "stdio"
                }
            }
        )

        tools = await client.get_tools()

        # =================================================
        # SANITIZE TOOLS (Remove problematic regex patterns)
        # =================================================
        sanitized_tools = []

        def remove_pattern(obj):
            if isinstance(obj, dict):
                # Remove 'pattern' key as it causes validation issues in some LLMs (like Groq)
                if "pattern" in obj:
                    del obj["pattern"]
                for key, value in obj.items():
                    remove_pattern(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_pattern(item)

        for tool in tools:
            # ONLY keep essential search to save tokens
            if tool.name != "firecrawl_search":
                continue
                
            try:
                # Forcefully trim internal tool descriptions which are often huge
                if hasattr(tool, "description"):
                    tool.description = tool.description[:100] + "..."

                # Forcefully convert tool to a standard LangChain tool if it's not
                # and strip patterns from its arguments
                if hasattr(tool, "args"):
                    remove_pattern(tool.args)
                    # Limit number of properties in search to reduce schema size
                    if "properties" in tool.args:
                        # Keep only query and limit
                        essential_props = {}
                        if "query" in tool.args["properties"]: essential_props["query"] = tool.args["properties"]["query"]
                        if "limit" in tool.args["properties"]: essential_props["limit"] = tool.args["properties"]["limit"]
                        tool.args["properties"] = essential_props
                
                # Check for nested schema objects in Pydantic-based tools
                if hasattr(tool, "args_schema") and tool.args_schema:
                    try:
                        # Some versions of LangChain/Pydantic use .schema(), others .model_json_schema()
                        if hasattr(tool.args_schema, "schema"):
                            schema = tool.args_schema.schema()
                        else:
                            schema = tool.args_schema.model_json_schema()
                        remove_pattern(schema)
                    except:
                        pass

                sanitized_tools.append(tool)
            except Exception as e:
                print(f"Warning: Could not sanitize tool {tool.name}: {e}")
                sanitized_tools.append(tool)

        return sanitized_tools

    # =====================================================
    # STORE DOCUMENTS
    # =====================================================

    def store_documents(self, texts, url=""):

        docs = []

        for text in texts:

            chunks = self.text_splitter.split_text(
                text
            )

            for chunk in chunks:

                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": url} if url else {}
                    )
                )

        if docs:

            self.vector_db.add_documents(docs)

    # =====================================================
    # SCRAPE PRODUCTS LOGIC
    # =====================================================

    def _scrape_and_store_products_logic(
        self,
        query: str,
        url: str
    ) -> str:
        """Scrape product details from a specific review or blog URL and store it into the vector DB.
        
        Args:
            query: The search term or name of the product.
            url: The specific web URL of the product page or review article to scrape.
        """

        try:
            print(f"DEBUG: [scrape_tool] Invoked with query='{query}', url='{url}'")

            result = self.scraper_tool.extract(

                f"""
                Extract product information for:

                {query}

                Return:
                - product_name
                - price
                - rating
                - specifications
                - reviews_summary
                - pros
                - cons

                Return JSON only.
                """,

                url=url
            )

            # ============================================
            # HANDLE RESPONSE
            # ============================================

            if hasattr(result, "status"):

                if result.status != "success":

                    return f"""
                    ScrapeGraph Error:

                    {result.error}
                    """

                # Extract the actual JSON dictionary or list from the ScrapeGraphAI SDK result object
                if hasattr(result.data, "json_data") and result.data.json_data is not None:
                    parsed = result.data.json_data
                elif hasattr(result.data, "results") and result.data.results is not None:
                    parsed = result.data.results
                else:
                    parsed = result.data

            else:

                parsed = result

            texts = []

            # ============================================
            # HANDLE LIST OR DICT
            # ============================================

            if isinstance(parsed, list):

                for item in parsed:

                    text = f"""
                    Product Name:
                    {item.get('product_name', 'N/A')}

                    Price:
                    {item.get('price', 'N/A')}

                    Rating:
                    {item.get('rating', 'N/A')}

                    Specifications:
                    {item.get('specifications', 'N/A')}

                    Pros:
                    {item.get('pros', 'N/A')}

                    Cons:
                    {item.get('cons', 'N/A')}

                    Reviews:
                    {item.get('reviews_summary', 'N/A')}
                    """

                    texts.append(text)

            elif isinstance(parsed, dict):

                text = json.dumps(
                    parsed,
                    indent=2
                )

                texts.append(text)

            else:

                return "No valid product data found."

            # ============================================
            # STORE INTO VECTOR DB
            # ============================================

            self.store_documents(texts, url=url)
            print(f"DEBUG: Stored {len(texts)} documents in ChromaDB with source: {url}.")

            return f"""
            SUCCESS: Scraped and stored {len(texts)} product entries into the RAG Vector DB.
            The AI can now retrieve this specific data for comparison.
            """

        except Exception as e:

            return f"""
            Scraping Error:

            {str(e)}
            """

    # =====================================================
    # RETRIEVAL LOGIC
    # =====================================================

    def _retrieve_product_knowledge_logic(
        self,
        query: str
    ) -> str:
        """Retrieve product knowledge from vector database."""

        try:

            retriever = self.vector_db.as_retriever(
                search_kwargs={"k": 5}
            )

            docs = retriever.invoke(query)
            print(f"DEBUG: Retrieved {len(docs)} documents from ChromaDB for query: {query}")

            results = []

            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown Source")
                results.append(
                    f"--- Chunk {i+1} [Source: {source}] ---\n{doc.page_content}"
                )

            return "\n\n".join(results)

        except Exception as e:

            return f"""
            Retrieval Error:

            {str(e)}
            """

    # =====================================================
    # RUN AGENT
    # =====================================================

    async def run(self, query):

        try:

            response = await self.agent.ainvoke({

                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]

            })

            final_response = response[
                "messages"
            ][-1].content

            return final_response

        except Exception as e:

            return f"""
            Agent Error:

            {str(e)}
            """