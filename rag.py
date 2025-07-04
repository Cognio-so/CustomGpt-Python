import os
import shutil
import asyncio
import time
import json
import base64
import re
from datetime import datetime  # Add datetime import at the top
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from urllib.parse import urlparse
import uuid
import httpx
import subprocess
import aiohttp
from dataclasses import dataclass, field
from typing import List, Tuple
import logging  # Add this import

from dotenv import load_dotenv

# --- Qdrant ---
from qdrant_client import QdrantClient, models as qdrant_models
from langchain_qdrant import QdrantVectorStore

# --- Langchain & OpenAI Core Components ---
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal



# Document Loaders & Transformers
from langchain_community.document_loaders import (
    PDFPlumberLoader, Docx2txtLoader, BSHTMLLoader, TextLoader, UnstructuredURLLoader
)
from langchain_community.document_transformers import Html2TextTransformer

# Add import for image processing
try:
    from PIL import Image
    from io import BytesIO
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    print("PIL not found. Install with: pip install pillow")

# Web Search (Tavily)
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    AsyncTavilyClient = None
    print("Tavily Python SDK not found. Web search will be disabled.")

# BM25 (Improved with fallback) ---
HYBRID_SEARCH_AVAILABLE = True
try:
    from langchain_community.retrievers import BM25Retriever
    from rank_bm25 import OkapiBM25
    HYBRID_SEARCH_AVAILABLE = True
    print("✅ BM25 package imported successfully")
except ImportError:
    # Implement our own simplified BM25 functionality
    print("⚠️ Standard rank_bm25 import failed. Implementing custom BM25 solution...")
    # Custom BM25 implementation
    import numpy as np
    from langchain_core.retrievers import BaseRetriever
    from typing import List, Dict, Any, Optional, Iterable, Callable
    from pydantic import Field, ConfigDict
    def default_preprocessing_func(text: str) -> List[str]:
        """Default preprocessing function that splits text on whitespace."""
        return text.lower().split()
    class BM25Okapi:
        """Simplified implementation of BM25Okapi when the rank_bm25 package is not available."""
        def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = []
            self.avgdl = 0
            self.N = 0
            if not self.corpus:
                return
            self.N = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.N
            # Calculate document frequencies
            for document in corpus:
                self.doc_len.append(len(document))
                freq = {}
                for word in document:
                    if word not in freq:
                        freq[word] = 0
                    freq[word] += 1
                self.doc_freqs.append(freq)
                # Update inverse document frequency
                for word, _ in freq.items():
                    if word not in self.idf:
                        self.idf[word] = 0
                    self.idf[word] += 1
            # Calculate inverse document frequency
            for word, freq in self.idf.items():
                self.idf[word] = np.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
        def get_scores(self, query):
            scores = [0] * self.N
            for q in query:
                if q not in self.idf:
                    continue
                q_idf = self.idf[q]
                for i, doc_freqs in enumerate(self.doc_freqs):
                    if q not in doc_freqs:
                        continue
                    doc_freq = doc_freqs[q]
                    doc_len = self.doc_len[i]
                    # BM25 scoring formula
                    numerator = q_idf * doc_freq * (self.k1 + 1)
                    denominator = doc_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    scores[i] += numerator / denominator
            return scores
        def get_top_n(self, query, documents, n=5):
            if not query or not documents or not self.N:
                return documents[:min(n, len(documents))]
            scores = self.get_scores(query)
            top_n = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:n]
            return [documents[i] for i in top_n]
    class SimpleBM25Retriever(BaseRetriever):
        """A simplified BM25 retriever implementation when rank_bm25 is not available."""
        vectorizer: Any = None
        docs: List[Document] = Field(default_factory=list, repr=False)
        k: int = 4
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
        )
        @classmethod
        def from_texts(
            cls,
            texts: Iterable[str],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[Iterable[str]] = None,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
        ) -> "SimpleBM25Retriever":
            """
            Create a SimpleBM25Retriever from a list of texts.
            Args:
                texts: A list of texts to vectorize.
                metadatas: A list of metadata dicts to associate with each text.
                ids: A list of ids to associate with each text.
                bm25_params: Parameters to pass to the BM25 vectorizer.
                preprocess_func: A function to preprocess each text before vectorization.
                **kwargs: Any other arguments to pass to the retriever.
            Returns:
                A SimpleBM25Retriever instance.
            """
            texts_list = list(texts)  # Convert iterable to list if needed
            texts_processed = [preprocess_func(t) for t in texts_list]
            bm25_params = bm25_params or {}
            # Create custom BM25Okapi vectorizer
            vectorizer = BM25Okapi(texts_processed, **bm25_params)
            # Create documents with metadata and ids
            documents = []
            metadatas = metadatas or ({} for _ in texts_list)
            if ids:
                documents = [
                    Document(page_content=t, metadata=m, id=i)
                    for t, m, i in zip(texts_list, metadatas, ids)
                ]
            else:
                documents = [
                    Document(page_content=t, metadata=m)
                    for t, m in zip(texts_list, metadatas)
                ]
            return cls(
                vectorizer=vectorizer,
                docs=documents,
                preprocess_func=preprocess_func,
                **kwargs
            )
        @classmethod
        def from_documents(
            cls,
            documents: Iterable[Document],
            *,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
        ) -> "SimpleBM25Retriever":
            """
            Create a SimpleBM25Retriever from a list of Documents.
            Args:
                documents: A list of Documents to vectorize.
                bm25_params: Parameters to pass to the BM25 vectorizer.
                preprocess_func: A function to preprocess each text before vectorization.
                **kwargs: Any other arguments to pass to the retriever.
            Returns:
                A SimpleBM25Retriever instance.
            """
            documents_list = list(documents)  # Convert iterable to list if needed
            # Extract texts, metadatas, and ids from documents
            texts = []
            metadatas = []
            ids = []
            for doc in documents_list:
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                if hasattr(doc, 'id') and doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
            return cls.from_texts(
                texts=texts,
                bm25_params=bm25_params,
                metadatas=metadatas,
                ids=ids,
                preprocess_func=preprocess_func,
                **kwargs,
            )
        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            """Get documents relevant to a query."""
            processed_query = self.preprocess_func(query)
            if self.vectorizer and processed_query:
                return self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
            return self.docs[:min(self.k, len(self.docs))]
        async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
            """Asynchronously get documents relevant to a query."""
            # Async implementation just calls the sync version for simplicity
            return self._get_relevant_documents(query, run_manager=run_manager)
    # Replace the standard BM25Retriever with our custom implementation
    BM25Retriever = SimpleBM25Retriever
    HYBRID_SEARCH_AVAILABLE = True
    print("✅ Custom BM25 implementation active - hybrid search enabled")

# Custom local imports
from storage import CloudflareR2Storage

try:
    from langchain_community.chat_message_histories import ChatMessageHistory # Updated import
except ImportError:
    from langchain.memory import ChatMessageHistory # Fallback for older versions, though the target is community

# Add imports for other providers
try:
    import anthropic  # for Claude
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Anthropic Python SDK not found. Claude models will be unavailable.")

try:
    import google.generativeai as genai  # for Gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google GenerativeAI SDK not found. Gemini models will be unavailable.")

try:
    from llama_cpp import Llama  # for Llama models
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("llama-cpp-python not found. Llama models will be unavailable.")

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq Python SDK not found. Llama models will use Groq as fallback.")

# OpenRouter (Optional)
try:
    # OpenRouter uses the same API format as OpenAI
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    print("OpenRouter will use OpenAI client for API calls.")

# Add new imports for tool calling
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticToolsParser
from typing import Literal

# Add this import around line 10-15 with other imports
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Vector params for OpenAI's text-embedding-3-small
QDRANT_VECTOR_PARAMS = qdrant_models.VectorParams(size=1536, distance=qdrant_models.Distance.COSINE)
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"

if os.name == 'nt':  # Windows
    pass

# Create tool schemas for detecting query type
class RAGQueryTool(BaseModel):
    """Process a general information query using RAG (Retrieval Augmented Generation)."""
    query_type: Literal["rag"] = Field(description="Indicates this is a general information query that should use RAG")
    explanation: str = Field(description="Explanation of why this query should use RAG")

class MCPServerQueryTool(BaseModel):
    """Process a query using an MCP (Model Context Protocol) server."""
    query_type: Literal["mcp"] = Field(description="Indicates this is a query that should use an MCP server")
    server_name: str = Field(description="Name of the MCP server to use if specified in the query")
    explanation: str = Field(description="Explanation of why this query should use MCP")

# Add this configuration class near the top of the file, after imports
@dataclass
class RAGConfiguration:
    """Configuration class to replace hardcoded values"""
    
    # Default URLs that can be overridden via environment variables
    default_qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    
    # LLM-based analysis settings
    use_llm_for_query_analysis: bool = True
    analysis_model: str = "gpt-4o-mini"
    analysis_temperature: float = 0.5
    
    # Dynamic keyword detection settings
    enable_dynamic_keyword_detection: bool = True
    
    # Web search similarity threshold (50% = 0.5)
    web_search_similarity_threshold: float = 0.4
    # Lower threshold for follow-up queries (30% = 0.3)
    follow_up_search_threshold: float = 0.3
    
    # Memory management - ENHANCED
    memory_expiry_minutes: int = 10  # Increased from 5 to 10 minutes
    max_conversation_turns: int = 10  # Increased from 6 to 10 turns
    
    # Context tracking - ENHANCED
    enforce_context_continuity: bool = True  # Ensure responses maintain topic continuity
    enable_conversational_memory: bool = True  # New: Enable enhanced conversational memory
    max_context_messages: int = 6  # Maximum messages to include in context
    
    # Regex patterns for URL detection (configurable)
    url_patterns: List[str] = field(default_factory=lambda: [
        r'https?://[^\s]+',  # Full URLs with http/https
        r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',  # URLs starting with www
        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'  # Domain names with paths
    ])
    
    # File extension mappings
    html_extensions: Tuple[str, ...] = (".html", ".htm")
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    
    # Timeout settings
    default_timeout: int = 30000
    
    @classmethod
    def from_env(cls) -> 'RAGConfiguration':
        """Create configuration from environment variables"""
        return cls()

class EnhancedRAG:
    """Enhanced RAG system with advanced context handling, MCP support, and conversational memory."""
    
    def __init__(
        self,
        gpt_id: str,
        r2_storage_client: CloudflareR2Storage,
        openai_api_key: Optional[str] = None,
        default_llm_model_name: str = "gpt-4o",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        temp_processing_path: str = "local_rag_data/temp_downloads",
        tavily_api_key: Optional[str] = None,
        default_system_prompt: Optional[str] = None,
        default_temperature: float = 0.2,
        default_use_hybrid_search: bool = True,
        initial_mcp_enabled_config: Optional[bool] = None,
        initial_mcp_schema_config: Optional[str] = None,
        config: Optional[RAGConfiguration] = None  # Add configuration parameter
    ):
        # Initialize configuration
        self.config = config or RAGConfiguration.from_env()
        
        # Initialize basic attributes
        self.gpt_id = gpt_id
        self.r2_storage_client = r2_storage_client
        self.openai_api_key = openai_api_key
        self.default_llm_model_name = default_llm_model_name
        self.default_system_prompt = default_system_prompt or "You are a helpful AI assistant with access to relevant documents and information."
        self.default_temperature = default_temperature
        self.default_use_hybrid_search = default_use_hybrid_search
        self.temp_processing_path = temp_processing_path
        
        # Track active MCP processes for cleanup
        self.active_mcp_processes: Dict[str, asyncio.subprocess.Process] = {}
        self.mcp_cleanup_lock = asyncio.Lock()

        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        self.max_tokens_llm = 32000 
        # IMPORTANT: Force hybrid search to always be True regardless of input setting
        self.default_use_hybrid_search = True
        print(f"✅ Hybrid search FORCE ENABLED for all queries regardless of config setting")

        # Store the initial full MCP configuration for this GPT
        self.gpt_mcp_enabled_config = initial_mcp_enabled_config
        self.gpt_mcp_full_schema_str = initial_mcp_schema_config # JSON string of all MCP servers for this GPT

        self.embeddings_model = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
        self.async_openai_client = AsyncOpenAI(
            api_key=self.openai_api_key, timeout=timeout_config, max_retries=1
        )

        # Use configuration for URLs instead of hardcoded values
        self.qdrant_url = qdrant_url or self.config.default_qdrant_url
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("Qdrant URL must be provided either as a parameter or via QDRANT_URL environment variable.")

        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=20.0)
        print(f"Qdrant client initialized for GPT '{self.gpt_id}' at URL: {self.qdrant_url}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.html_transformer = Html2TextTransformer()

        self.kb_collection_name = f"kb_{self.gpt_id}".replace("-", "_").lower()
        self.kb_retriever: Optional[BaseRetriever] = self._get_qdrant_retriever_sync(self.kb_collection_name)

        self.user_collection_retrievers: Dict[str, BaseRetriever] = {}
        self.user_memories: Dict[str, ChatMessageHistory] = {}

        self.tavily_client = None
        if self.tavily_api_key and TAVILY_AVAILABLE:
            try:
                self.tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
                print(f"✅ Tavily client initialized for GPT '{self.gpt_id}'")
            except Exception as e:
                print(f"❌ Error initializing Tavily client for GPT '{self.gpt_id}': {e}")
        elif not TAVILY_AVAILABLE:
             print(f"ℹ️ Tavily package not available for GPT '{self.gpt_id}'. Install with: pip install tavily-python")
        else:
            print(f"ℹ️ No Tavily API key provided for GPT '{self.gpt_id}'. Web search will be disabled.")
        
        self.anthropic_client = None
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if CLAUDE_AVAILABLE and self.claude_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)
            print(f"✅ Claude client initialized for GPT '{self.gpt_id}'")
        
        self.gemini_client = None
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if GEMINI_AVAILABLE and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai
            print(f"✅ Gemini client initialized for GPT '{self.gpt_id}'")
        
        self.llama_model = None
        if LLAMA_AVAILABLE:
            llama_model_path = os.getenv("LLAMA_MODEL_PATH")
            if llama_model_path and os.path.exists(llama_model_path):
                try:
                    self.llama_model = Llama(model_path=llama_model_path, verbose=False) # Reduce verbosity
                    print(f"✅ Llama model loaded for GPT '{self.gpt_id}'")
                except Exception as e:
                    print(f"❌ Error loading Llama model for GPT '{self.gpt_id}': {e}")

        self.groq_client = None
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if GROQ_AVAILABLE and self.groq_api_key:
            self.groq_client = AsyncGroq(api_key=self.groq_api_key)
            print(f"✅ Groq client initialized for GPT '{self.gpt_id}'")
        
        self.has_vision_capability = default_llm_model_name.lower() in [
            "gpt-4o", "gpt-4o-mini", "gpt-4-vision", # OpenAI
            "gemini-1.5-pro", "gemini-1.5-flash", # Google, using general names, specific API names handled in _process_image
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3.5-sonnet-20240620", # Anthropic
            "llava-v1.5-7b" # Example LLaVA model, actual name might vary based on Groq/OpenRouter
            # Add other vision models if directly supported without OpenRouter
        ] or "vision" in default_llm_model_name.lower() # General check

        normalized_model_name = default_llm_model_name.lower().replace("-", "").replace("_", "")
        self.is_gemini_model = "gemini" in normalized_model_name
        
        if self.has_vision_capability:
            print(f"✅ Vision capabilities may be available with model: {default_llm_model_name} for GPT '{self.gpt_id}'")
        else:
            print(f"⚠️ Model {default_llm_model_name} may not support vision. Image processing limited for GPT '{self.gpt_id}'.")
    
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        # Use configuration for OpenRouter URL
        self.openrouter_url = self.config.openrouter_base_url
        self.openrouter_client = None
        if OPENROUTER_AVAILABLE and self.openrouter_api_key:
            self.openrouter_client = AsyncOpenAI(
                api_key=self.openrouter_api_key, base_url=self.openrouter_url,
                timeout=timeout_config, max_retries=1
            )
            print(f"✅ OpenRouter client initialized for GPT '{self.gpt_id}'")
        else:
            print(f"ℹ️ OpenRouter API key not provided or client not available. OpenRouter disabled for GPT '{self.gpt_id}'.")

        # Initialize session info dictionary
        self.session_info: Dict[str, Dict[str, Any]] = {}

        # Initialize MCP related attributes with better validation
        self.mcp_enabled = False  # Default to False, will be enabled if validation passes
        self.gpt_mcp_full_schema_str = None
        self.mcp_servers_config = {}
        self.active_mcp_processes: Dict[str, asyncio.subprocess.Process] = {}

        print(f"Initializing MCP with enabled={initial_mcp_enabled_config}, schema_provided={bool(initial_mcp_schema_config)}")
        
        # Only proceed with MCP setup if it's explicitly enabled
        if initial_mcp_enabled_config:
            if initial_mcp_schema_config:
                try:
                    schema = json.loads(initial_mcp_schema_config)
                    if isinstance(schema, dict) and "mcpServers" in schema:
                        self.mcp_servers_config = schema["mcpServers"]
                        self.gpt_mcp_full_schema_str = initial_mcp_schema_config
                        self.mcp_enabled = True
                        print(f"✅ MCP Enabled with servers: {list(self.mcp_servers_config.keys())}")
                    else:
                        print("⚠️ Invalid MCP schema - missing mcpServers dictionary")
                except json.JSONDecodeError as e:
                    print(f"⚠️ Failed to parse MCP schema: {e}")
            else:
                print("⚠️ MCP enabled but no schema provided")

    def _get_user_qdrant_collection_name(self, session_id: str) -> str:
        safe_session_id = "".join(c if c.isalnum() else '_' for c in session_id)
        return f"user_{safe_session_id}".replace("-", "_").lower()

    def _ensure_qdrant_collection_exists_sync(self, collection_name: str):
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception as e:
            if "not found" in str(e).lower() or ("status_code=404" in str(e) if hasattr(e, "status_code") else False):
                print(f"Qdrant collection '{collection_name}' not found. Creating...")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=QDRANT_VECTOR_PARAMS
                )
                print(f"Qdrant collection '{collection_name}' created.")
            else:
                print(f"Error checking/creating Qdrant collection '{collection_name}': {e} (Type: {type(e)})")
                raise

    def _get_qdrant_retriever_sync(self, collection_name: str, search_k: int = 5) -> Optional[BaseRetriever]:
        self._ensure_qdrant_collection_exists_sync(collection_name)
        try:
            qdrant_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=self.embeddings_model,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )
            print(f"Initialized Qdrant retriever for collection: {collection_name}")
            return qdrant_store.as_retriever(search_kwargs={'k': search_k})
        except Exception as e:
            print(f"Failed to create Qdrant retriever for collection '{collection_name}': {e}")
            return None
            
    async def _get_user_retriever(self, session_id: str, search_k: int = 3) -> Optional[BaseRetriever]:
        collection_name = self._get_user_qdrant_collection_name(session_id)
        if session_id not in self.user_collection_retrievers or self.user_collection_retrievers.get(session_id) is None:
            await asyncio.to_thread(self._ensure_qdrant_collection_exists_sync, collection_name)
            self.user_collection_retrievers[session_id] = self._get_qdrant_retriever_sync(collection_name, search_k=search_k)
            if self.user_collection_retrievers[session_id]:
                print(f"User documents Qdrant retriever for session '{session_id}' (collection '{collection_name}') initialized.")
            else:
                print(f"Failed to initialize user documents Qdrant retriever for session '{session_id}'.")
        
        retriever = self.user_collection_retrievers.get(session_id)
        if retriever and hasattr(retriever, 'search_kwargs'):
            retriever.search_kwargs['k'] = search_k
        return retriever

    async def _get_user_memory(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.user_memories:
            self.user_memories[session_id] = ChatMessageHistory()
            print(f"Initialized new memory for session: {session_id}")
        return self.user_memories[session_id]

    async def _download_and_split_one_doc(self, r2_key_or_url: str) -> List[Document]:
        unique_suffix = uuid.uuid4().hex[:8]
        base_filename = os.path.basename(urlparse(r2_key_or_url).path) or f"doc_{hash(r2_key_or_url)}_{unique_suffix}"
        temp_file_path = os.path.join(self.temp_processing_path, f"{self.gpt_id}_{base_filename}")
        
        loaded_docs: List[Document] = []
        try:
            is_full_url = r2_key_or_url.startswith("http://") or r2_key_or_url.startswith("https://")
            r2_object_key_to_download = ""

            if is_full_url:
                parsed_url = urlparse(r2_key_or_url)
                is_our_r2_url = self.r2_storage_client.account_id and self.r2_storage_client.bucket_name and \
                                f"{self.r2_storage_client.bucket_name}.{self.r2_storage_client.account_id}.r2.cloudflarestorage.com" in parsed_url.netloc
                if is_our_r2_url:
                    r2_object_key_to_download = parsed_url.path.lstrip('/')
                else:
                    # Optimize web URL loading - use simpler approach for speed
                    try:
                        # Fast track for URL content - reduce processing overhead
                        async with aiohttp.ClientSession() as session:
                            async with session.get(r2_key_or_url, timeout=10) as response:
                                if response.status == 200:
                                    content = await response.text()
                                    doc = Document(
                                        page_content=content[:20000],  # Limit size for performance
                                        metadata={"source": r2_key_or_url, "source_type": "url"}
                                    )
                                    loaded_docs = [doc]
                                    return self.text_splitter.split_documents(loaded_docs)
                                else:
                                    print(f"Error fetching URL {r2_key_or_url}: {response.status}")
                                    return []
                    except Exception as e_url: 
                        print(f"Error fetching URL {r2_key_or_url}: {e_url}")
                        return []
            else:
                r2_object_key_to_download = r2_key_or_url
            
            if not loaded_docs and r2_object_key_to_download:
                # Optimize file download with more efficient processing
                download_task = asyncio.create_task(
                    asyncio.to_thread(self.r2_storage_client.download_file, r2_object_key_to_download, temp_file_path)
                )
                
                # Set a reasonable timeout
                try:
                    download_success = await asyncio.wait_for(download_task, timeout=15.0)
                    if not download_success: 
                        print(f"Failed R2 download: {r2_object_key_to_download}")
                        return []
                except asyncio.TimeoutError:
                    print(f"R2 download timeout for: {r2_object_key_to_download}")
                    return []

                _, ext = os.path.splitext(temp_file_path)
                ext = ext.lower()
                
                # Use configuration for image extensions instead of hardcoded list
                is_image = ext in self.config.image_extensions
                if is_image:
                    # Image processing - optimized
                    try:
                        with open(temp_file_path, 'rb') as img_file:
                            image_data = img_file.read()
                        
                        # Parallel processing of image with vision capabilities
                        image_content = await self._process_image_with_vision(image_data)
                        
                        if image_content:
                            doc = Document(
                                page_content=image_content,
                                metadata={
                                    "source": r2_key_or_url,
                                    "file_type": "image",
                                    "content_source": "vision_api"
                                }
                            )
                            loaded_docs = [doc]
                        else:
                            doc = Document(
                                page_content="[This is an image file that couldn't be processed. Please ask specific questions about its content.]",
                                metadata={"source": r2_key_or_url, "file_type": "image"}
                            )
                            loaded_docs = [doc]
                    except Exception as e_img:
                        print(f"Image processing failed: {e_img}")
                        doc = Document(
                            page_content="[This is an image file that could not be processed.]",
                            metadata={"source": r2_key_or_url, "file_type": "image"}
                        )
                        loaded_docs = [doc]
                else:
                    # Optimize document loading - use faster parallel processing
                    loader = None
                    try:
                        # Create an appropriate loader based on file extension
                        if ext == ".pdf": 
                            loader = PDFPlumberLoader(temp_file_path)
                        elif ext == ".docx": 
                            loader = Docx2txtLoader(temp_file_path)
                        elif ext in self.config.html_extensions: 
                            loader = BSHTMLLoader(temp_file_path, open_encoding='utf-8')
                        else: 
                            loader = TextLoader(temp_file_path, autodetect_encoding=True)
                        
                        # More efficient document loading using parallel processing
                        load_task = asyncio.create_task(asyncio.to_thread(loader.load))
                        try:
                            loaded_docs = await asyncio.wait_for(load_task, timeout=10.0)
                            # Special handling for HTML if needed
                            if ext in self.config.html_extensions and loaded_docs:
                                transform_task = asyncio.create_task(
                                    asyncio.to_thread(self.html_transformer.transform_documents, loaded_docs)
                                )
                                loaded_docs = await asyncio.wait_for(transform_task, timeout=5.0)
                        except asyncio.TimeoutError:
                            print(f"Document loading timeout for: {temp_file_path}")
                            return []
                    except Exception as e_load:
                        print(f"Error loading document: {e_load}")
                        return []
            
            if loaded_docs:
                for doc in loaded_docs:
                    doc.metadata["source"] = r2_key_or_url 
                
                # More efficient document splitting - optimize chunk size for faster processing
                split_task = asyncio.create_task(
                    asyncio.to_thread(self.text_splitter.split_documents, loaded_docs)
                )
                try:
                    return await asyncio.wait_for(split_task, timeout=10.0)
                except asyncio.TimeoutError:
                    print(f"Document splitting timeout for: {r2_key_or_url}")
                    # Return unsplit documents as fallback if splitting times out
                    return loaded_docs
            return []
        except Exception as e:
            print(f"Error processing source '{r2_key_or_url}': {e}")
            return []
        finally:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as e_del: print(f"Error deleting temp file {temp_file_path}: {e_del}")

    async def _process_image_with_vision(self, image_data: bytes) -> str:
        """Process an image using the user's chosen model with vision capabilities"""
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Original model name selected by the user
            user_selected_model_name_lower = self.default_llm_model_name.lower()
            
            # 1. Gemini models
            if "gemini" in user_selected_model_name_lower and GEMINI_AVAILABLE and self.gemini_client:
                print(f"Using {self.default_llm_model_name} for image processing via Gemini")
                gemini_api_name = "gemini-1.5-pro" # Default vision model for Gemini
                try:
                    if "flash" in user_selected_model_name_lower:
                        gemini_api_name = "gemini-1.5-flash"
                    # (No other specific Gemini model name checks needed, defaults to 1.5-pro for vision)

                    image_parts = [{"mime_type": "image/jpeg", "data": base64_image}]
                    prompt_text = "Describe the content of this image in detail, including any visible text."
                    
                    api_model_to_call = self.gemini_client.GenerativeModel(gemini_api_name)
                    response = await api_model_to_call.generate_content_async(contents=[prompt_text] + image_parts)
                    
                    if hasattr(response, "text") and response.text:
                        return f"Image Content ({gemini_api_name} Analysis):\n{response.text}"
                    else:
                        error_message_from_response = "No text content in response"
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                            error_message_from_response = f"Blocked: {getattr(response.prompt_feedback, 'block_reason_message', '') or response.prompt_feedback.block_reason}"
                        elif hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 'STOP':
                            error_message_from_response = f"Finished with reason: {response.candidates[0].finish_reason}"
                        raise Exception(f"Gemini Vision ({gemini_api_name}) processing issue: {error_message_from_response}")

                except Exception as e_gemini:
                    resolved_gemini_api_name = gemini_api_name if 'gemini_api_name' in locals() else 'N/A'
                    print(f"Error with Gemini Vision (input: {self.default_llm_model_name} -> attempted: {resolved_gemini_api_name}): {e_gemini}")
                    raise Exception(f"Gemini Vision processing failed: {e_gemini}")
            
            # 2. OpenAI models (GPT-4o, GPT-4o-mini, GPT-4-vision)
            elif "gpt-" in user_selected_model_name_lower:
                openai_model_to_call = self.default_llm_model_name # Default to user selected
                if user_selected_model_name_lower == "gpt-4o-mini":
                    openai_model_to_call = "gpt-4o" # Use gpt-4o for gpt-4o-mini's vision tasks
                    print(f"Using gpt-4o for image processing (selected: {self.default_llm_model_name})")
                else:
                    print(f"Using {self.default_llm_model_name} for image processing")
                
                try:
                    response = await self.async_openai_client.chat.completions.create(
                        model=openai_model_to_call,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the content of this image in detail, including any visible text."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }]
                    )
                    return f"Image Content ({openai_model_to_call} Analysis):\n{response.choices[0].message.content}"
                except Exception as e_openai:
                    print(f"Error with OpenAI Vision ({openai_model_to_call}): {e_openai}")
                    raise Exception(f"OpenAI Vision processing failed: {e_openai}")
            
            # 3. Claude models
            elif "claude" in user_selected_model_name_lower and CLAUDE_AVAILABLE and self.anthropic_client:
                print(f"Using {self.default_llm_model_name} for image processing")
                try:
                    claude_model_to_call = "claude-3.5-sonnet-20240620" # Default to Claude 3.5 Sonnet
                    if "opus" in user_selected_model_name_lower:
                        claude_model_to_call = "claude-3-opus-20240229"
                    # No need to check for "3-5" in sonnet/haiku explicitly, direct model names are better
                    elif "claude-3-sonnet" in user_selected_model_name_lower: # Catches "claude-3-sonnet-20240229"
                         claude_model_to_call = "claude-3-sonnet-20240229"
                    elif "claude-3-haiku" in user_selected_model_name_lower: # Catches "claude-3-haiku-20240307"
                         claude_model_to_call = "claude-3-haiku-20240307"
                    # Specific checks for 3.5 models to ensure correct IDs
                    elif "claude-3.5-sonnet" in user_selected_model_name_lower:
                        claude_model_to_call = "claude-3.5-sonnet-20240620"
                    elif "claude-3.5-haiku" in user_selected_model_name_lower:
                        claude_model_to_call = "claude-3.5-haiku-20240307" # Assuming this is the correct ID from Anthropic docs

                    response = await self.anthropic_client.messages.create(
                        model=claude_model_to_call,
                        messages=[{
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "Describe the content of this image in detail, including any visible text."},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                            ]
                        }]
                    )
                    return f"Image Content ({claude_model_to_call} Analysis):\n{response.content[0].text}"
                except Exception as e_claude:
                    print(f"Error with Claude Vision: {e_claude}")
                    raise Exception(f"Claude Vision processing failed: {e_claude}")
            
            # 4. Llama models (via Groq)
            elif "llama" in user_selected_model_name_lower and GROQ_AVAILABLE and self.groq_client:
                print(f"Processing Llama model {self.default_llm_model_name} for image via Groq")
                try:
                    groq_model_to_call = None
                    # More robust matching for Llama 4 Scout and Maverick
                    if "llama" in user_selected_model_name_lower and "4" in user_selected_model_name_lower and "scout" in user_selected_model_name_lower:
                        groq_model_to_call = "meta-llama/llama-4-scout-17b-16e-instruct"
                    elif "llama" in user_selected_model_name_lower and "4" in user_selected_model_name_lower and "maverick" in user_selected_model_name_lower:
                        groq_model_to_call = "meta-llama/llama-4-maverick-17b-128e-instruct"
                    elif "llava" in user_selected_model_name_lower: # For models like "llava-v1.5-7b"
                        groq_model_to_call = "llava-v1.5-7b-4096-preview"
                    elif "llama3" in user_selected_model_name_lower or "llama-3" in user_selected_model_name_lower:
                        # Llama 3 models on Groq do not support vision. This is an explicit failure.
                        raise Exception(f"The selected Llama 3 model ({self.default_llm_model_name}) does not support vision capabilities on Groq. Please choose a Llama 4 or LLaVA model for vision.")
                    else:
                        # Fallback for other Llama models not explicitly listed for vision
                        raise Exception(f"No configured vision-capable Llama model on Groq for '{self.default_llm_model_name}'. Supported for vision are Llama 4 Scout/Maverick and LLaVA.")

                    print(f"Attempting to use Groq vision model: {groq_model_to_call}")
                    
                    messages_for_groq = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the content of this image in detail, including any visible text."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                    if self.default_system_prompt:
                        messages_for_groq.insert(0, {"role": "system", "content": "You are an AI assistant that accurately describes images."})

                    response = await self.groq_client.chat.completions.create(
                        model=groq_model_to_call,
                        messages=messages_for_groq,
                        temperature=0.2,
                        stream=False
                    )
                    return f"Image Content ({groq_model_to_call} Analysis via Groq):\n{response.choices[0].message.content}"
                except Exception as e_llama_groq:
                    print(f"Error with Llama Vision through Groq (Model: {self.default_llm_model_name}): {e_llama_groq}")
                    raise Exception(f"Llama Vision processing failed: {e_llama_groq}")
            
            # If model doesn't match any of the known vision-capable types
            raise Exception(f"Model {self.default_llm_model_name} doesn't have a configured vision capability handler or required SDKs are not available.")
        except Exception as e:
            print(f"Error using Vision API: {e}")
            # Basic image properties fallback
            try:
                img = Image.open(BytesIO(image_data))
                width, height = img.size
                format_type = img.format
                mode = img.mode
                return f"[Image file: {width}x{height} {format_type} in {mode} mode. Vision processing failed with error: {str(e)}]"
            except Exception as e_img:
                return "[Image file detected but couldn't be processed. Vision API error: " + str(e) + "]"

    async def _index_documents_to_qdrant_batch(self, docs_to_index: List[Document], collection_name: str):
        if not docs_to_index: return

        try:
            await asyncio.to_thread(self._ensure_qdrant_collection_exists_sync, collection_name)
            qdrant_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                embedding=self.embeddings_model,
                content_payload_key=CONTENT_PAYLOAD_KEY,
                metadata_payload_key=METADATA_PAYLOAD_KEY
            )
            print(f"Adding {len(docs_to_index)} document splits to Qdrant collection '{collection_name}' via Langchain wrapper...")
            await asyncio.to_thread(
                qdrant_store.add_documents,
                documents=docs_to_index,
                batch_size=100
            )
            print(f"Successfully added/updated {len(docs_to_index)} splits in Qdrant collection '{collection_name}'.")
        except Exception as e:
            print(f"Error adding documents to Qdrant collection '{collection_name}' using Langchain wrapper: {e}")
            raise

    async def update_knowledge_base_from_r2(self, r2_keys_or_urls: List[str]):
        print(f"Updating KB for gpt_id '{self.gpt_id}' (collection '{self.kb_collection_name}') with {len(r2_keys_or_urls)} R2 documents...")
        
        processing_tasks = [self._download_and_split_one_doc(key_or_url) for key_or_url in r2_keys_or_urls]
        results_list_of_splits = await asyncio.gather(*processing_tasks)
        all_splits: List[Document] = [split for sublist in results_list_of_splits for split in sublist]

        if not all_splits:
            print(f"No content extracted from R2 sources for KB collection {self.kb_collection_name}.")
            if not self.kb_retriever:
                self.kb_retriever = self._get_qdrant_retriever_sync(self.kb_collection_name)
            return

        await self._index_documents_to_qdrant_batch(all_splits, self.kb_collection_name)
        self.kb_retriever = self._get_qdrant_retriever_sync(self.kb_collection_name)
        print(f"Knowledge Base for gpt_id '{self.gpt_id}' update process finished.")

    async def update_user_documents_from_r2(self, session_id: str, r2_keys_or_urls: List[str]):
        # Clear existing documents and retriever for this user session first
        print(f"Clearing existing user-specific context for session '{session_id}' before update...")
        await self.clear_user_session_context(session_id)

        user_collection_name = self._get_user_qdrant_collection_name(session_id)
        print(f"Updating user documents for session '{session_id}' (collection '{user_collection_name}') with {len(r2_keys_or_urls)} R2 docs...")
        
        processing_tasks = [self._download_and_split_one_doc(key_or_url) for key_or_url in r2_keys_or_urls]
        results_list_of_splits = await asyncio.gather(*processing_tasks)
        all_splits: List[Document] = [split for sublist in results_list_of_splits for split in sublist]

        if not all_splits:
            print(f"No content extracted from R2 sources for user collection {user_collection_name}.")
            # Ensure retriever is (re)initialized even if empty, after clearing
            self.user_collection_retrievers[session_id] = self._get_qdrant_retriever_sync(user_collection_name)
            return

        await self._index_documents_to_qdrant_batch(all_splits, user_collection_name)
        # Re-initialize the retriever for the session now that new documents are indexed
        self.user_collection_retrievers[session_id] = self._get_qdrant_retriever_sync(user_collection_name)
        print(f"User documents for session '{session_id}' update process finished.")

    async def clear_user_session_context(self, session_id: str):
        """Clear all user-specific context, documents, and active MCP processes for a session."""
        
        # Clean up any active MCP processes for this session
        await self._cleanup_mcp_processes(session_id)
        
        # Clear user documents
        user_collection_name = self._get_user_qdrant_collection_name(session_id)
        if self.qdrant_client:
            try:
                await asyncio.to_thread(
                    self.qdrant_client.delete_collection,
                    collection_name=user_collection_name
                )
                print(f"Deleted user collection: {user_collection_name}")
            except Exception as e:
                print(f"Error deleting user collection {user_collection_name}: {e}")
        
        # # Clear user retriever
        # if session_id in self.user_session_retrievers:
        #     del self.user_session_retrievers[session_id]
        
        # # Clear user memory
        # if session_id in self.user_memories:
        #     del self.user_memories[session_id]
        
        print(f"Cleared all context for user session: {session_id}")

    async def _cleanup_mcp_processes(self, session_id: str = None):
        """Clean up active MCP processes for a specific session or all sessions."""
        async with self.mcp_cleanup_lock:
            if session_id:
                # Clean up processes for specific session
                session_processes = [k for k in self.active_mcp_processes.keys() if k.startswith(session_id)]
                for process_key in session_processes:
                    process = self.active_mcp_processes.get(process_key)
                    if process and process.returncode is None:
                        try:
                            print(f"🧹 Terminating MCP process for session {session_id}")
                            process.terminate()
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            print(f"🔪 Force killing MCP process for session {session_id}")
                            process.kill()
                            await process.wait()
                        except Exception as e:
                            print(f"Error cleaning up MCP process: {e}")
                    del self.active_mcp_processes[process_key]
            else:
                # Clean up all MCP processes
                for process_key, process in list(self.active_mcp_processes.items()):
                    if process and process.returncode is None:
                        try:
                            process.terminate()
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                        except Exception as e:
                            print(f"Error cleaning up MCP process {process_key}: {e}")
                self.active_mcp_processes.clear()

    async def _get_retrieved_documents(
        self, 
        retriever: Optional[BaseRetriever], 
        query: str, 
        k_val: int = 3,
        is_hybrid_search_active: bool = True,
        is_user_doc: bool = False
    ) -> List[Document]:
        # Enhanced user document search - increase candidate pool for user docs
        candidate_k = k_val * 3 if is_user_doc else (k_val * 2 if is_hybrid_search_active and HYBRID_SEARCH_AVAILABLE else k_val)
        
        # Expanded candidate retrieval
        if hasattr(retriever, 'search_kwargs'):
            original_k = retriever.search_kwargs.get('k', k_val)
            retriever.search_kwargs['k'] = candidate_k
        
        # Vector retrieval
        docs = await retriever.ainvoke(query) if hasattr(retriever, 'ainvoke') else await asyncio.to_thread(retriever.invoke, query)
        
        # Stage 2: Apply BM25 re-ranking if hybrid search is active
        if is_hybrid_search_active and HYBRID_SEARCH_AVAILABLE and docs:
            print(f"Hybrid search active: Applying BM25 re-ranking to {len(docs)} vector search candidates")
            
            # BM25 re-ranking function
            def bm25_process(documents_for_bm25, q, target_k):
                bm25_ret = BM25Retriever.from_documents(documents_for_bm25, k=target_k)
                return bm25_ret.get_relevant_documents(q)
            
            # Execute BM25 re-ranking
            try:
                loop = asyncio.get_event_loop()
                bm25_reranked_docs = await loop.run_in_executor(None, bm25_process, docs, query, k_val)
                return bm25_reranked_docs
            except Exception as e:
                print(f"BM25 re-ranking error: {e}. Falling back to vector search results.")
                return docs[:k_val]
        else:
            # For user docs, return more results to provide deeper context
            return docs[:int(k_val * 1.5)] if is_user_doc else docs[:k_val]

    def _format_docs_for_llm_context(self, documents: List[Document], source_name: str) -> str:
        if not documents: return ""
        
        # Format the documents as before
        formatted_sections = []
        web_docs = []
        other_docs = []
        
        for doc in documents:
            source_type = doc.metadata.get("source_type", "")
            if source_type == "web_search" or "Web Search" in doc.metadata.get("source", ""):
                web_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # Process all documents without limits
        # Process web search documents first
        if web_docs:
            formatted_sections.append("## 🌐 WEB SEARCH RESULTS")
            for doc in web_docs:
                source = doc.metadata.get('source', source_name)
                title = doc.metadata.get('title', '')
                url = doc.metadata.get('url', '')
                
                # Create a more visually distinct header for each web document
                header = f"📰 **WEB SOURCE: {title}**"
                if url: header += f"\n🔗 **URL: {url}**"
                
                formatted_sections.append(f"{header}\n\n{doc.page_content}")
        
        # Process other documents
        if other_docs:
            if web_docs:  # Only add this separator if we have web docs
                formatted_sections.append("## 📚 KNOWLEDGE BASE & USER DOCUMENTS")
            
            for doc in other_docs:
                source = doc.metadata.get('source', source_name)
                score = f"Score: {doc.metadata.get('score', 'N/A'):.2f}" if 'score' in doc.metadata else ""
                title = doc.metadata.get('title', '')
                
                # Create a more visually distinct header for each document
                if "user" in source.lower():
                    header = f"📄 **USER DOCUMENT: {source}**"
                else:
                    header = f"📚 **KNOWLEDGE BASE: {source}**"
                    
                if title: header += f" - **{title}**"
                if score: header += f" - {score}"
                
                formatted_sections.append(f"{header}\n\n{doc.page_content}")
        
        return "\n\n---\n\n".join(formatted_sections)

    async def _get_web_search_docs(self, query: str, enable_web_search: bool, num_results: int = 3) -> List[Document]:
        if not enable_web_search or not self.tavily_client: 
            print(f"🌐 Web search is DISABLED for this query.")
            return []
        
        print(f"🌐 Web search is ENABLED. Searching web for: '{query}'")
        try:
            search_response = await self.tavily_client.search(
                query=query, 
                search_depth="advanced", # Changed from "basic" to "advanced" for more comprehensive search
                max_results=num_results,
                include_raw_content=True,
                include_domains=[]  # Can be customized to limit to specific domains
            )
            results = search_response.get("results", [])
            web_docs = []
            if results:
                print(f"🌐 Web search returned {len(results)} results")
                for i, res in enumerate(results):
                    content_text = res.get("raw_content") or res.get("content", "")
                    title = res.get("title", "N/A")
                    url = res.get("url", "N/A")
                    
                    if content_text:
                        print(f"🌐 Web result #{i+1}: '{title}' - {url[:60]}...")
                        web_docs.append(Document(
                            page_content=content_text[:4000],
                            metadata={
                                "source": f"Web Search: {title}",
                                "source_type": "web_search", 
                                "title": title, 
                                "url": url
                            }
                        ))
            return web_docs
        except Exception as e: 
            print(f"❌ Error during web search: {e}")
            return []
            
    async def _generate_llm_response(
        self, session_id: str, query: str, all_context_docs: List[Document],
        chat_history_messages: List[Dict[str, str]], llm_model_name_override: Optional[str],
        system_prompt_override: Optional[str], stream: bool = False
    ) -> Union[AsyncGenerator[str, None], str]:
        current_llm_model = llm_model_name_override or self.default_llm_model_name
        
        # Normalize model names for consistent matching
        normalized_model = current_llm_model.lower().strip()
        
        # Convert variations to canonical model names
        if "llama 4" in normalized_model or "llama-4" in normalized_model:
            current_llm_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        elif "llama" in normalized_model and "3" in normalized_model:
            current_llm_model = "llama-3.1-8b-instant"
        elif "gemini" in normalized_model and "flash" in normalized_model:
            current_llm_model = "gemini-flash-2.5"
        elif "gemini" in normalized_model and "pro" in normalized_model:
            current_llm_model = "gemini-pro-2.5"
        elif "claude" in normalized_model:
            current_llm_model = "claude-3.5-haiku-20240307"  # Use exact model ID with version
        elif normalized_model == "gpt-4o" or normalized_model == "gpt-4o-mini":
            current_llm_model = normalized_model  # Keep as is for OpenAI models
        
        current_system_prompt = system_prompt_override or self.default_system_prompt
        
        # Format context and query
        context_str = self._format_docs_for_llm_context(all_context_docs, "Retrieved Context")
        if not context_str.strip():
            context_str = "No relevant context could be found from any available source for this query. Please ensure documents are uploaded and relevant to your question."

        # Add current date and time to the query
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Updated user query message with stronger emphasis on accuracy and follow-up handling
        user_query_message_content = (
    f"📘 **CONTEXT:**\n{context_str}\n\n"
    f"🕒 **Current Time:** {formatted_time}\n\n"
    f"💬 **USER QUESTION:** {query}\n\n"
    f"📝 **CONVERSATION HISTORY:** {len(chat_history_messages)} previous messages\n\n"
    f"📌 **RESPONSE RULES:**\n"
    f"- Keep it **short and conversational**, like you're chatting with a friend.\n"
    f"- Use **only the given context**. Don't guess or assume.\n"
    f"- If the context isn't enough, say that clearly and briefly.\n"
    f"- Use *General Insight:* only when adding general knowledge.\n"
    f"- Give **longer or detailed replies only when web search is involved**.\n"
    f"- For follow-up questions, focus on providing NEW information that wasn't covered in previous answers. \n"
    f"- For 'tell me more' requests, provide additional details or examples not mentioned before.\n"
    f"- If recognizing a reference to something previously discussed, provide fresh information about that entity.\n"
    f"- For date questions, reply in format: DD/MM/YYYY (Day of week), and say **nothing else**.\n\n"
    f"✅ **STYLE & FORMAT:**\n"
    f"- 🏷️ Start with an emoji + quick headline\n"
    f"- 📋 Use bullets or short paras for clarity\n"
    f"- 💡 Emphasize main points\n"
    f"- 😊 Make it friendly and human\n"
    f"- 🤝 If it makes sense, end with a light follow-up to keep the chat going\n"
    f"- For short follow-up queries like 'make it shorter', 'in one line', etc., apply the formatting instruction to your previous response."
)


        messages = [{"role": "system", "content": current_system_prompt}]
        messages.extend(chat_history_messages)
        messages.append({"role": "user", "content": user_query_message_content})

        user_memory = await self._get_user_memory(session_id)
        
        # Check if it's an OpenRouter model (various model names supported by OpenRouter)
        use_openrouter = (self.openrouter_client is not None and 
                         (normalized_model.startswith("openai/") or 
                          normalized_model.startswith("anthropic/") or
                          normalized_model.startswith("meta-llama/") or
                          normalized_model.startswith("google/") or
                          normalized_model.startswith("mistral/") or
                          "openrouter" in normalized_model))

        # Special case: Handle router-engine and OpenRouter routing models
        if normalized_model == "router-engine" or normalized_model.startswith("openrouter/"):
            if normalized_model == "router-engine":
                print(f"Converting 'router-engine' to 'openrouter/auto' for OpenRouter routing")
                current_llm_model = "openrouter/auto"  # Use OpenRouter's auto-routing
            # If it already starts with "openrouter/", keep it as is
            use_openrouter = True

        if use_openrouter:
            # Implementation for OpenRouter models (stream and non-stream)
            if stream:
                async def openrouter_stream_generator():
                    full_response_content = ""
                    try:
                        # Add this validation check
                        if not self.openrouter_api_key or len(self.openrouter_api_key.strip()) < 10:
                            yield "Error: Valid OpenRouter API key is required. Please check your API key configuration."
                            return
                        response_stream = await self.openrouter_client.chat.completions.create(
                            model=current_llm_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                    except Exception as e_stream:
                        print(f"OpenRouter streaming error: {e_stream}")
                        yield f"I apologize, but I couldn't process your request successfully with OpenRouter. Please try asking in a different way."
                    finally:
                        # Store full human message but summarized AI message
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        # Summarize the AI response before storing it
                        summarized_response = await self._summarize_ai_message(full_response_content)
                        await asyncio.to_thread(user_memory.add_ai_message, summarized_response)
                return openrouter_stream_generator()
            else:
                response_content = ""
                try:
                    completion = await self.openrouter_client.chat.completions.create(
                        model=current_llm_model, 
                        messages=messages, 
                        temperature=self.default_temperature,
                        stream=False
                    )
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_nostream:
                    print(f"OpenRouter non-streaming error: {e_nostream}")
                    response_content = f"Error with OpenRouter: {str(e_nostream)}"
                
                # Store full human message but summarized AI message
                await asyncio.to_thread(user_memory.add_user_message, query)
                # Summarize the AI response before storing it
                summarized_response = await self._summarize_ai_message(response_content)
                await asyncio.to_thread(user_memory.add_ai_message, summarized_response)
                return response_content
        
        # GPT-4o or GPT-4o-mini models (OpenAI)
        if current_llm_model.startswith("gpt-"):
            # Implementation for OpenAI models (stream and non-stream)
            if stream:
                async def stream_generator():
                    full_response_content = ""
                    try:
                        response_stream = await self.async_openai_client.chat.completions.create(
                            model=current_llm_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                    except Exception as e_stream:
                        print(f"Error during streaming: {e_stream}")
                        yield f"I apologize, but I couldn't process your request successfully. Please try asking in a different way."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return stream_generator()
            else:
                response_content = ""
                try:
                    completion = await self.async_openai_client.chat.completions.create(
                        model=current_llm_model, messages=messages, temperature=self.default_temperature,
                        stream=False
                    )
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_nostream:
                    print(f"LLM non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Claude 3.5 Haiku
        elif current_llm_model.startswith("claude") and CLAUDE_AVAILABLE and self.anthropic_client:
            if stream:
                async def claude_stream_generator():
                    full_response_content = ""
                    try:
                        system_content = current_system_prompt
                        claude_messages = []
                        
                        for msg in chat_history_messages:
                            if msg["role"] != "system":
                                claude_messages.append(msg)
                        
                        claude_messages.append({"role": "user", "content": user_query_message_content})
                        
                        # Use the updated Claude model
                        response_stream = await self.anthropic_client.messages.create(
                            model="claude-3.5-haiku-20240307",  # Use the exact model ID including version
                            system=system_content,
                            messages=claude_messages,
                            stream=True,
                            max_tokens=4000
                        )
                        
                        async for chunk in response_stream:
                            if chunk.type == "content_block_delta" and chunk.delta.text:
                                content_piece = chunk.delta.text
                                full_response_content += content_piece
                                yield content_piece
                                
                    except Exception as e_stream:
                        print(f"Claude streaming error: {e_stream}")
                        yield "I apologize, but I couldn't process your request successfully. Please try again later."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return claude_stream_generator()
            else:
                # Non-streaming Claude implementation
                response_content = ""
                try:
                    system_content = current_system_prompt
                    claude_messages = []
                    
                    for msg in chat_history_messages:
                        if msg["role"] != "system":
                            claude_messages.append(msg)
                    
                    claude_messages.append({"role": "user", "content": user_query_message_content})
                    
                    # Use the updated Claude model
                    response = await self.anthropic_client.messages.create(
                        model="claude-3.5-haiku-20240307",  # Use the exact model ID including version
                        system=system_content,
                        messages=claude_messages,
                        max_tokens=4000
                    )
                    response_content = response.content[0].text
                except Exception as e_nostream:
                    print(f"Claude non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Gemini models (flash-2.5 and pro-2.5)
        elif current_llm_model.startswith("gemini") and GEMINI_AVAILABLE and self.gemini_client:
            if stream:
                async def gemini_stream_generator():
                    full_response_content = ""
                    try:
                        # Convert messages to Gemini format
                        gemini_messages = []
                        for msg in messages:
                            if msg["role"] == "system":
                                continue
                            elif msg["role"] == "user":
                                gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                            elif msg["role"] == "assistant":
                                gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                        
                        # Add system message to first user message if needed
                        if messages[0]["role"] == "system" and len(gemini_messages) > 0:
                            for i, msg in enumerate(gemini_messages):
                                if msg["role"] == "user" and (not msg["parts"] or not msg["parts"][0].get("text")):
                                    msg["parts"][0]["text"] = "Please provide information based on the context."
                                    break
                        
                        # Map to the specific Gemini model version with exact identifiers
                        gemini_model_name = current_llm_model
                        if current_llm_model == "gemini-flash-2.5":
                            gemini_model_name = "gemini-2.5-flash-preview-04-17"
                        elif current_llm_model == "gemini-pro-2.5":
                            gemini_model_name = "gemini-2.5-pro-preview-05-06"
                            
                        model = self.gemini_client.GenerativeModel(model_name=gemini_model_name)
                        
                        response_stream = await model.generate_content_async(
                            gemini_messages,
                            generation_config={"temperature": self.default_temperature},
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            if hasattr(chunk, "text"):
                                content_piece = chunk.text
                                if content_piece:
                                    full_response_content += content_piece
                                    yield content_piece
                        
                    except Exception as e_stream:
                        print(f"Gemini streaming error: {e_stream}")
                        if "429" in str(e_stream) and "quota" in str(e_stream).lower():
                            yield "I apologize, but the Gemini service is currently rate limited. The system will automatically fall back to GPT-4o."
                            # Fall back to GPT-4o silently
                            try:
                                response_stream = await self.async_openai_client.chat.completions.create(
                                    model="gpt-4o", 
                                    messages=messages, 
                                    temperature=self.default_temperature,
                                    stream=True
                                )
                                
                                async for chunk in response_stream:
                                    content_piece = chunk.choices[0].delta.content
                                    if content_piece:
                                        full_response_content += content_piece
                                        yield content_piece
                            except Exception as fallback_e:
                                print(f"Gemini fallback error: {fallback_e}")
                                yield "I apologize, but I couldn't process your request successfully. Please try again later."
                        else:
                            yield "I apologize, but I couldn't process your request successfully. Please try again later."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return gemini_stream_generator()
            else:
                # Non-streaming Gemini implementation
                response_content = ""
                try:
                    # Convert messages to Gemini format
                    gemini_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            continue
                        elif msg["role"] == "user":
                            gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                        elif msg["role"] == "assistant":
                            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                    
                    # Add system message to first user message if needed
                    if messages[0]["role"] == "system" and len(gemini_messages) > 0:
                        for i, msg in enumerate(gemini_messages):
                            if msg["role"] == "user" and (not msg["parts"] or not msg["parts"][0].get("text")):
                                msg["parts"][0]["text"] = "Please provide information based on the context."
                                break
                    
                    # Map to the specific Gemini model version with exact identifiers
                    gemini_model_name = current_llm_model
                    if current_llm_model == "gemini-flash-2.5":
                        gemini_model_name = "gemini-2.5-flash-preview-04-17"
                    elif current_llm_model == "gemini-pro-2.5":
                        gemini_model_name = "gemini-2.5-pro-preview-05-06"
                    
                    model = self.gemini_client.GenerativeModel(model_name=gemini_model_name)
                    response = await model.generate_content_async(
                        gemini_messages,
                        generation_config={"temperature": self.default_temperature}
                    )
                    
                    if hasattr(response, "text"):
                        response_content = response.text
                    else:
                        response_content = "Error: Could not generate response from Gemini."
                except Exception as e_nostream:
                    print(f"Gemini non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Llama models (Llama 3 and Llama 4 Scout via Groq)
        elif (current_llm_model.startswith("llama") or current_llm_model.startswith("meta-llama/")) and GROQ_AVAILABLE and self.groq_client:
            # Map to the correct Llama model with vision capabilities
            if "4" in current_llm_model.lower() or "llama-4" in current_llm_model.lower() or current_llm_model.startswith("meta-llama/llama-4"):
                # Use a model that actually exists in Groq as fallback
                groq_model = "llama-3.3-70b-versatile"  # Higher quality Llama model available on Groq
                print(f"Using Groq with llama-3.3-70b-versatile model (as fallback for Llama 4 Scout)")
            else:
                groq_model = "llama-3.1-8b-instant"  # Keep default for Llama 3
                print(f"Using Groq with llama-3.1-8b-instant model")
            
            if stream:
                async def groq_stream_generator():
                    full_response_content = ""
                    try:
                        groq_messages = [{"role": "system", "content": current_system_prompt}]
                        groq_messages.extend(chat_history_messages)
                        groq_messages.append({"role": "user", "content": user_query_message_content})
                        
                        response_stream = await self.groq_client.chat.completions.create(
                            model=groq_model,
                            messages=groq_messages,
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                
                    except Exception as e_stream:
                        print(f"Groq streaming error: {e_stream}")
                        yield "I apologize, but I couldn't process your request successfully. Please try again later."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return groq_stream_generator()
            else:
                # Non-streaming Groq implementation
                response_content = ""
                try:
                    groq_messages = [{"role": "system", "content": current_system_prompt}]
                    groq_messages.extend(chat_history_messages)
                    groq_messages.append({"role": "user", "content": user_query_message_content})
                    
                    completion = await self.groq_client.chat.completions.create(
                        model=groq_model,
                        messages=groq_messages,
                        temperature=self.default_temperature,
                        stream=False
                    )
                    
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_nostream:
                    print(f"Groq non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        # Fallback to GPT-4o when model not recognized
        else:
            print(f"Model {current_llm_model} not recognized. Falling back to gpt-4o.")
            fallback_model = "gpt-4o"
            
            # If streaming is requested, we must return a generator
            if stream:
                async def fallback_stream_generator():
                    full_response_content = ""
                    try:
                        completion = await self.async_openai_client.chat.completions.create(
                            model=fallback_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True  # Important: use streaming for streaming requests
                        )
                        
                        async for chunk in completion:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                full_response_content += content_piece
                                yield content_piece
                    except Exception as e_stream:
                        print(f"Fallback model streaming error: {e_stream}")
                        yield f"I apologize, but I couldn't process your request successfully. Please try asking in a different way."
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return fallback_stream_generator()
            else:
                # Non-streaming fallback implementation
                try:
                    completion = await self.async_openai_client.chat.completions.create(
                        model=fallback_model, 
                        messages=messages, 
                        temperature=self.default_temperature,
                        stream=False
                    )
                    response_content = completion.choices[0].message.content or ""
                except Exception as e_fallback:
                    print(f"Fallback model error: {e_fallback}")
                    response_content = "I apologize, but I couldn't process your request with the requested model. Please try again with a different model."
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content

    async def _get_formatted_chat_history(self, session_id: str) -> List[Dict[str,str]]:
        user_memory = await self._get_user_memory(session_id)
        history_messages = []
        for msg in user_memory.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history_messages.append({"role": role, "content": msg.content})
        return history_messages

    async def query_stream(
        self, session_id: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None,
        user_r2_document_keys: Optional[List[str]] = None, use_hybrid_search: Optional[bool] = None,
        llm_model_name: Optional[str] = None, system_prompt_override: Optional[str] = None,
        enable_web_search: Optional[bool] = False,
        mcp_enabled: Optional[bool] = None,
        mcp_schema: Optional[str] = None,
        api_keys: Optional[Dict[str, str]] = None,
        is_new_chat: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            print(f"[{session_id}] Query at {time.strftime('%Y-%m-%d %H:%M:%S')}: {query}")
            print(f"[{session_id}] 🧠 Starting enhanced context analysis...")
            
            # Check for MCP command first
            has_mcp_reference = bool(re.search(r'@([a-zA-Z0-9_-]+)', query))
            
            if has_mcp_reference:
                # Detect if this should use MCP
                detection_result = await self.detect_query_type(query)
                print(f"[{session_id}] MCP Detection Result:", detection_result)
                
                if detection_result["query_type"] == "mcp":
                    print(f"[{session_id}] 🚀 Using MCP server: {detection_result['server_name']}")
                    async for chunk in self._handle_mcp_request(
                        query=query,
                        selected_server_config_str=self.gpt_mcp_full_schema_str,
                        chat_history=chat_history or [],
                        api_keys_for_mcp=api_keys,
                        detected_server_name=detection_result["server_name"]
                    ):
                        yield chunk
                    return  # Important: return here to prevent fallback to RAG
                else:
                    # If MCP reference was found but MCP is disabled or server not found
                    error_msg = f"Cannot process MCP command: {detection_result['explanation']}"
                    print(f"[{session_id}] ⚠️ {error_msg}")
                    yield {"type": "error", "data": error_msg}
                    return  # Return here instead of falling back to RAG
            
            # Only proceed with RAG if no MCP reference was found
            print(f"[{session_id}] 📊 Processing as RAG query")
            async for chunk in self._process_rag_query(
                session_id, query, chat_history, user_r2_document_keys,
                use_hybrid_search, llm_model_name, system_prompt_override,
                enable_web_search, is_new_chat
            ):
                yield chunk

        except Exception as e:
            error_msg = f"Error in query_stream: {str(e)}"
            print(f"[{session_id}] ❌ {error_msg}")
            yield {"type": "error", "data": error_msg}

    async def _handle_mcp_request(self, query: str, selected_server_config_str: str, chat_history: List[Dict[str, str]], api_keys_for_mcp: Optional[Dict[str, str]] = None, detected_server_name: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            if not self.mcp_enabled:
                yield {"type": "error", "data": "MCP is disabled"}
                return
            
            if not detected_server_name or detected_server_name not in self.mcp_servers_config:
                yield {"type": "error", "data": f"MCP server '{detected_server_name}' not found"}
                return
            
            server_config = self.mcp_servers_config[detected_server_name]
            print(f"Executing MCP server '{detected_server_name}' with config: {server_config}")
            
            # Execute the server
            buffer = ""
            async for response in self._execute_mcp_server_properly(
                server_name=detected_server_name,
                server_config=server_config,
                query=query,
                chat_history=chat_history
            ):
                # Format the response in markdown
                if isinstance(response, str):
                    # If response is code block, preserve it
                    if response.startswith('```') and response.endswith('```'):
                        formatted_response = response
                    else:
                        # Add proper line breaks and ensure markdown formatting
                        lines = response.split('\n')
                        formatted_lines = []
                        for line in lines:
                            line = line.strip()
                            if line:
                                # Preserve existing markdown formatting if present
                                if not (line.startswith('#') or line.startswith('*') or line.startswith('```') or line.startswith('>')):
                                    line = line.replace('**', '').replace('__', '')  # Remove any malformed markdown
                                formatted_lines.append(line)
                        formatted_response = '\n\n'.join(formatted_lines)
                    
                    yield {"type": "content", "data": formatted_response}
                
        except Exception as e:
            print(f"❌ Error in MCP request handling: {str(e)}")
            yield {"type": "error", "data": f"MCP execution error: {str(e)}"}

    def _detect_mcp_server_from_query(self, query: str, available_servers: Dict[str, Any]) -> Optional[str]:
        """Detect which MCP server to use based on the query content."""
        query_lower = query.lower()
        
        # ONLY check if any server name is explicitly mentioned in the query
        for server_name in available_servers.keys():
            if server_name.lower() in query_lower:
                print(f"MCP server '{server_name}' explicitly mentioned in query.")
                return server_name
        
        # Return None if no exact server name is mentioned
        # This will trigger a fallback to RAG in the calling method
        return None

    async def _generate_fallback_response(self, query: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Generate a fallback response when MCP fails.
        Uses web search if the query would benefit from it, otherwise uses basic LLM.
        """
        try:
            # Use a random session ID for the fallback response
            fallback_session_id = f"fallback_{uuid.uuid4().hex[:8]}"
            
            # Convert chat history to the format expected by _generate_llm_response
            formatted_chat_history = chat_history.copy() if chat_history else []
            
            # First analyze if web search would be beneficial for this query
            web_search_analysis = await self._analyze_web_search_necessity(query, formatted_chat_history)
            should_use_web_search = web_search_analysis.get("should_use_web_search", False)
            
            retrieved_docs = []
            
            # If web search is recommended and available, perform it
            if should_use_web_search and self.tavily_client:
                print(f"MCP fallback: Using web search for query")
                web_docs = await self._get_web_search_docs(query, True, num_results=3)
                if web_docs:
                    retrieved_docs.extend(web_docs)
                    print(f"MCP fallback: Retrieved {len(web_docs)} web search documents")
            
            # Generate response based on retrieved documents or with empty context if none found
            llm_stream_generator = await self._generate_llm_response(
                fallback_session_id, query, retrieved_docs, formatted_chat_history,
                None, None, stream=True
            )
            
            # Stream the response
            async for content_chunk in llm_stream_generator:
                yield content_chunk
                
        except Exception as e:
            print(f"Error generating fallback response: {e}")
            yield f"Sorry, I couldn't process your request. Please try again with different wording."

    async def _execute_mcp_server_properly(self, server_name: str, server_config: Dict[str, Any], query: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Actually execute the MCP server command and stream the real response.
           server_config is expected to have 'command', 'args', and 'env'.
        """
        try:
            command = server_config.get("command")
            args = server_config.get("args", [])
            env_vars = server_config.get("env", {}) 
            
            if not command:
                raise ValueError(f"No command specified for MCP server '{server_name}'")
            
            print(f"Executing MCP server '{server_name}' with command: '{command}'")
            if env_vars:
                print(f"  with custom environment variables: {list(env_vars.keys())}")

            async for chunk in self._execute_generic_mcp_server(command, args, env_vars, query, chat_history):
                yield chunk
                
        except Exception as e:
            print(f"Error in _execute_mcp_server_properly for '{server_name}': {e}")
            yield f"Error executing MCP server '{server_name}': {str(e)}"

    def _is_valid_json_line(self, line: str) -> bool:
        """Check if a line contains valid JSON"""
        try:
            json.loads(line.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    async def _read_json_response(self, process_stdout, max_lines: int = 10, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Read JSON response from MCP server stdout, skipping informational messages.
        
        Args:
            process_stdout: The stdout stream of the MCP process
            max_lines: Maximum number of lines to read before giving up
            timeout: Timeout in seconds
            
        Returns:
            Parsed JSON response or None if no valid JSON found
        """
        lines_read = 0
        
        try:
            while lines_read < max_lines:
                try:
                    # Read line with timeout
                    line_bytes = await asyncio.wait_for(process_stdout.readline(), timeout=timeout)
                    if not line_bytes:
                        break
                        
                    line = line_bytes.decode().strip()
                    lines_read += 1
                    
                    if not line:
                        continue
                        
                    # Check if this line contains valid JSON
                    if self._is_valid_json_line(line):
                        try:
                            return json.loads(line)
                        except json.JSONDecodeError:
                            continue
                    else:
                        # This is an informational message, log it and continue
                        print(f"MCP server info: {line}")
                        continue
                        
                except asyncio.TimeoutError:
                    print(f"Timeout reading line {lines_read + 1} from MCP server")
                    break
                except Exception as e:
                    print(f"Error reading line {lines_read + 1}: {e}")
                    break
                    
        except Exception as e:
            print(f"Error in _read_json_response: {e}")
            
        return None

    async def _execute_generic_mcp_server(self, command: str, args: List[str], env_vars: Dict[str, str], query: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Enhanced generic execution of MCP servers with better process management and robust JSON parsing."""
        
        # Create process key for tracking
        process_key = f"{command}_{int(time.time())}"
        process = None
        tool_name = None  # Initialize tool_name here to avoid undefined variable error
        detected_urls = []  # Initialize detected_urls here as well
        
        try:
            print(f"Executing MCP server '{command.split('/')[-1]}' with command: '{command}'")
            
            # Check if command exists
            if os.name == 'nt':  # Windows
                result = subprocess.run(['where', command], capture_output=True, text=True)
                if result.returncode != 0:
                    yield f"Command '{command}' not found on Windows"
                    return
                actual_command = result.stdout.strip().split('\n')[0]
                print(f"Found {command} at: {actual_command}")
                
                # Check if the command is a batch file (.bat or .cmd)
                if actual_command.lower().endswith(('.bat', '.cmd')) or not actual_command.lower().endswith('.exe'):
                    # Use cmd /c to execute batch files
                    full_command = ['cmd', '/c', actual_command] + args
                else:
                    full_command = [actual_command] + args
            else:  # Unix-like
                result = subprocess.run(['which', command], capture_output=True, text=True)
                if result.returncode != 0:
                    yield f"Command '{command}' not found"
                    return
                actual_command = result.stdout.strip()
                full_command = [actual_command] + args
                
            print(f"Executing command: {' '.join(full_command)}")
            
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, **env_vars}
            )
            
            # Track the process for potential cleanup
            async with self.mcp_cleanup_lock:
                self.active_mcp_processes[process_key] = process

            # MCP JSON-RPC Communication Protocol
            # Step 1: Initialize the MCP session
            initialize_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        }
                    },
                    "clientInfo": {
                        "name": "RAG-MCP-Client",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Send initialize request
            if process.stdin:
                init_json = json.dumps(initialize_request) + "\n"
                process.stdin.write(init_json.encode())
                await process.stdin.drain()
                
                # Wait for initialize response with robust JSON parsing
                init_response = await self._read_json_response(process.stdout, max_lines=10, timeout=10.0)
                if init_response:
                    print(f"MCP server initialized: {init_response}")
                else:
                    print("Warning: No valid initialize response received, continuing anyway...")
                
                # Step 2: Send initialized notification
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                
                init_notif_json = json.dumps(initialized_notification) + "\n"
                process.stdin.write(init_notif_json.encode())
                await process.stdin.drain()
                
                # Step 3: List available tools
                list_tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list"
                }
                
                list_tools_json = json.dumps(list_tools_request) + "\n"
                process.stdin.write(list_tools_json.encode())
                await process.stdin.drain()
                
                # Wait for tools list response with robust JSON parsing
                tools_response = await self._read_json_response(process.stdout, max_lines=10, timeout=10.0)
                available_tools = []
                if tools_response and "result" in tools_response and "tools" in tools_response["result"]:
                    available_tools = tools_response["result"]["tools"]
                    print(f"Available tools: {[tool.get('name', 'unknown') for tool in available_tools]}")
                else:
                    print("Warning: No valid tools response received")
                
                # Step 4: Call the appropriate tool with the query
                if available_tools:
                    # Use the first available tool (no hardcoding)
                    tool_to_use = available_tools[0]
                    tool_name = tool_to_use.get("name")
                    
                    if not tool_name:
                        yield "Error: No valid tool name found from MCP server"
                        return
                    
                    # Convert chat history and current query into messages format
                    messages = []
                    
                    # Add chat history
                    if chat_history:
                        for msg in chat_history:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            if role and content:
                                messages.append({
                                    "role": role,
                                    "content": content
                                })
                    
                    # Add current query as user message
                    messages.append({
                        "role": "user", 
                        "content": query
                    })
                    
                    # Enhanced tool call arguments construction with URL detection
                    tool_schema = tool_to_use.get("inputSchema", {})
                    tool_properties = tool_schema.get("properties", {})
                    required_params = tool_schema.get("required", [])
                    
                    tool_arguments = {}
                    
                    # Extract URLs from query for URL-based tools
                    detected_urls = self._extract_urls_from_query(query)
                    
                    # Priority 1: Handle URL parameter if tool expects it
                    if "url" in tool_properties:
                        if detected_urls:
                            tool_arguments["url"] = detected_urls[0]  # Use first detected URL
                            print(f"🔗 Using detected URL: {detected_urls[0]} for tool '{tool_name}'")
                        else:
                            # Try to extract domain from query for simple domain mentions
                            words = query.lower().split()
                            for word in words:
                                if ('.' in word and not word.startswith('@') and 
                                    any(tld in word for tld in ['.com', '.org', '.net', '.edu', '.gov', '.io', '.co'])):
                                    url = f"https://{word.strip('.,!?;')}"
                                    tool_arguments["url"] = url
                                    print(f"🔗 Constructed URL from domain: {url} for tool '{tool_name}'")
                                    break
                            
                            # If still no URL found but tool requires it, use the query as potential URL
                            if "url" not in tool_arguments and "url" in required_params:
                                # Last resort: try to find anything that looks like a domain
                                import re
                                domain_pattern = r'\b([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
                                domain_matches = re.findall(domain_pattern, query)
                                if domain_matches:
                                    constructed_url = f"https://{domain_matches[0].rstrip('.')}"
                                    tool_arguments["url"] = constructed_url
                                    print(f"🔗 Last resort URL construction: {constructed_url} for tool '{tool_name}'")
                    
                    # Priority 2: Handle other common parameters
                    if "messages" in tool_properties and "messages" not in tool_arguments:
                        tool_arguments["messages"] = messages
                    elif "query" in tool_properties and "query" not in tool_arguments:
                        # For query parameter, use the original query or cleaned query without URLs if URL was extracted
                        if detected_urls and "url" in tool_arguments:
                            # Remove the URL from the query for the query parameter
                            cleaned_query = query
                            for url in detected_urls:
                                cleaned_query = cleaned_query.replace(url, '').strip()
                            tool_arguments["query"] = cleaned_query if cleaned_query else query
                        else:
                            tool_arguments["query"] = query
                    elif "question" in tool_properties and "question" not in tool_arguments:
                        tool_arguments["question"] = query
                    elif "prompt" in tool_properties and "prompt" not in tool_arguments:
                        tool_arguments["prompt"] = query
                    elif "text" in tool_properties and "text" not in tool_arguments:
                        tool_arguments["text"] = query
                    elif "instruction" in tool_properties and "instruction" not in tool_arguments:
                        tool_arguments["instruction"] = query
                    
                    # Priority 3: Handle required parameters that haven't been set
                    for param in required_params:
                        if param not in tool_arguments:
                            if param in ["content", "input", "data"]:
                                tool_arguments[param] = query
                            elif param in ["action", "command"]:
                                # For action/command parameters, try to extract action from query
                                action_words = ['open', 'click', 'type', 'scroll', 'navigate', 'extract', 'fetch']
                                for action in action_words:
                                    if action in query.lower():
                                        tool_arguments[param] = action
                                        break
                                else:
                                    tool_arguments[param] = "navigate" if detected_urls else "execute"
                            elif "message" in param.lower():
                                tool_arguments[param] = messages if isinstance(messages, list) else query
                            else:
                                # Default fallback for unknown required parameters
                                tool_arguments[param] = query
                    
                    # Ensure we have at least one argument
                    if not tool_arguments:
                        tool_arguments["query"] = query
                    
                    call_tool_request = {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": tool_arguments
                        }
                    }
                    
                    print(f"Calling tool '{tool_name}' with arguments: {list(tool_arguments.keys())}")
                    call_tool_json = json.dumps(call_tool_request) + "\n"
                    process.stdin.write(call_tool_json.encode())
                    await process.stdin.drain()
                    
                    # Read the tool call response with robust JSON parsing
                    tool_response = await self._read_json_response(process.stdout, max_lines=20, timeout=30.0)
                    if tool_response:
                        print(f"Tool response received: {tool_response}")
                        
                        if "result" in tool_response:
                            result = tool_response["result"]
                            if isinstance(result, dict):
                                # Handle different response formats dynamically
                                if "content" in result:
                                    # MCP standard format
                                    content_items = result["content"]
                                    if isinstance(content_items, list):
                                        for content_item in content_items:
                                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                                yield content_item.get("text", "")
                                    else:
                                        yield str(content_items)
                                elif "text" in result:
                                    # Simple text response
                                    yield result["text"]
                                elif "response" in result:
                                    # Response field
                                    yield result["response"]
                                elif "answer" in result:
                                    # Answer field
                                    yield result["answer"]
                                elif "output" in result:
                                    # Output field
                                    yield result["output"]
                                else:
                                    # Fallback: stringify the result
                                    yield str(result)
                            elif isinstance(result, str):
                                yield result
                            else:
                                yield str(result)
                        elif "error" in tool_response:
                            error_info = tool_response["error"]
                            if isinstance(error_info, dict):
                                error_message = error_info.get("message", str(error_info))
                            else:
                                error_message = str(error_info)
                            yield f"MCP tool error: {error_message}"
                        else:
                            yield f"Unexpected response format: {tool_response}"
                    else:
                        yield "Error: No valid tool response received from MCP server"
                else:
                    yield "No tools available from MCP server"
                
                # Gracefully close stdin to signal completion
                if process.stdin and not process.stdin.is_closing():
                    process.stdin.close()
                    print(f"MCP server stdin closed for '{tool_name}' execution")

                # 🚀 ENHANCED: Keep browser open indefinitely for navigation tools
                if tool_name and "navigate" in tool_name.lower() and detected_urls:
                    yield f"\n🌐 Browser opened and navigated to {detected_urls[0]}"
                    yield f"\n✨ Browser will remain open for continued interaction..."
                    yield f"\n💡 Tip: Send a new query (non-MCP) to close the browser and start normal chat"
                    print(f"🌐 Browser will remain open for {detected_urls[0]} - process tracked for cleanup")
                    
                    # Don't terminate the process - let it keep running
                    # The process will be cleaned up when:
                    # 1. User starts a new query (via _cleanup_mcp_processes)
                    # 2. Session ends
                    # 3. System shutdown
                    return  # Exit without terminating the process
                
        except Exception as e:
            print(f"Error in _execute_generic_mcp_server: {e}")
            import traceback
            traceback.print_exc()
            yield f"Error executing MCP server: {str(e)}"
            
        finally:
            # Only clean up if this isn't a navigation tool that should stay open
            if process and process_key in self.active_mcp_processes:
                if not (tool_name and "navigate" in tool_name.lower()):
                    try:
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close()
                        # Wait for process to terminate, with a timeout
                        await asyncio.wait_for(process.wait(), timeout=10.0)
                    except asyncio.TimeoutError:
                        print(f"Timeout waiting for MCP process {process.pid} to terminate. Keeping alive.")
                        # Don't kill - let it run
                    except Exception as e_proc:
                        print(f"Exception during MCP process cleanup: {e_proc}")
                    finally:
                        # Remove from tracking if it was terminated
                        if process.returncode is not None:
                            async with self.mcp_cleanup_lock:
                                if process_key in self.active_mcp_processes:
                                    del self.active_mcp_processes[process_key]

    # Add these new helper methods to the EnhancedRAG class

    def _select_best_tool_for_query(self, query: str, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best tool for the given query based on query intent"""
        query_lower = query.lower()
        
        # Create a priority map for different tools based on query intent
        tool_priorities = {}
        
        for tool in available_tools:
            tool_name = tool.get("name", "").lower()
            priority = 0
            
            # Navigation tools
            if "navigate" in tool_name:
                if any(nav_word in query_lower for nav_word in ['open', 'go to', 'navigate', 'visit']):
                    priority += 100
                elif any(url_indicator in query_lower for url_indicator in ['http', 'www', '.com', '.org']):
                    priority += 50
            
            # Screenshot tools
            elif "screenshot" in tool_name:
                if any(screen_word in query_lower for screen_word in ['screenshot', 'capture', 'image', 'picture']):
                    priority += 100
                elif any(nav_word in query_lower for nav_word in ['open', 'navigate', 'visit']):
                    priority += 30  # Screenshots often follow navigation
            
            # Click tools
            elif "click" in tool_name:
                if any(click_word in query_lower for click_word in ['click', 'press', 'button']):
                    priority += 100
            
            # Fill tools
            elif "fill" in tool_name:
                if any(fill_word in query_lower for fill_word in ['fill', 'type', 'enter', 'input']):
                    priority += 100
            
            # Evaluate tools
            elif "evaluate" in tool_name:
                if any(eval_word in query_lower for eval_word in ['evaluate', 'execute', 'run', 'script']):
                    priority += 100
            
            tool_priorities[tool] = priority
        
        # Return the tool with the highest priority, or the first tool if all have equal priority
        if tool_priorities:
            best_tool = max(tool_priorities.items(), key=lambda x: x[1])
            if best_tool[1] > 0:
                print(f"Selected tool '{best_tool[0].get('name')}' with priority {best_tool[1]}")
                return best_tool[0]
        
        # Fallback to first available tool
        return available_tools[0]

    async def _construct_url_from_query(self, query: str) -> Optional[str]:
        """Construct a URL from query text with improved domain detection"""
        import re
        
        # Use configuration patterns instead of hardcoded ones
        domain_patterns = self.config.url_patterns
        
        # Look for standalone domain names
        words = query.split()
        for word in words:
            # Clean the word
            clean_word = word.strip('.,!?;()[]"\'')
            
            # Check if it looks like a domain
            if '.' in clean_word and not clean_word.startswith('@'):
                # Check against domain patterns
                for pattern in domain_patterns:
                    if re.match(pattern, clean_word):
                        # Construct URL
                        if not clean_word.startswith(('http://', 'https://')):
                            if clean_word.startswith('www.'):
                                return f"https://{clean_word}"
                            else:
                                return f"https://{clean_word}"
                        return clean_word
        
        # Use LLM-based analysis instead of hardcoded site mappings
        return await self._llm_based_url_construction(query)

    async def _llm_based_url_construction(self, query: str) -> Optional[str]:
        """Use LLM to intelligently construct URLs from natural language"""
        if not self.config.use_llm_for_query_analysis:
            return None
            
        analysis_prompt = f"""
Extract or construct a URL from this query if possible:

QUERY: "{query}"

Guidelines:
- Look for website names, domain references, or service names
- Return the most likely URL in format: https://domain.com
- If no clear URL can be determined, return: NONE

Respond with only the URL or NONE:
"""

        try:
            messages = [
                {"role": "system", "content": "You are a URL extraction assistant. Be precise and only return valid URLs."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=self.config.analysis_temperature,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            if result.lower() != "none" and result.startswith(('http://', 'https://')):
                return result
                
        except Exception as e:
            print(f"Error in LLM-based URL construction: {e}")
        
        return None

    def _get_fallback_parameter_value(self, param: str, query: str, messages: List[Dict[str, str]], 
                                    detected_urls: List[str]) -> Any:
        """Get fallback values for required parameters using LLM analysis when possible"""
        
        param_lower = param.lower()
        
        # Use static fallbacks for simple cases
        if param_lower in ["content", "input", "data"]:
            return query
        elif param_lower in ["timeout"]:
            return self.config.default_timeout
        elif param_lower in ["name", "title"]:
            # Extract potential names or titles from query
            words = query.split()
            if len(words) > 1:
                return " ".join(words[:3])  # First few words as title
            return query
        elif "message" in param_lower:
            return messages if isinstance(messages, list) else query
        elif param_lower in ["action", "command"]:
            # Use LLM to extract action instead of hardcoded list
            return self._extract_action_with_llm(query, detected_urls)
        else:
            # Default fallback
            return query

    async def _extract_action_with_llm(self, query: str, detected_urls: List[str]) -> str:
        """Extract action intent using LLM instead of hardcoded keywords"""
        if not self.config.use_llm_for_query_analysis:
            # Fallback to simple heuristic
            return "navigate" if detected_urls else "execute"
            
        analysis_prompt = f"""
Determine the primary action intent from this query:

QUERY: "{query}"
HAS_URLS: {len(detected_urls) > 0}

Return one word action from: open, click, type, scroll, navigate, extract, fetch, execute

Action:
"""

        try:
            messages = [
                {"role": "system", "content": "You are an action extraction assistant. Return only the action word."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=self.config.analysis_temperature,
                max_tokens=10
            )
            
            action = response.choices[0].message.content.strip().lower()
            return action if action else ("navigate" if detected_urls else "execute")
            
        except Exception as e:
            print(f"Error in LLM-based action extraction: {e}")
            return "navigate" if detected_urls else "execute"

    def _detect_navigation_intent(self, query: str) -> bool:
        """Detect if query intends navigation vs content extraction using LLM analysis"""
        if not self.config.enable_dynamic_keyword_detection:
            # Simple pattern matching fallback
            return any(word in query.lower() for word in ['open', 'go', 'navigate', 'visit', 'browse'])
        
        return self._llm_detect_navigation_intent(query)

    async def _llm_detect_navigation_intent(self, query: str) -> bool:
        """Use LLM to detect navigation intent"""
        analysis_prompt = f"""
Does this query indicate navigation intent (opening/visiting pages) vs content extraction?

QUERY: "{query}"

Respond with only: NAVIGATION or EXTRACTION
"""

        try:
            messages = [
                {"role": "system", "content": "You are an intent classifier. Be precise."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=self.config.analysis_temperature,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            return "navigation" in result
            
        except Exception as e:
            print(f"Error in LLM-based navigation detection: {e}")
            # Fallback to pattern matching
            return any(word in query.lower() for word in ['open', 'go', 'navigate', 'visit', 'browse'])

    def _intelligent_server_selection(self, query: str, available_servers: List[str]) -> str:
        """Intelligently select MCP server based on query intent using LLM analysis"""
        if not self.config.use_llm_for_query_analysis or not available_servers:
            return available_servers[0] if available_servers else None
            
        return self._llm_select_server(query, available_servers)

    async def _llm_select_server(self, query: str, available_servers: List[str]) -> str:
        """Use LLM to select the most appropriate server"""
        analysis_prompt = f"""
Select the best MCP server for this query:

QUERY: "{query}"
AVAILABLE_SERVERS: {', '.join(available_servers)}

Consider:
- Navigation servers for opening/browsing
- Fetch/scraper servers for content extraction  
- Search servers for finding information
- File servers for file operations

Return only the server name:
"""

        try:
            messages = [
                {"role": "system", "content": "You are a server selection assistant. Return only the server name."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=self.config.analysis_temperature,
                max_tokens=20
            )
            
            selected = response.choices[0].message.content.strip()
            # Verify the selection is valid
            if selected in available_servers:
                return selected
                
        except Exception as e:
            print(f"Error in LLM-based server selection: {e}")
        
        # Fallback to first available server
        return available_servers[0]

    async def _review_combined_sources(self, query: str, all_docs: List[Document], system_prompt: str) -> Dict[str, List[Document]]:
        # Organize documents by source type
        kb_docs = []
        user_docs = []
        web_docs = []
        
        # Categorize documents by source
        for doc in all_docs:
            source_type = doc.metadata.get("source_type", "")
            source = doc.metadata.get("source", "").lower()
            
            if source_type == "web_search" or "web search" in source:
                web_docs.append(doc)
            elif "user" in source:
                user_docs.append(doc)
            else:
                kb_docs.append(doc)
        
        # If user documents exist, disable web search entirely
        is_web_search_query = False if user_docs else await self._llm_analyze_web_search_need(query)
        
        result = {
            "user_docs": user_docs,
            "kb_docs": kb_docs,
            "web_docs": web_docs,
            "is_web_search_query": is_web_search_query,
            "is_follow_up": False,
            "referring_entity": None
        }
        
        print(f"Document review results: {len(user_docs)} user docs, {len(kb_docs)} KB docs, {len(web_docs)} web docs")
        print(f"Web search: {'USED' if is_web_search_query and len(web_docs) > 0 else 'DISABLED'} (User docs present: {len(user_docs) > 0})")
        return result

    async def _llm_analyze_web_search_need(self, query: str) -> bool:
        """Use LLM to determine if web search is needed for current/recent information"""
        if not self.config.use_llm_for_query_analysis:
            return False  # Conservative fallback
            
        analysis_prompt = f"""
Does this query require web search for current/recent information?

QUERY: "{query}"

Consider:
- Current events, news, recent updates: YES
- General knowledge, explanations, how-to: NO  
- Real-time data (weather, stocks, prices): YES
- Academic/technical concepts: NO

Respond with only: YES or NO
"""

        try:
            messages = [
                {"role": "system", "content": "You are a query analyzer for web search necessity."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=self.config.analysis_temperature,
                max_tokens=5
            )
            
            result = response.choices[0].message.content.strip().lower()
            return "yes" in result
            
        except Exception as e:
            print(f"Error in LLM-based web search analysis: {e}")
            return False

    async def _analyze_web_search_necessity(self, query: str, chat_history: List[Dict[str, str]], user_r2_document_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Intelligent analysis to determine if web search is needed for the query.
        Always prioritizes user documents when available.
        """
        query_lower = query.lower().strip()
        
        # ENHANCED: Check for greetings/conversational responses FIRST
        is_greeting = await self._llm_detect_greeting(query_lower)
        if is_greeting:
            return {
                "should_use_web_search": False,
                "reasoning": "Simple greeting, acknowledgment, or conversational response detected",
                "confidence": "high" 
            }
        
        # Simple heuristic for very short queries
        if len(query_lower.split()) <= 3:
            # Additional check for short conversational responses
            short_conversational = ["great", "good", "nice", "cool", "ok", "okay", "sure", "yes", "no", "wow"]
            if any(word == query_lower for word in short_conversational):
                return {
                    "should_use_web_search": False,
                    "reasoning": "Short conversational response detected",
                    "confidence": "high" 
                }
        
        # FIXED: Check if user documents are available - pass the parameter properly
        has_user_documents = user_r2_document_keys and len(user_r2_document_keys) > 0
        
        # If user documents are provided, ALWAYS prioritize them and disable web search
        if has_user_documents:
            return {
                "should_use_web_search": False,
                "reasoning": "User documents are available, using only RAG with provided documents",
                "confidence": "high"
            }
        
        # Only use LLM analysis if no user documents are provided
        should_search = await self._llm_analyze_web_search_need(query)
        
        return {
            "should_use_web_search": should_search,
            "reasoning": "LLM analysis of query intent (no user documents available)",
            "confidence": "high" if self.config.use_llm_for_query_analysis else "medium"
        }

    async def _llm_detect_greeting(self, query: str) -> bool:
        """Use LLM to detect greetings and conversational responses with better nuance"""
        analysis_prompt = f"""
Determine if this is a simple conversational response, greeting, or acknowledgment that should be handled conversationally (without document retrieval).

TEXT: "{query}"

CONVERSATIONAL RESPONSES include:
- Greetings: hello, hi, hey, good morning
- Acknowledgments: thanks, thank you, great, good, nice, awesome, cool, perfect
- Simple reactions: wow, amazing, excellent, fantastic, wonderful, brilliant
- Short confirmations: ok, okay, sure, yes, no, alright
- Casual responses: haha, lol, that's funny, interesting

Return YES only if this is a simple conversational response that doesn't require factual information or document retrieval.
Return NO if it's asking for information, explanation, or specific content.

Respond with only: YES or NO
"""

        try:
            messages = [
                {"role": "system", "content": "You are a conversational intent detector. Be precise about whether something needs factual information or is just conversational."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=0.1,  # Lower temperature for more consistent detection
                max_tokens=5
            )
            
            result = response.choices[0].message.content.strip().lower()
            return "yes" in result
            
        except Exception as e:
            print(f"Error in LLM-based greeting detection: {e}")
            # Enhanced fallback with more conversational words
            conversational_words = [
                "hello", "hi", "hey", "thanks", "thank you", "great", "good", "nice", "awesome", 
                "cool", "ok", "okay", "sure", "yes", "no", "wow", "amazing", 
                "perfect", "excellent", "fantastic", "wonderful", "brilliant", "alright",
                "haha", "lol", "interesting", "that's funny"
            ]
            return any(word.strip().lower() == query.strip().lower() for word in conversational_words)

    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extract URLs from query text using configurable patterns"""
        import re
        
        # Use configuration patterns instead of hardcoded ones
        url_patterns = self.config.url_patterns
        
        urls = []
        for pattern in url_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                # Clean up the URL
                url = match.strip('.,!?;()[]"\'')
                
                # Skip if it looks like an email or file extension
                if '@' in url or url.endswith(('.txt', '.pdf', '.doc', '.jpg', '.png')):
                    continue
                    
                # Ensure it has protocol
                if not url.startswith(('http://', 'https://')):
                    if url.startswith('www.'):
                        url = 'https://' + url
                    elif '.' in url and len(url.split('.')) >= 2:
                        # Check if it's a valid domain structure
                        parts = url.split('.')
                        if len(parts[-1]) >= 2:  # Valid TLD
                            url = 'https://' + url
                        else:
                            continue
                
                # Validate the URL structure
                if self._is_valid_url_structure(url):
                    urls.append(url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls

    def _is_valid_url_structure(self, url: str) -> bool:
        """Validate if the URL has a proper structure"""
        import re
        
        # Basic URL validation pattern
        url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/.*)?$'
        
        if not re.match(url_pattern, url):
            return False
        
        # Check for reasonable domain structure
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Domain should have at least one dot and valid characters
            if not parsed.netloc or '.' not in parsed.netloc:
                return False
                
            # TLD should be at least 2 characters
            tld = parsed.netloc.split('.')[-1]
            if len(tld) < 2:
                return False
                
            return True
        except:
            return False

    async def query(
        self, session_id: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None,
        user_r2_document_keys: Optional[List[str]] = None, use_hybrid_search: Optional[bool] = None,
        llm_model_name: Optional[str] = None, system_prompt_override: Optional[str] = None,
        enable_web_search: Optional[bool] = False
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Determine effective hybrid search setting
        actual_use_hybrid_search = use_hybrid_search if use_hybrid_search is not None else self.default_use_hybrid_search
        if actual_use_hybrid_search:
            print(f"Hybrid search is ACTIVE for this query (session: {session_id}). BM25 Available: {HYBRID_SEARCH_AVAILABLE}")
        else:
            print(f"Hybrid search is INACTIVE for this query (session: {session_id}).")

        formatted_chat_history = await self._get_formatted_chat_history(session_id)
        retrieval_query = query

        all_retrieved_docs: List[Document] = []
        kb_docs = await self._get_retrieved_documents(
            self.kb_retriever, 
            retrieval_query, 
            k_val=5, 
            is_hybrid_search_active=actual_use_hybrid_search
        )
        if kb_docs: all_retrieved_docs.extend(kb_docs)
        
        user_session_retriever = await self._get_user_retriever(session_id)
        user_session_docs = await self._get_retrieved_documents(
            user_session_retriever, 
            retrieval_query, 
            k_val=3,  # Change from 5 to 3
            is_hybrid_search_active=actual_use_hybrid_search
        )
        if user_session_docs:
            # If user documents were found in the retriever, disable web search
            effective_enable_web_search = False
            print(f"[{session_id}] 📄 Retrieved {len(user_session_docs)} user-specific documents - FORCING RAG ONLY")
        else:
            web_search_analysis = await self._analyze_web_search_necessity(query, formatted_chat_history, user_r2_document_keys)
            effective_enable_web_search = web_search_analysis.get("should_use_web_search", False)
            print(f"[{session_id}] 🌐 Web search: {'ENABLED' if effective_enable_web_search else 'DISABLED'} (user override)")
            print(f"[{session_id}] 💡 Analysis suggests: {'WEB SEARCH' if web_search_analysis['should_use_web_search'] else 'NO WEB SEARCH'} - {web_search_analysis['reasoning']}")
        
        if user_r2_document_keys:
            adhoc_load_tasks = [self._download_and_split_one_doc(r2_key) for r2_key in user_r2_document_keys]
            results_list_of_splits = await asyncio.gather(*adhoc_load_tasks)
            for splits_from_one_doc in results_list_of_splits: all_retrieved_docs.extend(splits_from_one_doc)
        
        if enable_web_search and self.tavily_client:
            web_docs = await self._get_web_search_docs(retrieval_query, True, num_results=3)
            if web_docs: all_retrieved_docs.extend(web_docs)

        unique_docs_content = set()
        deduplicated_docs = []
        for doc in all_retrieved_docs:
            if doc.page_content not in unique_docs_content:
                deduplicated_docs.append(doc); unique_docs_content.add(doc.page_content)
        all_retrieved_docs = deduplicated_docs
        
        source_names_used = list(set([doc.metadata.get("source", "Unknown") for doc in all_retrieved_docs if doc.metadata]))
        if not source_names_used and all_retrieved_docs: source_names_used.append("Processed Documents")
        elif not all_retrieved_docs: source_names_used.append("No Context Found")

        answer_content = await self._generate_llm_response(
            session_id, query, all_retrieved_docs, formatted_chat_history,
            llm_model_name, system_prompt_override, stream=False
        )
        return {
            "answer": answer_content, "sources": source_names_used,
            "retrieval_details": {"documents_retrieved_count": len(all_retrieved_docs)},
            "total_time_ms": int((time.time() - start_time) * 1000)
        }

    async def clear_knowledge_base(self):
        print(f"Clearing KB for gpt_id '{self.gpt_id}' (collection '{self.kb_collection_name}')...")
        try:
            await asyncio.to_thread(self.qdrant_client.delete_collection, collection_name=self.kb_collection_name)
        except Exception as e:
            if "not found" in str(e).lower() or ("status_code" in dir(e) and e.status_code == 404):
                print(f"KB Qdrant collection '{self.kb_collection_name}' not found, no need to delete.")
            else: print(f"Error deleting KB Qdrant collection '{self.kb_collection_name}': {e}")
        self.kb_retriever = None
        await asyncio.to_thread(self._ensure_qdrant_collection_exists_sync, self.kb_collection_name)
        print(f"Knowledge Base for gpt_id '{self.gpt_id}' cleared and empty collection ensured.")

    async def clear_all_context(self):
        """Clear all user memories and MCP processes."""
        self.user_memories.clear()
        await self._cleanup_mcp_processes()
        print("All user session contexts and MCP processes have been cleared.")

    def _find_matching_mcp_server(self, referenced_server: str, available_servers: List[str]) -> Optional[str]:
        """Find the best matching MCP server from the referenced name."""
        if not referenced_server or not available_servers:
            return None
            
        # Direct match - highest priority
        if referenced_server in available_servers:
            return referenced_server
            
        # Case-insensitive match
        for server in available_servers:
            if server.lower() == referenced_server.lower():
                return server
                
        # Fuzzy match - check if any available server contains the referenced name
        for server in available_servers:
            if referenced_server.lower() in server.lower() or server.lower() in referenced_server.lower():
                return server
                
        # No match found
        return None

    async def detect_query_type(self, query: str) -> Dict[str, Any]:
        """Detect if the query should use MCP and which server to use."""
        # First check if MCP is properly initialized
        if not self.mcp_enabled or not self.mcp_servers_config:
            print(f"MCP Status Check: enabled={self.mcp_enabled}, servers_configured={bool(self.mcp_servers_config)}")
            return {"query_type": "rag", "explanation": "MCP is disabled"}
        
        # Check for MCP server references (@server)
        server_match = re.search(r'@([a-zA-Z0-9_-]+)', query)
        if server_match:
            referenced_server = server_match.group(1)
            print(f"Found server reference: @{referenced_server}")
            
            # Direct match
            if referenced_server in self.mcp_servers_config:
                print(f"✅ Found exact server match: {referenced_server}")
                return {
                    "query_type": "mcp",
                    "server_name": referenced_server,
                    "explanation": f"Query references MCP server '@{referenced_server}'"
                }
            
            # Try fuzzy matching
            matched_server = self._find_matching_mcp_server(referenced_server, list(self.mcp_servers_config.keys()))
            if matched_server:
                print(f"✅ Found fuzzy server match: {matched_server} for @{referenced_server}")
                return {
                    "query_type": "mcp",
                    "server_name": matched_server,
                    "explanation": f"Query matches MCP server '{matched_server}' (from @{referenced_server})"
                }
            
            print(f"❌ No matching server found for @{referenced_server}")
        
        return {"query_type": "rag", "explanation": "No valid MCP server reference found"}

    # Add this new method to the EnhancedRAG class
    async def clear_user_memory(self, session_id: str):
        """Clear the chat memory for a specific session but keep documents"""
        if session_id in self.user_memories:
            del self.user_memories[session_id]
            print(f"Chat memory cleared for session_id: {session_id}")
        
        # Initialize a fresh memory for this session
        await self._get_user_memory(session_id)
        return True

    async def _summarize_ai_message(self, content: str) -> str:
        # Don't summarize - keep the full content for context
        return content

    async def _extract_reference_entity(self, query: str) -> Optional[str]:
        """Extract the entity being referred to in a follow-up question"""
        try:
            # Get recent message history to extract context
            recent_messages = []
            for session_id, memory in self.user_memories.items():
                if memory and memory.messages:
                    recent_messages = [msg for msg in memory.messages[-4:]]  # Get last 4 messages
                    break
            
            if not recent_messages:
                return None
                
            # Extract context using simple heuristics first
            for msg in reversed(recent_messages):
                if isinstance(msg, HumanMessage):
                    words = msg.content.split()
                    if len(words) >= 2:
                        # Check for common name formats
                        for i in range(len(words)-1):
                            if words[i].lower() in ['mr', 'dr', 'mrs', 'ms']:
                                return f"{words[i]} {words[i+1]}"
                        
                        # Check for proper nouns (simple heuristic)
                        for word in words:
                            if word[0].isupper() and len(word) > 1 and word.lower() not in ['i', 'a', 'the', 'and', 'but']:
                                return word
            
            # If simple heuristics fail, use LLM
            if self.config.use_llm_for_query_analysis:
                messages = []
                for msg in recent_messages:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    messages.append({"role": role, "content": msg.content})
                
                messages.append({
                    "role": "user", 
                    "content": f"Based on this conversation, what entity (person, place, or thing) does the query '{query}' refer to? Return only the entity name without explanation."
                })
                
                response = await self.async_openai_client.chat.completions.create(
                    model=self.config.analysis_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=20
                )
                
                entity = response.choices[0].message.content.strip()
                return entity if entity and entity.lower() not in ["none", "unknown", "n/a"] else None
                
        except Exception as e:
            print(f"Error extracting reference entity: {e}")
            return None
        
        return None

    async def _enhanced_review_combined_sources(self, query: str, all_docs: List[Document], system_prompt: str, chat_history: List[Dict[str, str]]) -> Dict[str, List[Document]]:
        """
        Enhanced source review with follow-up detection and intelligent web search integration
        """
        # Organize documents by source type
        kb_docs = []
        user_docs = []
        web_docs = []
        
        # Categorize documents by source
        for doc in all_docs:
            source_type = doc.metadata.get("source_type", "")
            source = doc.metadata.get("source", "").lower()
            
            if source_type == "web_search" or "web search" in source:
                web_docs.append(doc)
            elif "user" in source:
                user_docs.append(doc)
            else:
                kb_docs.append(doc)
        
        # Detect follow-up questions and references
        follow_up_analysis = await self._detect_followup_and_references(query, chat_history)
        is_follow_up = follow_up_analysis["is_follow_up"]
        referring_entity = follow_up_analysis.get("referring_entity")
        
        # Enhanced web search decision with follow-up context
        is_web_search_query = await self._intelligent_web_search_decision(
            query, user_docs, web_docs, is_follow_up, referring_entity
        )
        
        # Apply cosine similarity ranking for web docs when user docs exist
        if user_docs and web_docs and is_web_search_query:
            web_docs = await self._rank_web_docs_by_similarity(query, user_docs, web_docs)
        
        result = {
            "user_docs": user_docs,
            "kb_docs": kb_docs,
            "web_docs": web_docs,
            "is_web_search_query": is_web_search_query,
            "is_follow_up": is_follow_up,
            "referring_entity": referring_entity,
            "follow_up_context": follow_up_analysis.get("context_needed", "")
        }
        
        print(f"📊 Enhanced Document Review:")
        print(f"   User docs: {len(user_docs)} | KB docs: {len(kb_docs)} | Web docs: {len(web_docs)}")
        print(f"   Follow-up: {'YES' if is_follow_up else 'NO'} | Entity: {referring_entity or 'None'}")
        print(f"   Web search: {'ENABLED' if is_web_search_query else 'DISABLED'}")
        
        return result

    async def _detect_followup_and_references(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Enhanced detection of follow-up queries and context tracking"""
        # Default return structure
        result = {
            "is_follow_up": False,
            "referring_entity": None,
            "context_topic": None
        }
        
        if not chat_history or len(chat_history) < 2:
            return result
            
        # Extract the most recent assistant message for context
        recent_messages = [msg for msg in chat_history[-3:] if msg.get("type") == "ai"]
        if not recent_messages:
            return result
            
        recent_ai_message = recent_messages[-1]["content"]
        
        # Extract topic entities from recent AI message using LLM if enabled
        if self.config.enforce_context_continuity:
            result["context_topic"] = await self._extract_main_topic(recent_ai_message)
            
            # Check if current query is a follow-up to the identified topic
            if result["context_topic"]:
                result["is_follow_up"] = True
                result["referring_entity"] = result["context_topic"]
                
        # Rest of existing logic for detecting follow-up queries...
        
        return result

    async def _intelligent_web_search_decision(self, query: str, user_docs: List[Document], 
                                             web_docs: List[Document], is_follow_up: bool, 
                                             referring_entity: Optional[str]) -> bool:
        """
        Enhanced web search decision that considers document availability and similarity filtering
        """
        # If explicitly requested web search in query
        query_lower = query.lower()
        explicit_web_request = any(phrase in query_lower for phrase in [
            "web search", "search online", "latest", "recent", "current", "today", "now"
        ])
        
        if explicit_web_request:
            print("🌐 Web search explicitly requested in query")
            return True
        
        # For follow-up questions, check if user docs are actually relevant
        if is_follow_up and user_docs:
            # First check if user documents are relevant to the enhanced query/referring entity
            if referring_entity:
                # Check if any user document mentions the referring entity
                entity_mentioned = any(
                    referring_entity.lower() in doc.page_content.lower() 
                    for doc in user_docs[:3]  # Check first 3 docs
                )
                
                if entity_mentioned:
                    # Only use web search for follow-ups if it needs current info
                    needs_current_info = any(phrase in query_lower for phrase in [
                        "latest", "recent", "current events", "news", "today", "now", "updates"
                    ])
                    if needs_current_info:
                        print("🔄 Follow-up needs current information - enabling web search")
                        return True
                    else:
                        print("🔄 Follow-up can be satisfied by relevant existing documents")
                        return False
                else:
                    # User docs don't contain the referring entity - enable web search
                    print(f"🔄 Follow-up about '{referring_entity}' not found in user documents - enabling web search")
                    return True
            else:
                # Fallback to similarity check if no referring entity
                print("🔄 Follow-up without specific entity - checking document relevance")
                return True
        
        # ENHANCED: Always try web search when available if explicitly enabled, let similarity filter handle quality
        if self.tavily_client and not user_docs:
            print("🌐 No user documents - enabling web search (will filter by similarity)")
            return True
        elif self.tavily_client and user_docs:
            # Even with user docs, try complementary web search with similarity filtering
            print("🌐 User documents available - enabling complementary web search (will filter by >0.4 similarity)")
            return True
        
        return False

    async def _analyze_complementary_search_need(self, query: str, user_docs: List[Document]) -> bool:
        """
        Analyze if web search would provide complementary information to user documents
        """
        # Extract key topics from user documents
        user_content_sample = " ".join([doc.page_content[:100] for doc in user_docs[:3]])
        
        analysis_prompt = f"""
Would web search provide valuable complementary information for this query?

USER DOCUMENTS SAMPLE: {user_content_sample[:500]}...

QUERY: "{query}"

Consider:
- Does query ask for recent developments not in user docs?
- Does query need external perspectives or comparisons?
- Does query require current data/statistics?
- Would web sources add significant value?

Respond with: YES or NO
"""

        try:
            messages = [
                {"role": "system", "content": "You analyze whether web search adds value to existing documents."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=messages,
                temperature=0.1,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            return "yes" in result
            
        except Exception as e:
            print(f"Error in complementary search analysis: {e}")
            return False

    async def _rank_web_docs_by_similarity(self, query: str, user_docs: List[Document], 
                                         web_docs: List[Document], 
                                         is_follow_up: bool = False) -> Tuple[List[Document], Dict[str, Any]]:
        """Filter web documents based on similarity to user documents and query"""
        filtered_docs = []
        metadata = {"total": len(web_docs), "passed": 0, "filtered": 0}
        
        # Use configured threshold (0.4 by default)
        threshold = self.config.follow_up_search_threshold if is_follow_up else self.config.web_search_similarity_threshold
        
        print(f"🎯 Using {'follow-up' if is_follow_up else 'standard'} similarity threshold: {threshold} ({threshold*100}%)")
        
        try:
            # Get embeddings for user documents
            user_embeddings = []
            for user_doc in user_docs[:5]:  # Limit to first 5 docs for performance
                try:
                    embedding = await self._get_text_embedding(user_doc.page_content[:500])
                    user_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error getting user doc embedding: {e}")
                    continue
            
            if not user_embeddings:
                print("⚠️ No user document embeddings available, falling back to query similarity")
                return await self._filter_by_query_similarity_only(query, web_docs), metadata
            
            # Filter web docs based on similarity to user docs and query
            for web_doc in web_docs:
                try:
                    # Calculate similarity to query
                    query_similarity = await self._calculate_doc_query_similarity(query, web_doc)
                    
                    # Calculate similarity to user documents (take max similarity)
                    web_embedding = await self._get_text_embedding(web_doc.page_content[:500])
                    user_similarities = []
                    
                    for user_embedding in user_embeddings:
                        similarity = cosine_similarity([web_embedding], [user_embedding])[0][0]
                        user_similarities.append(similarity)
                    
                    max_user_similarity = max(user_similarities) if user_similarities else 0
                    
                    # Use the higher of query similarity or user document similarity
                    final_similarity = max(query_similarity, max_user_similarity)
                    
                    if final_similarity >= threshold:
                        print(f"✅ Web doc passed: query_sim={query_similarity:.3f}, max_user_sim={max_user_similarity:.3f}, final={final_similarity:.3f} >= {threshold}")
                        filtered_docs.append(web_doc)
                        metadata["passed"] += 1
                    else:
                        print(f"❌ Web doc filtered: query_sim={query_similarity:.3f}, max_user_sim={max_user_similarity:.3f}, final={final_similarity:.3f} < {threshold}")
                        metadata["filtered"] += 1
                        
                except Exception as e:
                    print(f"Error calculating web doc similarity: {e}")
                    metadata["filtered"] += 1
                    
        except Exception as e:
            print(f"Error in similarity ranking: {e}")
            # Fallback to query similarity only
            return await self._filter_by_query_similarity_only(query, web_docs), metadata
                
        return filtered_docs, metadata

    async def _calculate_doc_query_similarity(self, query: str, doc: Document) -> float:
        """
        Calculate semantic similarity between a query and a document
        
        Args:
            query: The user query
            doc: The document to compare
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            # Get embeddings for query and document
            query_embedding = await self._get_text_embedding(query)
            # Limit text to first 500 chars to optimize embedding calculation
            doc_embedding = await self._get_text_embedding(doc.page_content[:500])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            # Return moderate similarity as fallback
            return 0.4

    async def _get_text_embedding(self, text: str):
        """
        Get embedding for text with error handling
        """
        try:
            # Use OpenAI embeddings (fix the attribute name)
            embedding = await self.embeddings_model.aembed_query(text)
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return random embedding as fallback
            return np.random.random(1536)

    async def _create_advanced_rag_chain(self, documents: List[Document], query: str, 
                                       is_follow_up: bool, referring_entity: Optional[str]):
        """
        Create advanced RAG chain using LangChain Expression Language (LCEL)
        """
        try:
            # Create retriever from documents
            retriever = self._create_document_retriever(documents)
            
            # Create prompt template with follow-up awareness
            prompt_template = self._create_followup_aware_prompt(is_follow_up, referring_entity)
            
            # Build the RAG chain using LCEL
            rag_chain = (
                RunnableParallel({
                    "context": retriever | self._format_documents,
                    "question": RunnablePassthrough(),
                    "is_follow_up": RunnableLambda(lambda x: is_follow_up),
                    "referring_entity": RunnableLambda(lambda x: referring_entity)
                })
                | prompt_template
                | self._get_llm_runnable()
                | StrOutputParser()
            )
            
            return rag_chain
            
        except Exception as e:
            print(f"Error creating advanced RAG chain: {e}")
            return None

    def _create_followup_aware_prompt(self, is_follow_up: bool, referring_entity: Optional[str]) -> ChatPromptTemplate:
        """
        Create context-aware prompt template for follow-up questions
        """
        if is_follow_up:
            system_prompt = """You are answering a FOLLOW-UP question. The user is asking for MORE information about a topic previously discussed.

Context: {context}
Previous Entity: {referring_entity}
Follow-up Question: {question}

IMPORTANT FOLLOW-UP RULES:
1. Focus on NEW information not covered in previous responses
2. Provide additional details, examples, or aspects about {referring_entity}
3. Don't repeat information already provided
4. Be more detailed and comprehensive than usual
5. Connect new information to what was previously discussed

Provide a detailed, informative response with new insights."""
        else:
            system_prompt = """You are a helpful AI assistant providing comprehensive information.

Context: {context}
Question: {question}

Provide a clear, accurate, and well-structured response based on the given context."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents for context"""
        return "\n\n".join([doc.page_content for doc in docs])

    def _get_llm_runnable(self):
        """Get LLM as a runnable for LCEL chains"""
        try:
            # Return a simple lambda that processes the formatted prompt
            async def process_llm_input(input_dict):
                # Extract the formatted prompt
                messages = input_dict.get("messages", [])
                # This would integrate with your existing LLM generation logic
                return "Processed response"  # Placeholder
            
            return RunnableLambda(process_llm_input)
        except Exception as e:
            print(f"Error creating LLM runnable: {e}")
            return RunnableLambda(lambda x: "Error processing request")

    # Replace the original _review_combined_sources method
    async def _review_combined_sources(self, query: str, all_docs: List[Document], system_prompt: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, List[Document]]:
        """
        Enhanced wrapper that uses the new advanced review system
        """
        if chat_history is None:
            chat_history = []
        
        try:
            # Use enhanced review with error handling
            return await self._enhanced_review_combined_sources(query, all_docs, system_prompt, chat_history)
        except Exception as e:
            print(f"Error in enhanced source review, falling back to basic review: {e}")
            # Fallback to basic review
            return await self._basic_review_combined_sources(query, all_docs, system_prompt)

    async def _basic_review_combined_sources(self, query: str, all_docs: List[Document], system_prompt: str) -> Dict[str, List[Document]]:
        """
        Basic fallback source review method
        """
        kb_docs = []
        user_docs = []
        web_docs = []
        
        for doc in all_docs:
            source_type = doc.metadata.get("source_type", "")
            source = doc.metadata.get("source", "").lower()
            
            if source_type == "web_search" or "web search" in source:
                web_docs.append(doc)
            elif "user" in source:
                user_docs.append(doc)
            else:
                kb_docs.append(doc)
        
        # More permissive web search logic
        is_web_search_query = True if web_docs else await self._llm_analyze_web_search_need(query)
        
        return {
            "user_docs": user_docs,
            "kb_docs": kb_docs,
            "web_docs": web_docs,
            "is_web_search_query": is_web_search_query,
            "is_follow_up": False,
            "referring_entity": None
        }

    # Enhanced response generation with follow-up awareness
    async def _generate_enhanced_llm_response(
        self, session_id: str, query: str, all_context_docs: List[Document],
        chat_history_messages: List[Dict[str, str]], llm_model_name_override: Optional[str],
        system_prompt_override: Optional[str], stream: bool = False,
        is_follow_up: bool = False, referring_entity: Optional[str] = None
    ) -> Union[AsyncGenerator[str, None], str]:
        """
        Enhanced LLM response generation with follow-up awareness
        """
        current_llm_model = llm_model_name_override or self.default_llm_model_name
        current_system_prompt = system_prompt_override or self.default_system_prompt
        
        # Enhanced context formatting with follow-up awareness
        context_str = self._format_docs_for_llm_context(all_context_docs, "Retrieved Context")
        if not context_str.strip():
            context_str = "No relevant context could be found from any available source for this query."

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Enhanced prompt for follow-up questions
        if is_follow_up and referring_entity:
            follow_up_instruction = f"""
🔄 **FOLLOW-UP CONTEXT**: This is a follow-up question about: {referring_entity}
- Provide NEW information not covered in previous responses
- Focus on additional details, examples, or aspects about {referring_entity}  
- Be more comprehensive and detailed than usual
- Connect new information to previous discussion
"""
        else:
            follow_up_instruction = ""
        
        user_query_message_content = (
            f"📘 **CONTEXT:**\n{context_str}\n\n"
            f"🕒 **Current Time:** {formatted_time}\n\n"
            f"{follow_up_instruction}"
            f"💬 **USER QUESTION:** {query}\n\n"
            f"📝 **CONVERSATION HISTORY:** {len(chat_history_messages)} previous messages\n\n"
            f"📌 **RESPONSE RULES:**\n"
            f"- {'Provide detailed follow-up information' if is_follow_up else 'Keep it conversational and informative'}\n"
            f"- Use **only the given context**. Don't guess or assume.\n"
            f"- If the context isn't enough, say that clearly and briefly.\n"
            f"- Use *General Insight:* only when adding general knowledge.\n"
            f"- For follow-up questions, focus on NEW information not already covered.\n"
            f"- For 'tell me more' requests, provide additional details or examples.\n"
            f"- Vary your response structure and content to avoid repetition.\n\n"
            f"✅ **STYLE & FORMAT:**\n"
            f"- 🏷️ Start with an emoji + quick headline\n"
            f"- 📋 Use bullets or short paras for clarity\n"
            f"- 💡 Emphasize main points\n"
            f"- 😊 Make it friendly and human\n"
            f"- 🤝 End with different follow-up suggestions to keep conversation flowing\n"
        )

        messages = [{"role": "system", "content": current_system_prompt}]
        messages.extend(chat_history_messages)
        messages.append({"role": "user", "content": user_query_message_content})

        # Use existing LLM generation logic with enhanced prompting
        user_memory = await self._get_user_memory(session_id)
        
        # Log the enhanced processing
        print(f"🧠 Enhanced LLM Generation: {'Follow-up' if is_follow_up else 'Initial'} | Entity: {referring_entity or 'None'}")
        
        # Continue with existing LLM generation logic...
        # (The rest follows the existing _generate_llm_response method structure)
        # This is a placeholder - you would integrate this with your existing generation logic
        
        return await self._generate_llm_response(
            session_id, query, all_context_docs, chat_history_messages,
            llm_model_name_override, system_prompt_override, stream
        )


    # Add comprehensive error handling to the main methods
    async def _robust_query_processing(self, session_id: str, query: str, **kwargs):
        """
        Robust query processing with comprehensive error handling
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if not session_id:
                raise ValueError("Session ID is required")
            
            # Process with error handling
            return await self._safe_execute_with_langsmith(
                "query_processing",
                self.query_stream,
                session_id, query, **kwargs
            )
            
        except Exception as e:
            print(f"🚨 Critical error in query processing: {e}")
            # Return error generator
            async def error_generator():
                yield {"type": "error", "data": f"Sorry, I encountered an error: {str(e)}"}
                yield {"type": "done", "data": "Error occurred during processing"}
            
            return error_generator()

    async def _generate_simple_greeting_response(self, query: str, chat_history: List[Dict[str, str]], llm_model_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate a natural, varied conversational response using direct LLM calls"""
        
        try:
            current_llm_model = llm_model_name or self.default_llm_model_name
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Enhanced conversational system prompt with better variety instructions
            conversational_system_prompt = f"""You are a friendly, helpful AI assistant engaged in natural conversation. 

IMPORTANT GUIDELINES:
- Respond naturally and conversationally to the user's message
- Vary your responses - never give the same response twice to similar inputs
- Be warm, friendly, and engaging without being overly enthusiastic
- Keep responses concise but personable (1-2 sentences usually)
- Reference the conversation context when appropriate
- Use light, appropriate emojis sparingly (0-1 per response)
- End with a gentle invitation to continue chatting when suitable

CONTEXT: Current time is {formatted_time}. Respond as if you're having a natural chat with a friend."""
            
            # Build conversation context (last 6 messages for better context)
            messages = [{"role": "system", "content": conversational_system_prompt}]
            
            # Include more recent chat history for better context
            if chat_history:
                recent_history = chat_history[-6:]  # Last 6 messages for richer context
                messages.extend(recent_history)
            
            # Add current message with timestamp variation for uniqueness
            messages.append({
                "role": "user", 
                "content": f"User says: {query}\n\nTime: {formatted_time}"
            })
            
            print(f"🤖 Generating varied conversational response for: '{query}'")
            
            # Try LLM first with higher temperature for variety
            normalized_model = current_llm_model.lower().strip()
            
            # OpenAI models (GPT-4o, GPT-4o-mini, etc.)
            if normalized_model.startswith("gpt-"):
                try:
                    openai_model_name = normalized_model
                    
                    # Validate model name
                    valid_openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
                    if openai_model_name not in valid_openai_models:
                        print(f"Unknown OpenAI model '{openai_model_name}', falling back to gpt-4o-mini")
                        openai_model_name = "gpt-4o-mini"
                    
                    print(f"Using OpenAI model: {openai_model_name} for conversational response")
                    
                    response_stream = await self.async_openai_client.chat.completions.create(
                        model=openai_model_name,
                        messages=messages,
                        temperature=0.8,  # Higher temperature for more variety
                        max_tokens=100,   # Slightly more room for natural responses
                        stream=True
                    )
                    
                    async for chunk in response_stream:
                        content_piece = chunk.choices[0].delta.content
                        if content_piece:
                            yield content_piece
                    return
                            
                except Exception as e:
                    print(f"Error with OpenAI conversational generation: {e}")
            
            # Claude models
            elif normalized_model.startswith("claude") and self.anthropic_client:
                try:
                    claude_messages = []
                    system_content = conversational_system_prompt
                    
                    for msg in messages[1:]:  # Skip system message
                        if msg["role"] != "system":
                            claude_messages.append(msg)
                    
                    response_stream = await self.anthropic_client.messages.create(
                        model="claude-3.5-haiku-20240307",
                        system=system_content,
                        messages=claude_messages,
                        max_tokens=100,
                        temperature=0.8,
                        stream=True
                    )
                    
                    async for chunk in response_stream:
                        if chunk.type == "content_block_delta" and chunk.delta.text:
                            yield chunk.delta.text
                    return
                            
                except Exception as e:
                    print(f"Error with Claude conversational generation: {e}")
            
            # Gemini models
            elif normalized_model.startswith("gemini") and self.gemini_client:
                try:
                    gemini_messages = []
                    for msg in messages[1:]:  # Skip system message
                        if msg["role"] == "user":
                            gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                        elif msg["role"] == "assistant":
                            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                    
                    gemini_model_name = "gemini-2.5-flash-preview-04-17" if "flash" in normalized_model else "gemini-2.5-pro-preview-05-06"
                    model = self.gemini_client.GenerativeModel(model_name=gemini_model_name)
                    
                    response_stream = await model.generate_content_async(
                        gemini_messages,
                        generation_config={"temperature": 0.8, "max_output_tokens": 100},
                        stream=True
                    )
                    
                    async for chunk in response_stream:
                        if hasattr(chunk, "text") and chunk.text:
                            yield chunk.text
                    return
                            
                except Exception as e:
                    print(f"Error with Gemini conversational generation: {e}")
            
            # Groq/Llama models
            elif ("llama" in normalized_model or normalized_model.startswith("meta-llama/")) and self.groq_client:
                try:
                    groq_model = "llama-3.1-8b-instant"
                    
                    response_stream = await self.groq_client.chat.completions.create(
                        model=groq_model,
                        messages=messages,
                        temperature=0.8,
                        max_tokens=100,
                        stream=True
                    )
                    
                    async for chunk in response_stream:
                        content_piece = chunk.choices[0].delta.content
                        if content_piece:
                            yield content_piece
                    return
                            
                except Exception as e:
                    print(f"Error with Groq conversational generation: {e}")
            
            # OpenRouter models
            elif self.openrouter_client:
                try:
                    response_stream = await self.openrouter_client.chat.completions.create(
                        model=current_llm_model,
                        messages=messages,
                        temperature=0.8,
                        max_tokens=100,
                        stream=True
                    )
                    
                    async for chunk in response_stream:
                        content_piece = chunk.choices[0].delta.content
                        if content_piece:
                            yield content_piece
                    return
                            
                except Exception as e:
                    print(f"Error with OpenRouter conversational generation: {e}")
            
            # Enhanced fallback with timestamp-based variety
            print("All LLM conversational methods failed, using varied fallback response")
            
            # Create varied responses based on the specific query and time
            fallback_responses = {
                "great": [
                    "That's wonderful! 😊 What else would you like to explore?",
                    "Awesome! I'm glad that worked out well. Anything else I can help with?",
                    "Fantastic! 🌟 Feel free to ask me anything else you're curious about.",
                    "Excellent! I'm here if you need help with anything else.",
                    "Perfect! 👍 What would you like to discuss next?"
                ],
                "good": [
                    "I'm glad you think so! What's next on your mind?",
                    "Great to hear! 😊 Anything else you'd like to know about?",
                    "That's nice! I'm here whenever you need assistance.",
                    "Wonderful! Feel free to ask me about anything else.",
                    "Happy to help! What else can I do for you?"
                ],
                "nice": [
                    "Thank you! I'm glad you liked it. What else interests you?",
                    "I appreciate that! 😊 How else can I assist you today?",
                    "That's kind of you to say! What would you like to explore next?",
                    "Thanks! I'm here to help with whatever you need.",
                    "So glad you think so! Anything else on your mind?"
                ],
                "thanks": [
                    "You're very welcome! Happy to help anytime. 😊",
                    "My pleasure! Feel free to ask if you need anything else.",
                    "Glad I could help! What else can I do for you?",
                    "You're welcome! I'm here whenever you need assistance.",
                    "Anytime! Let me know if there's anything else you'd like to know."
                ],
                "cool": [
                    "Right? 😎 What else would you like to check out?",
                    "I thought so too! Anything else you're curious about?",
                    "Pretty neat stuff! What else can I help you discover?",
                    "Glad you found it interesting! What's next?",
                    "Awesome! Feel free to ask me about anything else."
                ]
            }
            
            # Get query-specific responses or default
            query_lower = query.lower().strip()
            responses = fallback_responses.get(query_lower, [
                "I'm here and ready to help! 😊 What would you like to know about?",
                "Hello! How can I assist you today?",
                "Hi there! What's on your mind?",
                "Hey! What would you like to explore together?",
                "Hello! I'm here to help with whatever you need."
            ])
            
            # Use timestamp + conversation length for better variety
            current_timestamp = int(current_time.timestamp())
            conversation_context = len(chat_history) + len(query)
            response_idx = (current_timestamp + conversation_context) % len(responses)
            fallback_response = responses[response_idx]
            
            # Stream the response with natural pacing
            words = fallback_response.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                if i % 2 == 0:  # Small delay every couple words for natural flow
                    await asyncio.sleep(0.03)
                    
        except Exception as e:
            print(f"Error in conversational response generation: {e}")
            # Final simple fallback
            simple_response = "I'm here to help! 😊 What can I assist you with?"
            words = simple_response.split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                if i % 2 == 0:
                    await asyncio.sleep(0.03)

    # Add a new method to filter web docs with better logging
    async def _filter_web_docs_by_similarity(self, query: str, user_docs: List[Document], 
                                       web_docs: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Filter web documents by cosine similarity threshold with detailed metrics
        Enhanced to properly handle both user docs and query similarity
        """
        if not web_docs:
            return [], {"original": 0, "filtered": 0, "passed": 0, "threshold": self.config.web_search_similarity_threshold}
        
        print(f"🔍 Filtering {len(web_docs)} web documents with similarity threshold: {self.config.web_search_similarity_threshold}")
        
        # If user docs exist, use them for similarity comparison
        if user_docs:
            print(f"📚 Using {len(user_docs)} existing documents for similarity comparison")
            filtered_docs, metadata = await self._rank_web_docs_by_similarity(query, user_docs, web_docs)
        else:
            # If no user docs, use query similarity only
            print("🔍 No existing documents - using query similarity only")
            filtered_docs = await self._filter_by_query_similarity_only(query, web_docs)
            metadata = {
                "total": len(web_docs),
                "passed": len(filtered_docs),
                "filtered": len(web_docs) - len(filtered_docs)
            }
        
        metrics = {
            "original": len(web_docs),
            "filtered": len(web_docs) - len(filtered_docs),
            "passed": len(filtered_docs),
            "threshold": self.config.web_search_similarity_threshold
        }
        
        print(f"✅ Similarity filtering results: {metrics['passed']}/{metrics['original']} web docs passed threshold")
        return filtered_docs, metrics

    async def _filter_by_query_similarity_only(self, query: str, web_docs: List[Document]) -> List[Document]:
        """
        Filter web documents by query similarity only when no user docs available
        """
        try:
            query_embedding = await self._get_text_embedding(query)
            qualified_docs = []
            
            # Fixed threshold of 0.4 (40%) as requested
            query_threshold = 0.4
            print(f"Using fixed similarity threshold: {query_threshold} (40%)")
            
            for web_doc in web_docs:
                try:
                    web_embedding = await self._get_text_embedding(web_doc.page_content[:500])
                    query_sim = cosine_similarity([query_embedding], [web_embedding])[0][0]
                    
                    if query_sim >= query_threshold:
                        qualified_docs.append(web_doc)
                        print(f"✅ Web doc passed query similarity: {query_sim:.3f} >= {query_threshold}")
                    else:
                        print(f"❌ Web doc filtered (low query similarity): {query_sim:.3f} < {query_threshold}")
                        
                except Exception as e:
                    print(f"Error calculating query similarity: {e}")
            
            return qualified_docs
            
        except Exception as e:
            print(f"Error in query-only similarity filtering: {e}")
            return web_docs

    async def _manage_session_memory(self, session_id: str, is_new_chat: bool = False):
        """Manage session memory based on configuration rules"""
        # Get session info if exists
        session_info = await self._get_session_info(session_id)
        
        # Clear memory if:
        # 1. is_new_chat flag is True
        # 2. Last activity was more than config.memory_expiry_minutes ago
        # 3. Conversation turns exceed config.max_conversation_turns
        
        if is_new_chat:
            logger.info(f"[{session_id}] 🧹 New chat - clearing previous memory")
            await self.clear_user_memory(session_id)
            return
            
        if session_info:
            # Check inactivity timeout
            last_activity = session_info.get("last_activity", 0)
            current_time = time.time()
            inactivity_minutes = (current_time - last_activity) / 60
            
            if inactivity_minutes >= self.config.memory_expiry_minutes:
                logger.info(f"[{session_id}] 🧹 Session inactive for {inactivity_minutes:.1f} minutes - clearing memory")
                await self.clear_user_memory(session_id)
                return
                
            # Check conversation turn limit
            turns = session_info.get("conversation_turns", 0)
            if turns >= self.config.max_conversation_turns:
                logger.info(f"[{session_id}] 🧹 Reached {turns} conversation turns - clearing memory")
                await self.clear_user_memory(session_id)
                return
                
        # Update session info
        await self._update_session_info(session_id)

    async def _extract_main_topic(self, text: str) -> Optional[str]:
        """Extract the main topic from a text using the LLM"""
        if not self.config.use_llm_for_query_analysis:
            return None
        
        try:
            system_prompt = "Extract the main subject or topic being discussed in the text. Return only the topic name without any explanation or additional text."
            user_prompt = f"Text: {text[:1000]}"  # Limit text length
            
            client = AsyncOpenAI(api_key=self.openai_api_key)
            response = await client.chat.completions.create(
                model=self.config.analysis_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=30     # Short response
            )
            
            topic = response.choices[0].message.content.strip()
            return topic if topic else None
        except Exception as e:
            logger.error(f"Error extracting topic: {str(e)}")
            return None

    # Add these methods to the EnhancedRAG class
    async def _get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information for the given session ID
        
        Args:
            session_id: The session ID to retrieve information for
            
        Returns:
            Dictionary with session information or None if session doesn't exist
        """
        # Initialize the session_info dictionary if it doesn't exist yet
        if not hasattr(self, "session_info"):
            self.session_info = {}
            
        return self.session_info.get(session_id)

    async def _update_session_info(self, session_id: str) -> None:
        """
        Update session information for the given session ID
        - Updates last activity timestamp
        - Increments conversation turns counter
        
        Args:
            session_id: The session ID to update information for
        """
        # Initialize the session_info dictionary if it doesn't exist yet
        if not hasattr(self, "session_info"):
            self.session_info = {}
        
        # Create session entry if it doesn't exist
        if session_id not in self.session_info:
            self.session_info[session_id] = {
                "last_activity": time.time(),
                "conversation_turns": 1
            }
        else:
            # Update existing session
            self.session_info[session_id]["last_activity"] = time.time()
            self.session_info[session_id]["conversation_turns"] += 1
            
        # Log session activity
        logger.debug(f"[{session_id}] Session updated: {self.session_info[session_id]}")

    async def _enhanced_context_retrieval(self, query: str, session_id: str, formatted_chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Enhanced context retrieval that understands follow-up questions and references
        """
        print(f"[{session_id}] 🧠 Starting enhanced context analysis...")
        
        # Step 1: Detect if this is a follow-up question
        follow_up_info = await self._detect_followup_with_enhanced_logic(query, formatted_chat_history)
        
        # Step 2: If it's a follow-up, enhance the query with context
        enhanced_query = query
        if follow_up_info["is_follow_up"]:
            enhanced_query = await self._create_context_enhanced_query(query, follow_up_info, formatted_chat_history)
            print(f"[{session_id}] 🔄 Follow-up detected! Enhanced query: '{enhanced_query}'")
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "is_follow_up": follow_up_info["is_follow_up"],
            "referring_entity": follow_up_info.get("referring_entity"),
            "context_topic": follow_up_info.get("context_topic"),
            "conversation_context": follow_up_info.get("conversation_context", "")
        }

    async def _detect_followup_with_enhanced_logic(self, query: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Enhanced follow-up detection using LLM analysis of conversation context
        """
        result = {
            "is_follow_up": False,
            "referring_entity": None,
            "context_topic": None,
            "conversation_context": ""
        }
        
        if not chat_history or len(chat_history) < 2:
            return result
        
        # Get recent conversation context
        recent_messages = chat_history[-4:]  # Last 4 messages for context
        conversation_context = self._format_conversation_for_analysis(recent_messages)
        
        # Use LLM to analyze if this is a follow-up question
        analysis_prompt = f"""
You are analyzing whether a query is a follow-up question that references something from previous conversation.

RECENT CONVERSATION:
{conversation_context}

CURRENT QUERY: "{query}"

Analyze if the current query:
1. References something mentioned in the conversation (like "it", "that", "when was it released", etc.)
2. Asks for additional information about a topic already discussed
3. Is a follow-up question that needs context from previous messages

If it IS a follow-up:
- What entity/topic is being referenced?
- What context is needed to understand the query?

Respond in JSON format:
{{
    "is_follow_up": true/false,
    "referring_entity": "the main entity/topic being referenced or null",
    "context_topic": "broader topic context or null",
    "explanation": "brief explanation of why this is/isn't a follow-up",
    "conversation_context": "relevant context needed to understand the query"
}}
"""

        try:
            if self.config.use_llm_for_query_analysis:
                response = await self.async_openai_client.chat.completions.create(
                    model=self.config.analysis_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing conversational context and detecting follow-up questions."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                analysis_text = response.choices[0].message.content.strip()
                print(f"LLM Analysis: {analysis_text}")
                
                # Parse JSON response
                try:
                    analysis_result = json.loads(analysis_text)
                    result.update(analysis_result)
                except json.JSONDecodeError:
                    # Fallback to simple detection if JSON parsing fails
                    result["is_follow_up"] = any(word in query.lower() for word in [
                        "it", "that", "this", "when was", "what is", "how does", "why does",
                        "release", "released", "launch", "launched"
                    ])
                    if result["is_follow_up"] and recent_messages:
                        # Extract entity from last AI message
                        last_ai_message = next((msg["content"] for msg in reversed(recent_messages) if msg.get("role") == "assistant"), "")
                        result["referring_entity"] = await self._extract_main_topic(last_ai_message)
            
        except Exception as e:
            print(f"Error in follow-up analysis: {e}")
            # Fallback to simple heuristic detection
            result["is_follow_up"] = any(word in query.lower() for word in [
                "it", "that", "this", "when was", "what is", "how does", "why does"
            ])
        
        return result

    def _format_conversation_for_analysis(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages for LLM analysis"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Limit content length
            if role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    async def _create_context_enhanced_query(self, original_query: str, follow_up_info: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
        """
        Create an enhanced query that includes context from previous conversation
        """
        referring_entity = follow_up_info.get("referring_entity")
        conversation_context = follow_up_info.get("conversation_context", "")
        
        if referring_entity:
            # Create a more specific query that includes the entity context
            enhanced_query = f"{original_query} about {referring_entity}"
            print(f"Enhanced query with entity context: {enhanced_query}")
            return enhanced_query
        elif conversation_context:
            # Use conversation context to enhance the query
            enhanced_query = f"{original_query}. Context: {conversation_context[:100]}"
            return enhanced_query
        
        return original_query

    async def _enhanced_extract_main_topic(self, text: str) -> Optional[str]:
        """
        Enhanced topic extraction using LLM with better prompts
        """
        if not text or len(text.strip()) < 10:
            return None
        
        # Limit text length for processing
        text_sample = text[:500]
        
        extraction_prompt = f"""
Extract the main topic, entity, or subject being discussed in this text.
Focus on:
- Product names, company names, technology names
- People's names, places
- Specific topics or concepts
- Tools, services, or platforms

Text: "{text_sample}"
"""

        try:
            response = await self.async_openai_client.chat.completions.create(
                model=self.config.analysis_model,
                messages=[
                    {"role": "system", "content": "You extract the main topic or entity from text concisely."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            topic = response.choices[0].message.content.strip()
            return topic if topic.lower() != "none" else None
            
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return None

    # Replace the existing _extract_main_topic method
    async def _extract_main_topic(self, text: str) -> Optional[str]:
        """Extract main topic from text with enhanced logic"""
        return await self._enhanced_extract_main_topic(text)

    # Enhanced memory management with better context storage
    async def _save_message_to_memory(self, session_id: str, role: str, content: str):
        """Save message to memory with enhanced context tracking"""
        try:
            memory = await self._get_user_memory(session_id)
            
            if role == "user":
                memory.add_user_message(content)
            else:  # assistant
                memory.add_ai_message(content)
            
            # Update session activity
            await self._update_session_info(session_id)
            
        except Exception as e:
            print(f"Error saving message to memory: {e}")

    # Enhanced query processing with better memory integration
    async def query_stream(
        self, session_id: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None,
        user_r2_document_keys: Optional[List[str]] = None, use_hybrid_search: Optional[bool] = None,
        llm_model_name: Optional[str] = None, system_prompt_override: Optional[str] = None,
        enable_web_search: Optional[bool] = False,
        mcp_enabled: Optional[bool] = None,      # For this specific query
        mcp_schema: Optional[str] = None,        # JSON string of the SELECTED server's config for this query
        api_keys: Optional[Dict[str, str]] = None,  # API keys to potentially inject into MCP server env
        is_new_chat: bool = False  # Add this parameter to indicate a new chat
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enhanced query streaming with better memory management and context continuity.
        """
        
        start_time = time.time()
        print(f"[{session_id}] Query at {time.strftime('%Y-%m-%d %H:%M:%S')}: {query}")
        
        # Clean up any active MCP processes for this session when starting new query
        yield {"type": "progress", "data": "🧹 Cleaning up previous sessions..."}
        await self._cleanup_mcp_processes(session_id)
        
        # Enhanced context analysis with better memory integration
        print(f"[{session_id}] 🧠 Starting enhanced context analysis...")
        
        # Manage session memory (clear if new chat, update timestamp otherwise)
        await self._manage_session_memory(session_id, is_new_chat)
        
        # Check if session exists and should be cleared due to inactivity or conversation count
        await self._manage_session_memory(session_id, is_new_chat)
        
        # Clear memory if this is a new chat
        if is_new_chat:
            await self.clear_user_memory(session_id)
            print(f"[{session_id}] Starting new chat - memory cleared")
        
        # If provided chat_history is empty but we have memory, use the memory
        formatted_chat_history = chat_history or []
        if not formatted_chat_history and session_id in self.user_memories:
            # Convert memory to formatted chat history
            memory = self.user_memories[session_id]
            for msg in memory.messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                formatted_chat_history.append({"role": role, "content": msg.content})
            print(f"[{session_id}] Using {len(formatted_chat_history)} messages from memory")
        
        # ENHANCED: Better memory integration
        formatted_chat_history = chat_history or []
        if not formatted_chat_history and session_id in self.user_memories:
            # Convert memory to formatted chat history
            memory = self.user_memories[session_id]
            for msg in memory.messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                formatted_chat_history.append({"role": role, "content": msg.content})
            print(f"[{session_id}] Using {len(formatted_chat_history)} messages from memory")
        
        # ENHANCED: Context-aware query processing
        context_info = await self._enhanced_context_retrieval(query, session_id, formatted_chat_history)
        enhanced_query = context_info["enhanced_query"]
        is_follow_up = context_info["is_follow_up"]
        referring_entity = context_info["referring_entity"]
        
        # Save user message to memory FIRST
        await self._save_message_to_memory(session_id, "user", query)
        
        # Use enhanced query for retrieval if it's a follow-up
        retrieval_query = enhanced_query if is_follow_up else query
        
        # Continue with existing MCP detection logic...
        print(f"[{session_id}] 🔍 Frontend MCP setting: {mcp_enabled}")
        detection_result = await self.detect_query_type(retrieval_query)  # Use enhanced query
        processed_query = detection_result.get("cleaned_query", retrieval_query)
        
        # ... rest of existing MCP logic remains the same ...
        
        # MCP handling (unchanged)
        if detection_result["query_type"] == "rag":
            mcp_enabled = False
            print(f"[{session_id}] ✅ FORCED MCP DISABLED: {detection_result['explanation']}")
        else:
            mcp_enabled = True
            print(f"[{session_id}] ✅ MCP ENABLED: {detection_result['explanation']}")
            
        if mcp_enabled and mcp_schema:
            print(f"[{session_id}] 🔌 MCP ENABLED - Proceeding with MCP server execution (bypassing web search)")
            async for mcp_chunk in self._handle_mcp_request(processed_query, mcp_schema, formatted_chat_history, api_keys, detection_result.get("server_name")):
                yield mcp_chunk
            return
        
        # Continue with RAG processing using enhanced query...
        print(f"\n[{session_id}] 📊 SEARCH CONFIGURATION:")
        print(f"[{session_id}] 🔄 Original query: '{query}'")
        if is_follow_up:
            print(f"[{session_id}] 🔄 Enhanced query: '{retrieval_query}'")
            print(f"[{session_id}] 🎯 Referring to: {referring_entity}")
        
        # ... rest of the existing retrieval logic remains the same, but use retrieval_query ...
        
        actual_use_hybrid_search = True
        print(f"[{session_id}] 🔄 Hybrid search: ACTIVE (BM25 Available: {HYBRID_SEARCH_AVAILABLE})")
        
        has_user_documents = bool(user_r2_document_keys)
        web_search_analysis = await self._analyze_web_search_necessity(retrieval_query, formatted_chat_history, user_r2_document_keys)

        if (web_search_analysis.get("confidence", "medium") == "high" and 
            "greeting" in web_search_analysis.get("reasoning", "").lower()):
            print(f"[{session_id}] 👋 GREETING DETECTED - Bypassing all document retrieval and context")
            
            yield {"type": "progress", "data": "Generating friendly greeting response..."}
            
            simple_greeting_generator = self._generate_simple_greeting_response(
                query, formatted_chat_history, llm_model_name
            )
            
            async for content_chunk in simple_greeting_generator:
                yield {"type": "content", "data": content_chunk}
                
            # Save AI response to memory
            full_response = ""  # You'd need to collect the response
            # await self._save_message_to_memory(session_id, "assistant", full_response)
            
            total_time = int((time.time() - start_time) * 1000)
            yield {"type": "done", "data": {"total_time_ms": total_time}}
            return

        current_model = llm_model_name or self.default_llm_model_name
        print(f"[{session_id}] 🧠 Using model: {current_model}")
        print(f"[{session_id}] {'='*80}")

        print(f"\n[{session_id}] 📝 Processing query: '{retrieval_query}'")
        
        # Initialize a list to store ALL retrieved documents
        all_retrieved_docs: List[Document] = []
        retrieval_start_time = time.time()
        
        # Notify client that retrieval has started
        yield {"type": "progress", "data": "Starting search across all available sources..."}
        
        # Create tasks to run searches in parallel
        search_tasks = []
        
        # Task 1: Get user documents
        async def get_user_docs():
            if user_session_retriever := await self._get_user_retriever(session_id):
                user_docs = await self._get_retrieved_documents(
                    user_session_retriever, retrieval_query, k_val=3,
                    is_hybrid_search_active=actual_use_hybrid_search, is_user_doc=True
                )
                if user_docs:
                    print(f"[{session_id}] 📄 Retrieved {len(user_docs)} user-specific documents")
                    return user_docs
            return []
        
        # Task 2: Get knowledge base documents
        async def get_kb_docs():
            if self.kb_retriever:
                kb_docs = await self._get_retrieved_documents(
                    self.kb_retriever, retrieval_query, k_val=5, 
                    is_hybrid_search_active=actual_use_hybrid_search
                )
                if kb_docs:
                    print(f"[{session_id}] 📚 Retrieved {len(kb_docs)} knowledge base documents")
                    return kb_docs
            return []
        
        # Task 3: Get web search documents (FIXED)
        async def get_web_docs():
            # FIXED: Check web search analysis BEFORE attempting web search
            web_search_analysis = await self._analyze_web_search_necessity(retrieval_query, formatted_chat_history, user_r2_document_keys)
            should_use_web_search = web_search_analysis.get("should_use_web_search", False)
            
            # Only proceed with web search if analysis suggests it's needed OR if explicitly enabled and documents exist
            if self.tavily_client and (should_use_web_search or (enable_web_search and (user_docs or kb_docs))):
                web_docs = await self._get_web_search_docs(retrieval_query, True, num_results=5)
                if web_docs:
                    # FIXED: Pass actual retrieved documents for similarity filtering
                    existing_docs = []
                    if user_docs:
                        existing_docs.extend(user_docs)
                    if kb_docs:
                        existing_docs.extend(kb_docs)
                    
                    # Apply similarity filtering with existing documents
                    filtered_web_docs, filter_metrics = await self._filter_web_docs_by_similarity(
                        retrieval_query, existing_docs, web_docs
                    )
                    
                    if filtered_web_docs:
                        print(f"[{session_id}] 🌐 Retrieved {len(filtered_web_docs)} web documents (filtered from {filter_metrics['original']} by {filter_metrics['threshold']} similarity)")
                        return filtered_web_docs
                    else:
                        print(f"[{session_id}] 🚫 All {filter_metrics['original']} web documents filtered out by similarity threshold")
                        return []
            else:
                if not should_use_web_search:
                    print(f"[{session_id}] 🚫 Web search disabled by analysis: {web_search_analysis['reasoning']}")
            return []
        
        # Task 4: Process attached documents
        async def process_attachments():
            if user_r2_document_keys:
                adhoc_load_tasks = [self._download_and_split_one_doc(r2_key) for r2_key in user_r2_document_keys]
                results_list_of_splits = await asyncio.gather(*adhoc_load_tasks)
                attachment_docs = []
                for splits_from_one_doc in results_list_of_splits:
                    attachment_docs.extend(splits_from_one_doc)
                if attachment_docs:
                    print(f"[{session_id}] 📎 Processed {len(user_r2_document_keys or [])} attached documents into {len(attachment_docs)} splits")
                    return attachment_docs
            return []
        
        # Execute document retrieval tasks
        user_docs_task = asyncio.create_task(get_user_docs())
        kb_docs_task = asyncio.create_task(get_kb_docs())
        
        # Wait for user and kb docs before web search
        user_docs = await user_docs_task
        kb_docs = await kb_docs_task
        
        # Now get web docs with access to user_docs and kb_docs
        web_docs = await get_web_docs()
        
        # Process attachments in parallel with other operations
        attachment_docs = await process_attachments()
        
        # Combine all results - no need for additional asyncio.gather since we already have the results
        all_retrieved_docs = []
        all_retrieved_docs.extend(user_docs)
        all_retrieved_docs.extend(kb_docs) 
        all_retrieved_docs.extend(web_docs)
        all_retrieved_docs.extend(attachment_docs)
        
        # Report on results found
        if user_docs:
            yield {"type": "progress", "data": f"Found {len(user_docs)} relevant user documents"}
        
        if kb_docs:
            yield {"type": "progress", "data": f"Found {len(kb_docs)} relevant knowledge base documents"}
        
        if web_docs:
            yield {"type": "progress", "data": f"Found {len(web_docs)} high-quality web pages (similarity filtered)"}
        
        if attachment_docs:
            yield {"type": "progress", "data": f"Processed {len(user_r2_document_keys or [])} attached documents"}

        # Deduplicate documents
        unique_docs_content = set()
        deduplicated_docs = [doc for doc in all_retrieved_docs if doc.page_content not in unique_docs_content and not unique_docs_content.add(doc.page_content)]
        all_retrieved_docs = deduplicated_docs

        retrieval_time_ms = int((time.time() - retrieval_start_time) * 1000)
        print(f"\n[{session_id}] 🔍 Retrieved {len(all_retrieved_docs)} total unique documents in {retrieval_time_ms}ms")
        yield {"type": "progress", "data": f"Combined {len(all_retrieved_docs)} relevant documents from all sources in {retrieval_time_ms}ms"}

        # Enhanced source review with conversation context
        yield {"type": "progress", "data": "🧠 Analyzing sources with conversational context..."}
        current_system_prompt = system_prompt_override or self.default_system_prompt
        
        # Use enhanced review with chat history and follow-up information
        reviewed_docs = await self._enhanced_review_combined_sources(
            retrieval_query, all_retrieved_docs, current_system_prompt, formatted_chat_history
        )
        
        if is_follow_up:
            yield {"type": "progress", "data": f"🔄 Processing follow-up question about: {referring_entity or 'previous topic'}"}
        
        # Continue with document selection and LLM generation...
        final_docs_for_llm = []
        for source_type in ["user_docs", "kb_docs", "web_docs"]:
            docs = reviewed_docs.get(source_type, [])
            if docs:
                # Ensure we have Document objects, not nested lists
                for doc in docs[:3]:  # Take top 3 from each source
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        final_docs_for_llm.append(doc)
                    elif isinstance(doc, list):
                        # If it's a nested list, flatten it
                        for nested_doc in doc:
                            if hasattr(nested_doc, 'page_content') and hasattr(nested_doc, 'metadata'):
                                final_docs_for_llm.append(nested_doc)
        
        print(f"[{session_id}] 📋 Prepared {len(final_docs_for_llm)} documents for LLM context")
        
        # Generate response with enhanced context awareness
        yield {"type": "progress", "data": "Generating response with conversational context..."}
        
        # Enhanced LLM generation with conversation context
        llm_generator = await self._generate_enhanced_llm_response(
            session_id, query, final_docs_for_llm, formatted_chat_history,
            llm_model_name, system_prompt_override, stream=True,
            is_follow_up=is_follow_up, referring_entity=referring_entity
        )
        
        # Collect the full response to save to memory
        full_response = ""
        async for response_chunk in llm_generator:
            full_response += response_chunk
            yield {"type": "content", "data": response_chunk}
        
        # Save AI response to memory
        await self._save_message_to_memory(session_id, "assistant", full_response)
        
        total_time = int((time.time() - start_time) * 1000)
        yield {"type": "done", "data": {"total_time_ms": total_time}}

    # Enhanced prompt creation for better conversational context
    async def _create_conversational_prompt(self, query: str, documents: List[Document], 
                                          chat_history: List[Dict[str, str]], 
                                          is_follow_up: bool = False, 
                                          referring_entity: Optional[str] = None) -> str:
        """
        Create enhanced prompt that includes conversation context
        """
        # Format documents with error handling
        context_parts = []
        for doc in documents:
            try:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content
                    context_parts.append(f"Source: {source}\n{content}")
                else:
                    print(f"Warning: Invalid document object: {type(doc)}")
            except Exception as e:
                print(f"Error processing document: {e}")
                continue
        
        context = "\n\n".join(context_parts)
        
        # Format recent conversation
        conversation_context = ""
        if chat_history:
            recent_history = chat_history[-4:]  # Last 4 exchanges
            for msg in recent_history:
                role = "Human" if msg.get("role") == "user" else "Assistant"
                conversation_context += f"\n{role}: {msg.get('content', '')[:150]}..."
        
        # Create the enhanced prompt
        if is_follow_up and referring_entity:
            prompt = f"""You are answering a FOLLOW-UP question in an ongoing conversation.

CONVERSATION CONTEXT:{conversation_context}

The user is asking a follow-up question about: {referring_entity}

RELEVANT INFORMATION:
{context}

CURRENT QUESTION: {query}

IMPORTANT:
- This is a follow-up question referring to "{referring_entity}" from the previous conversation
- Provide specific information about what the user is asking
- Maintain continuity with the previous discussion
- If asking about release dates, launch dates, or timing, be specific with dates if available
- Reference the previous topic naturally in your response

Answer the follow-up question clearly and specifically:"""
        else:
            prompt = f"""You are a helpful AI assistant with access to relevant information.

{f"CONVERSATION CONTEXT:{conversation_context}" if conversation_context else ""}

RELEVANT INFORMATION:
{context}

USER QUESTION: {query}

Provide a comprehensive, accurate answer based on the available information. If specific details like dates, versions, or technical specifications are requested, include them if available in the sources."""
        
        return prompt

    async def _generate_enhanced_llm_response(
        self, session_id: str, query: str, all_context_docs: List[Document],
        chat_history_messages: List[Dict[str, str]], llm_model_name_override: Optional[str],
        system_prompt_override: Optional[str], stream: bool = False,
        is_follow_up: bool = False, referring_entity: Optional[str] = None
    ) -> Union[AsyncGenerator[str, None], str]:
        """
        Enhanced LLM response generation with conversational awareness
        """
        
        # Create conversational prompt
        enhanced_prompt = await self._create_conversational_prompt(
            query, all_context_docs, chat_history_messages, is_follow_up, referring_entity
        )
        
        # Use the existing LLM generation logic but with enhanced prompt
        messages = [
            {"role": "system", "content": system_prompt_override or self.default_system_prompt},
            {"role": "user", "content": enhanced_prompt}
        ]
        
        # Use existing model selection logic from the original method
        current_model = llm_model_name_override or self.default_llm_model_name
        # Normalize model name to lowercase for OpenAI compatibility
        current_model = current_model.lower() if current_model else current_model
        
        try:
            if stream:
                # Stream response
                response = await self.async_openai_client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=self.default_temperature,
                    stream=True
                )
                
                async def enhanced_stream_generator():
                    async for chunk in response:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return enhanced_stream_generator()
            else:
                # Non-stream response
                response = await self.async_openai_client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=self.default_temperature
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error in enhanced LLM generation: {e}")
            # Fallback to original method
            return await self._generate_llm_response(
                session_id, query, all_context_docs, chat_history_messages,
                llm_model_name_override, system_prompt_override, stream
            )

    async def _cleanup_mcp_processes(self, session_id: str = None):
        """Clean up active MCP processes for a specific session or all sessions."""
        async with self.mcp_cleanup_lock:
            if session_id:
                # Clean up processes for specific session
                session_processes = [k for k in self.active_mcp_processes.keys() if k.startswith(session_id)]
                for process_key in session_processes:
                    process = self.active_mcp_processes.get(process_key)
                    if process and process.returncode is None:
                        try:
                            print(f"🧹 Terminating MCP process for session {session_id}")
                            process.terminate()
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            print(f"🔪 Force killing MCP process for session {session_id}")
                            process.kill()
                            await process.wait()
                        except Exception as e:
                            print(f"Error cleaning up MCP process: {e}")
                    del self.active_mcp_processes[process_key]
            else:
                # Clean up all MCP processes
                for process_key, process in list(self.active_mcp_processes.items()):
                    if process and process.returncode is None:
                        try:
                            process.terminate()
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                        except Exception as e:
                            print(f"Error cleaning up MCP process {process_key}: {e}")
                self.active_mcp_processes.clear()

async def main_test_rag_qdrant():
    print("Ensure QDRANT_URL and OPENAI_API_KEY are set in .env for this test.")
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("QDRANT_URL")):
        print("Skipping test: OPENAI_API_KEY or QDRANT_URL not set.")
        return

    class DummyR2Storage:
        async def download_file(self, key: str, local_path: str) -> bool:
            with open(local_path, "w") as f:
                f.write("This is a test document for RAG.")
            return True

        async def upload_file(self, file_data, filename: str, is_user_doc: bool = False):
            return True, f"test/{filename}"

        async def download_file_from_url(self, url: str):
            return True, f"test/doc_from_url_{url[-10:]}"

    rag = EnhancedRAG(
        gpt_id="test_gpt",
        r2_storage_client=DummyR2Storage(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )

    await rag.update_knowledge_base_from_r2(["test/doc1.txt"])
    session_id = "test_session"
    await rag.update_user_documents_from_r2(session_id, ["test/doc2.txt"])

    async for chunk in rag.query_stream(session_id, "What is in the test document?", enable_web_search=False):
        print(chunk)

if __name__ == "__main__":
    print(f"rag.py loaded. Qdrant URL: {os.getenv('QDRANT_URL')}. Tavily available: {TAVILY_AVAILABLE}. BM25 available: {HYBRID_SEARCH_AVAILABLE}")

# Make BM25_AVAILABLE available for backwards compatibility
BM25_AVAILABLE = HYBRID_SEARCH_AVAILABLE

# Configure logger
logger = logging.getLogger(__name__)
