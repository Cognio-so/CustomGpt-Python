import os
import shutil
import asyncio
import time
import json
import base64
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from urllib.parse import urlparse
import uuid
import httpx
import subprocess
import aiohttp

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

# Document Loaders & Transformers
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, BSHTMLLoader, TextLoader, UnstructuredURLLoader
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

class EnhancedRAG:
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
        default_use_hybrid_search: bool = True,  # Default is already True
        # New params for initial full MCP config for this GPT instance
        initial_mcp_enabled_config: Optional[bool] = None,
        initial_mcp_schema_config: Optional[str] = None
    ):
        self.gpt_id = gpt_id
        self.r2_storage = r2_storage_client
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("Warning: OpenAI API key not found. Some functionalities might be limited.")
            # raise ValueError("OpenAI API key must be provided or set via OPENAI_API_KEY environment variable.")

        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        self.default_llm_model_name = default_llm_model_name
        self.default_system_prompt = default_system_prompt or (
            "You are a helpful and meticulous AI assistant. "
            "Provide comprehensive, detailed, and accurate answers based *solely* on the context provided. "
            "Structure your response clearly using Markdown. "
            "Use headings (#, ##, ###), subheadings, bullet points (* or -), and numbered lists (1., 2.) where appropriate to improve readability. "
            "For code examples, use Markdown code blocks with language specification (e.g., ```python ... ```). "
            "Feel free to use relevant emojis to make the content more engaging, but do so sparingly and appropriately. "
            "If the context is insufficient or does not contain the answer, clearly state that. "
            "Cite the source of your information if possible (e.g., 'According to document X...'). "
            "Do not make assumptions or use external knowledge beyond the provided context. "
            "Ensure your response is as lengthy and detailed as necessary to fully answer the query, up to the allowed token limit."
        )
        self.default_temperature = default_temperature
        self.max_tokens_llm = 32000 
        # IMPORTANT: Force hybrid search to always be True regardless of input setting
        self.default_use_hybrid_search = True
        print(f"✅ Hybrid search FORCE ENABLED for all queries regardless of config setting")

        # Store the initial full MCP configuration for this GPT
        self.gpt_mcp_enabled_config = initial_mcp_enabled_config
        self.gpt_mcp_full_schema_str = initial_mcp_schema_config # JSON string of all MCP servers for this GPT

        self.temp_processing_path = os.path.join(temp_processing_path, self.gpt_id) # Per-GPT temp path
        os.makedirs(self.temp_processing_path, exist_ok=True)

        self.embeddings_model = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
        self.async_openai_client = AsyncOpenAI(
            api_key=self.openai_api_key, timeout=timeout_config, max_retries=1
        )

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
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
        self.openrouter_url = "https://openrouter.ai/api/v1"
        self.openrouter_client = None
        if OPENROUTER_AVAILABLE and self.openrouter_api_key:
            self.openrouter_client = AsyncOpenAI(
                api_key=self.openrouter_api_key, base_url=self.openrouter_url,
                timeout=timeout_config, max_retries=1
            )
            print(f"✅ OpenRouter client initialized for GPT '{self.gpt_id}'")
        else:
            print(f"ℹ️ OpenRouter API key not provided or client not available. OpenRouter disabled for GPT '{self.gpt_id}'.")

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
                is_our_r2_url = self.r2_storage.account_id and self.r2_storage.bucket_name and \
                                f"{self.r2_storage.bucket_name}.{self.r2_storage.account_id}.r2.cloudflarestorage.com" in parsed_url.netloc
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
                    asyncio.to_thread(self.r2_storage.download_file, r2_object_key_to_download, temp_file_path)
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
                
                # Check if it's an image file by extension
                is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
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
                            loader = PyPDFLoader(temp_file_path)
                        elif ext == ".docx": 
                            loader = Docx2txtLoader(temp_file_path)
                        elif ext in [".html", ".htm"]: 
                            loader = BSHTMLLoader(temp_file_path, open_encoding='utf-8')
                        else: 
                            loader = TextLoader(temp_file_path, autodetect_encoding=True)
                        
                        # More efficient document loading using parallel processing
                        load_task = asyncio.create_task(asyncio.to_thread(loader.load))
                        try:
                            loaded_docs = await asyncio.wait_for(load_task, timeout=10.0)
                            # Special handling for HTML if needed
                            if ext in [".html", ".htm"] and loaded_docs:
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
                    claude_model_to_call = "claude-3-5-sonnet-20240620" # Default to Claude 3.5 Sonnet
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
        user_collection_name = self._get_user_qdrant_collection_name(session_id)
        try:
            print(f"Attempting to delete Qdrant collection: '{user_collection_name}' for session '{session_id}'")
            # Ensure the client is available for the deletion call
            if not self.qdrant_client:
                print(f"Qdrant client not initialized. Cannot delete collection {user_collection_name}.")
            else:
                await asyncio.to_thread(self.qdrant_client.delete_collection, collection_name=user_collection_name)
                print(f"Qdrant collection '{user_collection_name}' deleted.")
        except Exception as e:
            if "not found" in str(e).lower() or \
               (hasattr(e, "status_code") and e.status_code == 404) or \
               "doesn't exist" in str(e).lower() or \
               "collectionnotfound" in str(type(e)).lower() or \
               (hasattr(e, "error_code") and "collection_not_found" in str(e.error_code).lower()): # More robust error checking
                print(f"Qdrant collection '{user_collection_name}' not found during clear, no need to delete.")
            else:
                print(f"Error deleting Qdrant collection '{user_collection_name}': {e} (Type: {type(e)})")
        
        if session_id in self.user_collection_retrievers: del self.user_collection_retrievers[session_id]
        if session_id in self.user_memories: del self.user_memories[session_id]
        print(f"User session context (retriever, memory, Qdrant collection artifacts) cleared for session_id: {session_id}")
        # After deleting the collection, it's good practice to ensure a new empty one is ready if needed immediately.
        # This will be handled by _get_qdrant_retriever_sync when it's called next.

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
        
        # No document limiting - use all documents
        # Removed: max_docs = 2 and documents[:max_docs]
        
        # No content truncation
        # Removed: truncation of document content
        
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
            current_llm_model = "llama3-8b-8192"
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

        # Prepare user message
        user_query_message_content = (
            f"📚 **CONTEXT:**\n{context_str}\n\n"
            f"Based on the above context and any relevant chat history, provide a detailed, well-structured response to this query:\n\n"
            f"**QUERY:** {query}\n\n"
            f"Requirements for your response:\n"
            f"1. 🎯 Start with a relevant emoji and descriptive headline\n"
            f"2. 📋 Organize with clear headings and subheadings\n"
            f"3. 📊 Include bullet points or numbered lists where appropriate\n"
            f"4. 💡 Highlight key insights or important information\n"
            f"5. 📝 Reference specific information from the provided documents\n"
            f"6. 🔍 Use appropriate emojis (about 1-2 per section) to make content engaging\n"
            f"7. 📚 Make your response comprehensive, detailed and precise\n"
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
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
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
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
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
                groq_model = "llama3-70b-8192"  # Higher quality Llama model available on Groq
                print(f"Using Groq with llama3-70b-8192 model (as fallback for Llama 4 Scout)")
            else:
                groq_model = "llama3-8b-8192"  # Keep default for Llama 3
                print(f"Using Groq with llama3-8b-8192 model")
            
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
        mcp_enabled: Optional[bool] = None,      # For this specific query
        mcp_schema: Optional[str] = None,        # JSON string of the SELECTED server's config for this query
        api_keys: Optional[Dict[str, str]] = None,  # API keys to potentially inject into MCP server env
        is_new_chat: bool = False  # Add this parameter to indicate a new chat
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()
        
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
        
        # If mcp_enabled is not explicitly set, auto-detect using tool calling
        if mcp_enabled is None:
            # Auto-detect query type
            detection_result = await self.detect_query_type(query)
            
            # Set MCP mode based on detection
            if detection_result["type"] == "mcp":
                mcp_enabled = True
                
                # If MCP is enabled but no schema is provided, use the full schema from this GPT
                if mcp_schema is None and self.gpt_mcp_full_schema_str:
                    # Here we would need to parse the full schema and select the appropriate server
                    # This is a simplified approach - in a real implementation, you might want to
                    # parse the schema and select the most appropriate server based on the query
                    mcp_schema = self.gpt_mcp_full_schema_str
                    
                    # Log the auto-detection
                    print(f"Auto-detected MCP query: {detection_result['explanation']}")
                    if detection_result.get("server_name"):
                        print(f"Requested server: {detection_result['server_name']}")
            else:
                mcp_enabled = False
                print(f"Auto-detected RAG query: {detection_result['explanation']}")
        
        # If MCP is enabled for this query and a specific server schema is provided
        if mcp_enabled and mcp_schema:
            try:
                chat_history_processed = formatted_chat_history or []
                # Handle MCP request with the selected server
                async for chunk in self._handle_mcp_request(
                    query=query,
                    selected_server_config_str=mcp_schema,
                    chat_history=chat_history_processed,
                    api_keys_for_mcp=api_keys
                ):
                    yield chunk
                return
            except Exception as e:
                error_message = f"Error processing MCP request: {str(e)}"
                print(error_message)
                yield {"type": "error", "text": error_message}
                return
        
        # If MCP is not enabled, proceed with RAG processing
        print(f"\n[{session_id}] 📊 SEARCH CONFIGURATION:")
        
        # Always enable hybrid search unless explicitly disabled
        actual_use_hybrid_search = True  # Force to True always
        print(f"[{session_id}] 🔄 Hybrid search: ACTIVE (BM25 Available: {HYBRID_SEARCH_AVAILABLE})")
        
        # Always enable web search if Tavily is available and not explicitly disabled
        effective_enable_web_search = enable_web_search
        if effective_enable_web_search is None:  
            effective_enable_web_search = self.tavily_client is not None
        
        if effective_enable_web_search:
            if self.tavily_client: print(f"[{session_id}] 🌐 Web search: ENABLED with Tavily")
            else: print(f"[{session_id}] 🌐 Web search: REQUESTED but Tavily client not available/configured")
        else:
            print(f"[{session_id}] 🌐 Web search: DISABLED")
        
        current_model = llm_model_name or self.default_llm_model_name
        print(f"[{session_id}] 🧠 Using model: {current_model}")
        print(f"[{session_id}] {'='*80}")

        retrieval_query = query
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
        
        # Task 3: Get web search documents
        async def get_web_docs():
            if effective_enable_web_search and self.tavily_client:
                web_docs = await self._get_web_search_docs(retrieval_query, True, num_results=4)
                if web_docs:
                    print(f"[{session_id}] 🌐 Retrieved {len(web_docs)} web search documents")
                    return web_docs
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
                    print(f"[{session_id}] 📎 Processed {len(user_r2_document_keys)} attached documents into {len(attachment_docs)} splits")
                    return attachment_docs
            return []
        
        # Run all search tasks in parallel
        search_tasks.append(get_user_docs())
        search_tasks.append(get_kb_docs())
        search_tasks.append(get_web_docs())
        search_tasks.append(process_attachments())
        
        # Execute all search tasks concurrently
        yield {"type": "progress", "data": "Searching across all sources in parallel..."}
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine all results
        for docs in search_results:
            all_retrieved_docs.extend(docs)
        
        # Report on results found
        if search_results[0]:  # user docs
            yield {"type": "progress", "data": f"Found {len(search_results[0])} relevant user documents"}
        
        if search_results[1]:  # kb docs
            yield {"type": "progress", "data": f"Found {len(search_results[1])} relevant knowledge base documents"}
        
        if search_results[2]:  # web docs
            yield {"type": "progress", "data": f"Found {len(search_results[2])} relevant web pages"}
        
        if search_results[3]:  # attachment docs
            yield {"type": "progress", "data": f"Processed {len(user_r2_document_keys or [])} attached documents"}

        # Deduplicate documents
        unique_docs_content = set()
        deduplicated_docs = [doc for doc in all_retrieved_docs if doc.page_content not in unique_docs_content and not unique_docs_content.add(doc.page_content)]
        all_retrieved_docs = deduplicated_docs

        retrieval_time_ms = int((time.time() - retrieval_start_time) * 1000)
        print(f"\n[{session_id}] 🔍 Retrieved {len(all_retrieved_docs)} total unique documents in {retrieval_time_ms}ms")
        yield {"type": "progress", "data": f"Combined {len(all_retrieved_docs)} relevant documents from all sources in {retrieval_time_ms}ms"}

        # NEW: Add reviewer to analyze and prioritize documents based on query intent
        yield {"type": "progress", "data": "Analyzing sources for relevance to your query..."}
        current_system_prompt = system_prompt_override or self.default_system_prompt
        reviewed_docs = await self._review_combined_sources(query, all_retrieved_docs, current_system_prompt)
        
        # Prioritize documents based on the review
        final_docs_for_llm = []
        
        # Always include user uploaded documents first (highest priority)
        if reviewed_docs["user_docs"]:
            final_docs_for_llm.extend(reviewed_docs["user_docs"])
            yield {"type": "progress", "data": "Prioritizing your uploaded documents for this query"}
        
        # Add KB documents next
        if reviewed_docs["kb_docs"]:
            final_docs_for_llm.extend(reviewed_docs["kb_docs"])
            yield {"type": "progress", "data": "Including knowledge base information"}
        
        # Only include web documents if the query suggests web search is needed
        # or if there are no user/KB docs available
        if (reviewed_docs["is_web_search_query"] or (not reviewed_docs["user_docs"] and not reviewed_docs["kb_docs"])) and reviewed_docs["web_docs"]:
            final_docs_for_llm.extend(reviewed_docs["web_docs"])
            yield {"type": "progress", "data": "Adding relevant web search results"}
        
        print(f"\n[{session_id}] 🧠 Starting LLM stream generation with {len(final_docs_for_llm)} prioritized documents...")
        yield {"type": "progress", "data": "Generating response based on the most relevant information..."}
        
        llm_stream_generator = await self._generate_llm_response(
            session_id, query, final_docs_for_llm, formatted_chat_history,
            llm_model_name, system_prompt_override, stream=True
        )
        
        print(f"[{session_id}] 🔄 LLM stream initialized, beginning content streaming")
        async for content_chunk in llm_stream_generator:
            yield {"type": "content", "data": content_chunk}
        
        print(f"[{session_id}] ✅ Stream complete, sending done signal")
        total_time = int((time.time() - start_time) * 1000)
        print(f"[{session_id}] ⏱️ Total processing time: {total_time}ms")
        yield {"type": "done", "data": {"total_time_ms": total_time}}
        print(f"[{session_id}] {'='*80}\n")

    async def _handle_mcp_request(
        self,
        query: str,
        selected_server_config_str: str,
        chat_history: List[Dict[str, str]],
        api_keys_for_mcp: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a request using a single, pre-selected MCP server configuration."""
        server_name_for_log = "selected-mcp-server"
        try:
            selected_server_config = json.loads(selected_server_config_str)
            # Attempt to get a name for logging
            server_name_for_log = selected_server_config.get("name", server_name_for_log) 
            print(f"Executing single MCP server: {server_name_for_log} for query: {query[:30]}...")

            # Prepare environment variables - START WITH THE SERVER'S CONFIG
            # Don't automatically add environment variables the server didn't ask for
            effective_env_vars = {}
            
            # First, get the environment variables specified in the server config
            if "env" in selected_server_config and isinstance(selected_server_config["env"], dict):
                effective_env_vars = selected_server_config["env"].copy()

            # Next, add API keys from frontend ONLY if they match keys specified in the config
            if api_keys_for_mcp:
                # Only add keys the server configuration expects
                keys_to_add = {}
                for key_name, key_value in api_keys_for_mcp.items():
                    # Only add frontend keys that are specified in server config env
                    if key_name in effective_env_vars:
                        keys_to_add[key_name] = key_value
                
                if keys_to_add:
                    print(f"Merging provided API keys into MCP environment: {list(keys_to_add.keys())}")
                    effective_env_vars.update(keys_to_add)
            
            # Create the server_config structure for execution
            command_config_for_execution = {
                "command": selected_server_config.get("command"),
                "args": selected_server_config.get("args", []),
                "env": effective_env_vars 
            }

            async for response_chunk_str in self._execute_mcp_server_properly(
                server_name=server_name_for_log, 
                server_config=command_config_for_execution, 
                query=query,
                chat_history=chat_history
            ):
                yield {"type": "content", "data": response_chunk_str}
            
            yield {"type": "done", "data": f"MCP execution for '{server_name_for_log}' completed."}

        except json.JSONDecodeError as e_json:
            error_msg = f"Invalid MCP server configuration provided: {str(e_json)}"
            print(f"JSONDecodeError in _handle_mcp_request for '{server_name_for_log}': {error_msg}")
            yield {"type": "error", "error": error_msg}
            yield {"type": "done", "data": "MCP execution failed due to config error."}
        except Exception as e:
            error_msg = f"Failed to process MCP request with server '{server_name_for_log}': {str(e)}"
            print(f"Exception in _handle_mcp_request for '{server_name_for_log}': {error_msg}")
            import traceback
            traceback.print_exc()
            yield {"type": "error", "error": error_msg}
            yield {"type": "done", "data": "MCP execution failed."}

    async def _execute_mcp_server_properly(self, server_name: str, server_config: Dict[str, Any], query: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Actually execute the MCP server command and stream the real response.
           server_config is expected to have 'command', 'args', and 'env'.
        """
        try:
            command = server_config.get("command")
            args = server_config.get("args", [])
            # env_vars are already prepared and merged in server_config["env"]
            env_vars = server_config.get("env", {}) 
            
            if not command:
                raise ValueError(f"No command specified for MCP server '{server_name}'")
            
            print(f"Preparing to execute MCP server '{server_name}' with command: '{command}'")
            if env_vars:
                # Avoid logging sensitive values if any are present in env_vars
                print(f"  with custom environment variables: {list(env_vars.keys())}")

            async for chunk in self._execute_generic_mcp_server(command, args, env_vars, query, chat_history):
                yield chunk
                
        except Exception as e:
            print(f"Error in _execute_mcp_server_properly for '{server_name}': {e}")
            yield f"Error executing MCP server '{server_name}': {str(e)}"

    # _execute_generic_mcp_server remains largely the same.
    # Ensure it logs the env var keys being used for debugging, not values.
    async def _execute_generic_mcp_server(self, command: str, args: List[str], env_vars: Dict[str, str], query: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        process = None
        try:
            # Create a copy of the current environment and update it with custom variables from frontend
            proc_env = os.environ.copy()
            if env_vars:
                print(f"Updating process environment with keys: {list(env_vars.keys())}")
            proc_env.update(env_vars)
            
            # FIX FOR WINDOWS: Use correct command format
            if command == "npx" and os.name == 'nt':  # Windows system
                npx_cmd = shutil.which("npx.cmd") or shutil.which("npx")
                if npx_cmd:
                    command = npx_cmd
                    print(f"Found npx at: {command}")
                else:
                    yield "Error: npx command not found. Make sure Node.js is installed and in your PATH."
                    return
            
            print(f"Executing command: {command} {' '.join(args)}")
            
            # Prepare the full command
            full_command = [command] + args if args else [command]
            
            # Create the process with the environment variables
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                shell=False
            )
            
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
                
                # Wait for initialize response
                init_response_line = await process.stdout.readline()
                if init_response_line:
                    try:
                        init_response = json.loads(init_response_line.decode().strip())
                        print(f"MCP server initialized: {init_response}")
                    except json.JSONDecodeError:
                        print(f"Invalid initialize response: {init_response_line.decode().strip()}")
                
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
                
                # Wait for tools list response
                tools_response_line = await process.stdout.readline()
                available_tools = []
                if tools_response_line:
                    try:
                        tools_response = json.loads(tools_response_line.decode().strip())
                        if "result" in tools_response and "tools" in tools_response["result"]:
                            available_tools = tools_response["result"]["tools"]
                            print(f"Available tools: {[tool.get('name', 'unknown') for tool in available_tools]}")
                    except json.JSONDecodeError:
                        print(f"Invalid tools response: {tools_response_line.decode().strip()}")
                
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
                    
                    # Prepare tool call arguments dynamically based on tool schema
                    tool_schema = tool_to_use.get("inputSchema", {})
                    tool_properties = tool_schema.get("properties", {})
                    
                    tool_arguments = {}
                    
                    # Check if the tool expects a 'messages' parameter
                    if "messages" in tool_properties:
                        tool_arguments["messages"] = messages
                    # Check if the tool expects a 'query' parameter
                    elif "query" in tool_properties:
                        tool_arguments["query"] = query
                    # Check if the tool expects a 'question' parameter
                    elif "question" in tool_properties:
                        tool_arguments["question"] = query
                    # Check if the tool expects a 'prompt' parameter
                    elif "prompt" in tool_properties:
                        tool_arguments["prompt"] = query
                    # If no recognized parameter, try with the first required parameter
                    else:
                        required_params = tool_schema.get("required", [])
                        if required_params:
                            # Use the first required parameter with the query
                            first_param = required_params[0]
                            if first_param in ["text", "input", "content"]:
                                tool_arguments[first_param] = query
                            else:
                                # If messages format might be expected, try it
                                tool_arguments[first_param] = messages if "message" in first_param.lower() else query
                        else:
                            # Fallback: try common parameter names
                            tool_arguments["query"] = query
                    
                    # Add any other optional parameters with sensible defaults (only if they exist in schema)
                    if "model" in tool_properties and "model" not in tool_arguments:
                        # Don't add hardcoded model - let the server use its default
                        pass
                    
                    if "max_tokens" in tool_properties and "max_tokens" not in tool_arguments:
                        # Don't add hardcoded max_tokens - let the server use its default
                        pass
                    
                    if "temperature" in tool_properties and "temperature" not in tool_arguments:
                        # Don't add hardcoded temperature - let the server use its default
                        pass
                    
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
                    
                    # Read the tool call response
                    tool_response_line = await process.stdout.readline()
                    if tool_response_line:
                        try:
                            tool_response = json.loads(tool_response_line.decode().strip())
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
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse tool response: {e}")
                            yield tool_response_line.decode().strip()
                else:
                    yield "No tools available from MCP server"
                
                # Close stdin to signal we're done
                process.stdin.close()
            
            # Wait for the process to complete
            await process.wait()
            
            if process.returncode != 0:
                print(f"MCP server process exited with code {process.returncode}")

        except Exception as e:
            print(f"Error in _execute_generic_mcp_server: {e}")
            import traceback
            traceback.print_exc()
            yield f"Error executing MCP server: {str(e)}"
            
        finally:
            if process:
                try:
                    if process.stdin and not process.stdin.is_closing():
                        process.stdin.close()
                    # Wait for process to terminate, with a timeout
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    print(f"Timeout waiting for MCP process {process.pid} to terminate. Killing.")
                    if process.returncode is None:  # Still running
                        process.kill()
                        await process.wait()  # Ensure it's killed
                except Exception as e_proc:
                    print(f"Exception during MCP process cleanup: {e_proc}")


    # Store the implementation as _original_execute_generic_mcp_server
    _original_execute_generic_mcp_server = _execute_generic_mcp_server

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
        if user_session_docs: all_retrieved_docs.extend(user_session_docs)

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
        await self.clear_knowledge_base()
        active_session_ids = list(self.user_collection_retrievers.keys())
        for session_id in active_session_ids:
            await self.clear_user_session_context(session_id)
        self.user_collection_retrievers.clear(); self.user_memories.clear()
        if os.path.exists(self.temp_processing_path):
            try:
                await asyncio.to_thread(shutil.rmtree, self.temp_processing_path)
                os.makedirs(self.temp_processing_path, exist_ok=True)
            except Exception as e: print(f"Error clearing temp path '{self.temp_processing_path}': {e}")
        print(f"All context (KB, all user sessions, temp files) cleared for gpt_id '{self.gpt_id}'.")

    async def detect_query_type(self, query: str) -> Dict[str, Any]:
        """
        Automatically detect whether a query should be handled by RAG or MCP functionality.
        Uses LangChain's tool calling to let the LLM make the decision based on query content.
        
        Args:
            query: The user's query string
            
        Returns:
            Dictionary with 'type' and other relevant information
        """
        tools = [RAGQueryTool, MCPServerQueryTool]
        
        # Create a model instance for tool detection
        detection_model = AsyncOpenAI(
            api_key=self.openai_api_key,
            model="gpt-4o",  # Using a capable model for accurate detection
            temperature=0
        )
        
        # Use LangChain's bind_tools function
        model_with_tools = detection_model.bind_tools(tools)
        
        # Create a system prompt that explains the task
        system_message = (
            "You are a query router that determines whether a user's query should be processed using "
            "RAG (Retrieval Augmented Generation) or MCP (Model Context Protocol) server functionality. "
            "\n\n"
            "- Use RAG for general information queries, factual questions, or anything that would benefit from searching a knowledge base. "
            "- Use MCP when the user explicitly mentions an MCP server, asks about server functionality, or needs to perform actions "
            "that would require running an external program or service. "
            "\n\n"
            "Analyze the query carefully and choose the appropriate processing method."
        )
        
        # Create messages for the model
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        # Invoke the model
        response = await model_with_tools.ainvoke(messages)
        
        # Check if the model made a tool call
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "RAGQueryTool":
                return {
                    "type": "rag",
                    "explanation": tool_args.get("explanation", "General information query")
                }
            elif tool_name == "MCPServerQueryTool":
                return {
                    "type": "mcp",
                    "server_name": tool_args.get("server_name", ""),
                    "explanation": tool_args.get("explanation", "MCP server query")
                }
        
        # Default to RAG if no tool call was made
        return {
            "type": "rag",
            "explanation": "Defaulting to RAG for general information processing"
        }

    # Add this new method to the EnhancedRAG class
    async def clear_user_memory(self, session_id: str):
        """Clear the chat memory for a specific session but keep documents"""
        if session_id in self.user_memories:
            del self.user_memories[session_id]
            print(f"Chat memory cleared for session_id: {session_id}")
        
        # Initialize a fresh memory for this session
        await self._get_user_memory(session_id)
        return True

    async def _review_combined_sources(self, query: str, all_docs: List[Document], system_prompt: str) -> Dict[str, List[Document]]:
        """
        Review and prioritize documents from different sources based on relevance to the query.
        
        Args:
            query: The user's query
            all_docs: All retrieved documents from various sources
            system_prompt: The system prompt to use for guidance
        
        Returns:
            Dictionary with categorized and prioritized documents
        """
        # Organize documents by source type
        kb_docs = []
        user_docs = []
        web_docs = []
        
        # Check if the query contains web search keywords
        web_search_keywords = [
            "latest", "recent", "news", "current", "today", "update", 
            "online", "internet", "web", "search", "find online", 
            "look up", "google", "website", "2023", "2024"
        ]
        
        # Check if query explicitly asks for web search
        is_web_search_query = any(keyword.lower() in query.lower() for keyword in web_search_keywords)
        
        print(f"Query analyzed for web search relevance: {'WEB SEARCH INDICATED' if is_web_search_query else 'KB/USER DOCS PREFERRED'}")
        
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
        
        # Prioritize documents based on query intent
        result = {
            "user_docs": user_docs,
            "kb_docs": kb_docs,
            "web_docs": web_docs,
            "is_web_search_query": is_web_search_query
        }
        
        print(f"Document review results: {len(user_docs)} user docs, {len(kb_docs)} KB docs, {len(web_docs)} web docs")
        return result

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