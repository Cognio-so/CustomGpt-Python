import os
import shutil
import asyncio
import time
import json
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from urllib.parse import urlparse
import uuid
import httpx

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

# Web Search (Tavily)
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    AsyncTavilyClient = None
    print("Tavily Python SDK not found. Web search will be disabled.")

# BM25 (Optional)
try:
    from langchain_community.retrievers import BM25Retriever
    from rank_bm25 import OkapiBM25
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("BM25Retriever or rank_bm25 package not found. Hybrid search with BM25 will be limited.")

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

load_dotenv()

# Vector params for OpenAI's text-embedding-ada-002
QDRANT_VECTOR_PARAMS = qdrant_models.VectorParams(size=1536, distance=qdrant_models.Distance.COSINE)
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"

class EnhancedRAG:
    def __init__(
        self,
        gpt_id: str,
        r2_storage_client: CloudflareR2Storage,
        openai_api_key: str,
        default_llm_model_name: str = "gpt-4o",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        temp_processing_path: str = "local_rag_data/temp_downloads",
        tavily_api_key: Optional[str] = None,
        default_system_prompt: Optional[str] = None,
        default_temperature: float = 0.2,
        default_use_hybrid_search: bool = False,
    ):
        self.gpt_id = gpt_id
        self.r2_storage = r2_storage_client
        self.openai_api_key = openai_api_key
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
        self.max_tokens_llm = 32000  # Maximum for most models, will be overridden by API limits
        self.default_use_hybrid_search = default_use_hybrid_search

        self.temp_processing_path = temp_processing_path
        os.makedirs(self.temp_processing_path, exist_ok=True)

        self.embeddings_model = OpenAIEmbeddings(api_key=self.openai_api_key)
        
        # Configure AsyncOpenAI client with custom timeouts
        # Default httpx timeouts are often too short (5s for read/write/connect)
        # OpenAI library itself defaults to 600s total, but being explicit for stream reads is good.
        timeout_config = httpx.Timeout(
            connect=15.0,  # Connection timeout
            read=180.0,    # Read timeout (important for waiting for stream chunks)
            write=15.0,    # Write timeout
            pool=15.0      # Pool timeout
        )
        self.async_openai_client = AsyncOpenAI(
            api_key=self.openai_api_key,
            timeout=timeout_config,
            max_retries=1 # Default is 2, reducing to 1 for faster failure if unrecoverable
        )

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("Qdrant URL must be provided either as a parameter or via QDRANT_URL environment variable.")

        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=20.0)
        print(f"Qdrant client initialized for URL: {self.qdrant_url}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.html_transformer = Html2TextTransformer()

        self.kb_collection_name = f"kb_{self.gpt_id}".replace("-", "_").lower()
        self.kb_retriever: Optional[BaseRetriever] = self._get_qdrant_retriever_sync(self.kb_collection_name)

        self.user_collection_retrievers: Dict[str, BaseRetriever] = {}
        self.user_memories: Dict[str, ChatMessageHistory] = {}

        self.tavily_client = None
        if self.tavily_api_key:
            try:
                if TAVILY_AVAILABLE:
                    self.tavily_client = AsyncTavilyClient(api_key=self.tavily_api_key)
                    print(f"✅ Tavily client initialized successfully with API key")
                else:
                    print(f"❌ Tavily package not available. Install it with: pip install tavily-python")
            except Exception as e:
                print(f"❌ Error initializing Tavily client: {e}")
        else:
            print(f"❌ No Tavily API key provided. Web search will be disabled.")
        
        # Initialize clients for other providers
        self.anthropic_client = None
        self.gemini_client = None
        self.llama_model = None
        
        # Setup Claude client if available
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        if CLAUDE_AVAILABLE and self.claude_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=self.claude_api_key)
            print(f"✅ Claude client initialized successfully")
        
        # Setup Gemini client if available
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if GEMINI_AVAILABLE and self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai
            print(f"✅ Gemini client initialized successfully")
        
        # Setup Llama if available (local model)
        if LLAMA_AVAILABLE:
            # This would need a model path - could be configurable
            llama_model_path = os.getenv("LLAMA_MODEL_PATH")
            if llama_model_path and os.path.exists(llama_model_path):
                self.llama_model = Llama(model_path=llama_model_path)
                print(f"✅ Llama model loaded successfully")
        
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = None
        if GROQ_AVAILABLE and self.groq_api_key:
            self.groq_client = AsyncGroq(api_key=self.groq_api_key)
            print(f"✅ Groq client initialized successfully")
    
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
                    try:
                        loader = UnstructuredURLLoader(urls=[r2_key_or_url], mode="elements", strategy="fast", continue_on_failure=True, show_progress=False)
                        loaded_docs = await asyncio.to_thread(loader.load)
                        if loaded_docs and loaded_docs[0].page_content.startswith("Error fetching URL"): return []
                    except Exception as e_url: print(f"Error UnstructuredURLLoader {r2_key_or_url}: {e_url}"); return []
            else:
                r2_object_key_to_download = r2_key_or_url
            
            if not loaded_docs and r2_object_key_to_download:
                download_success = await asyncio.to_thread(
                    self.r2_storage.download_file, r2_object_key_to_download, temp_file_path
                )
                if not download_success: print(f"Failed R2 download: {r2_object_key_to_download}"); return []

                _, ext = os.path.splitext(temp_file_path); ext = ext.lower()
                loader: Any = None
                if ext == ".pdf": loader = PyPDFLoader(temp_file_path)
                elif ext == ".docx": loader = Docx2txtLoader(temp_file_path)
                elif ext in [".html", ".htm"]: loader = BSHTMLLoader(temp_file_path, open_encoding='utf-8')
                else: loader = TextLoader(temp_file_path, autodetect_encoding=True)
                
                loaded_docs = await asyncio.to_thread(loader.load)
                if ext in [".html", ".htm"] and loaded_docs:
                    loaded_docs = self.html_transformer.transform_documents(loaded_docs)
            
            if loaded_docs:
                for doc in loaded_docs:
                    doc.metadata["source"] = r2_key_or_url 
                return self.text_splitter.split_documents(loaded_docs)
            return []
        except Exception as e:
            print(f"Error processing source '{r2_key_or_url}': {e}")
            return []
        finally:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as e_del: print(f"Error deleting temp file {temp_file_path}: {e_del}")
    
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
        is_hybrid_search_active: bool = False,
        is_user_doc: bool = False
    ) -> List[Document]:
        # Enhanced user document search - increase candidate pool for user docs
        candidate_k = k_val * 3 if is_user_doc else (k_val * 2 if is_hybrid_search_active and BM25_AVAILABLE else k_val)
        
        # Expanded candidate retrieval
        if hasattr(retriever, 'search_kwargs'):
            original_k = retriever.search_kwargs.get('k', k_val)
            retriever.search_kwargs['k'] = candidate_k
        
        # Vector retrieval
        docs = await retriever.ainvoke(query) if hasattr(retriever, 'ainvoke') else await asyncio.to_thread(retriever.invoke, query)
        
        # Stage 2: Apply BM25 re-ranking if hybrid search is active
        if is_hybrid_search_active and BM25_AVAILABLE and docs:
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
                search_depth="advanced",  # Changed from "basic" to "advanced" for more comprehensive search
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
        current_system_prompt = system_prompt_override or self.default_system_prompt
        
        # Remove all base_model_type detection and context limit logic
        # Deleted: base_model_type detection, token calculation, etc.
        
        # Remove web search detection for context limits
        # Deleted: has_web_results detection
        
        # Continue with simplified code...
        context_str = self._format_docs_for_llm_context(all_context_docs, "Retrieved Context")
        if not context_str.strip():
            context_str = "No relevant context could be found from any available source for this query. Please ensure documents are uploaded and relevant to your question."

        # Enhanced user message with stronger formatting guidance
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
        
        # Determine which provider to use based on model name prefix
        if current_llm_model.startswith("gpt-"):
            # Use OpenAI (existing implementation)
            if stream:
                async def stream_generator():
                    full_response_content = ""
                    print("Stream generator started")
                    try:
                        print(f"Starting OpenAI stream with model: {current_llm_model}")
                        # Always use full context, no model-specific handling
                        response_stream = await self.async_openai_client.chat.completions.create(
                            model=current_llm_model, 
                            messages=messages, 
                            temperature=self.default_temperature,
                            stream=True
                        )
                        
                        async for chunk in response_stream:
                            content_piece = chunk.choices[0].delta.content
                            if content_piece:
                                print(f"Stream chunk received: {content_piece[:20]}...")
                                full_response_content += content_piece
                                yield content_piece
                        
                        print(f"Stream complete, total response length: {len(full_response_content)}")
                    except Exception as e_stream:
                        # Pass errors straight through
                        print(f"Error during streaming: {e_stream}")
                        # Don't show error to user
                        yield f"I apologize, but I couldn't process your request successfully. Please try asking in a different way."
                    finally:
                        print(f"Saving response to memory, length: {len(full_response_content)}")
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
        
        elif current_llm_model.startswith("claude") and CLAUDE_AVAILABLE and self.anthropic_client:
            # Use Claude implementation
            if stream:
                async def claude_stream_generator():
                    full_response_content = ""
                    try:
                        # Format messages for Claude - system message needs special handling
                        system_content = current_system_prompt
                        claude_messages = []
                        
                        # Extract regular messages (not system)
                        for msg in chat_history_messages:
                            if msg["role"] != "system":
                                claude_messages.append(msg)
                        
                        # Add user query
                        claude_messages.append({"role": "user", "content": user_query_message_content})
                        
                        # Claude streaming call with system as a separate parameter
                        response_stream = await self.anthropic_client.messages.create(
                            model="claude-3-opus-20240229" if "claude" == current_llm_model else current_llm_model,
                            system=system_content,  # System as a separate parameter
                            messages=claude_messages,  # Without system message in the array
                            stream=True
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
                    # Format messages for Claude - system message needs special handling
                    system_content = current_system_prompt
                    claude_messages = []
                    
                    # Extract regular messages (not system)
                    for msg in chat_history_messages:
                        if msg["role"] != "system":
                            claude_messages.append(msg)
                    
                    # Add user query
                    claude_messages.append({"role": "user", "content": user_query_message_content})
                    
                    # Claude API call with system as a separate parameter
                    response = await self.anthropic_client.messages.create(
                        model="claude-3-opus-20240229" if "claude" == current_llm_model else current_llm_model,
                        system=system_content,  # System as a separate parameter
                        messages=claude_messages  # Without system message in the array
                    )
                    response_content = response.content[0].text
                except Exception as e_nostream:
                    print(f"Claude non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        elif current_llm_model.startswith("gemini") and GEMINI_AVAILABLE and self.gemini_client:
            # Implement Gemini client call
            if stream:
                async def gemini_stream_generator():
                    full_response_content = ""
                    try:
                        # Convert messages to Gemini format
                        gemini_messages = []
                        for msg in messages:
                            if msg["role"] == "system":
                                # Prepend system message to first user message
                                continue
                            elif msg["role"] == "user":
                                gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
                            elif msg["role"] == "assistant":
                                gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
                        
                        # Add system message to first user message if needed
                        if messages[0]["role"] == "system" and len(gemini_messages) > 0:
                            for i, msg in enumerate(gemini_messages):
                                if msg["role"] == "user":
                                    gemini_messages[i]["parts"][0]["text"] = f"{messages[0]['content']}\n\n{gemini_messages[i]['parts'][0]['text']}"
                                    break
                        
                        # Always use gemini-1.5-flash regardless of the specific gemini model requested
                        # This model has higher quotas and better rate limits
                        model = self.gemini_client.GenerativeModel(model_name="gemini-1.5-flash")
                        print(f"Using gemini-1.5-flash with higher quotas")
                        
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
                        # If we still hit rate limits, fall back to OpenAI
                        if "429" in str(e_stream) or "quota" in str(e_stream).lower():
                            print("Gemini encountered an issue, switching to OpenAI")
                            try:
                                yield "\nSwitching to alternative AI model...\n\n"
                                fallback_stream = await self.async_openai_client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=messages, 
                                    temperature=self.default_temperature,
                                    stream=True
                                )
                                
                                async for chunk in fallback_stream:
                                    content_piece = chunk.choices[0].delta.content
                                    if content_piece:
                                        full_response_content += content_piece
                                        yield content_piece
                            except Exception as fallback_error:
                                yield "I apologize, but I couldn't process your request. Please try again later."
                        else:
                            yield f"\n[Error: {str(e_stream)}]\n"
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return gemini_stream_generator()
        
        elif current_llm_model.startswith("llama") and LLAMA_AVAILABLE and self.llama_model:
            # Implement Llama model call (local)
            if stream:
                async def llama_stream_generator():
                    full_response_content = ""
                    try:
                        # Format prompt for Llama
                        prompt = f"<s>[INST] {current_system_prompt}\n\n"
                        for msg in chat_history_messages:
                            role = msg["role"]
                            content = msg["content"]
                            if role == "user":
                                prompt += f"{content} [/INST]\n"
                            else:
                                prompt += f"{content} </s><s>[INST] "
                        
                        prompt += f"{user_query_message_content} [/INST]\n"
                        
                        # Call Llama in a thread to not block async
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, 
                            lambda: self.llama_model.create_completion(
                                prompt=prompt,
                                temperature=self.default_temperature,
                                stream=True
                            )
                        )
                        
                        for chunk in result:
                            if "text" in chunk:
                                content_piece = chunk["text"]
                                full_response_content += content_piece
                                yield content_piece
                        
                    except Exception as e_stream:
                        print(f"Llama streaming error: {e_stream}")
                        yield f"\n[Error: {str(e_stream)}]\n"
                    finally:
                        await asyncio.to_thread(user_memory.add_user_message, query)
                        await asyncio.to_thread(user_memory.add_ai_message, full_response_content)
                return llama_stream_generator()
            else:
                response_content = ""
                try:
                    # Format prompt for Llama (similar to above)
                    prompt = f"<s>[INST] {current_system_prompt}\n\n"
                    for msg in chat_history_messages:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "user":
                            prompt += f"{content} [/INST]\n"
                        else:
                            prompt += f"{content} </s><s>[INST] "
                    
                    prompt += f"{user_query_message_content} [/INST]\n"
                    
                    # Call Llama in a thread to not block async
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: self.llama_model.create_completion(
                            prompt=prompt,
                            temperature=self.default_temperature,
                            stream=False
                        )
                    )
                    
                    response_content = result["text"] if "text" in result else ""
                except Exception as e_nostream:
                    print(f"Llama non-streaming error: {e_nostream}")
                    response_content = f"Error: {str(e_nostream)}"
                
                await asyncio.to_thread(user_memory.add_user_message, query)
                await asyncio.to_thread(user_memory.add_ai_message, response_content)
                return response_content
        
        elif current_llm_model.startswith("llama") and GROQ_AVAILABLE and self.groq_client:
            # Map "llama" to Groq's Llama model
            groq_model = "llama3-8b-8192"  # Using a model that exists in Groq
            
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
                        yield f"\n[Error: {str(e_stream)}]\n"
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
        
        else:
            # Fallback to OpenAI if model not supported or client not available
            print(f"Model {current_llm_model} not supported or client not available. Falling back to OpenAI gpt-4o.")
            current_llm_model = "gpt-4o"
            # Continue with OpenAI implementation

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
        enable_web_search: Optional[bool] = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"\n{'='*80}\nStarting streaming query for session: {session_id}")
        start_time = time.time()
        
        # Print search configuration to terminal with debug info
        print(f"\n📊 SEARCH CONFIGURATION:")
        print(f"📌 Debug - Raw enable_web_search param value: {enable_web_search} (type: {type(enable_web_search)})")
        
        # Determine effective hybrid search setting
        actual_use_hybrid_search = use_hybrid_search if use_hybrid_search is not None else self.default_use_hybrid_search
        if actual_use_hybrid_search:
            print(f"🔄 Hybrid search: ACTIVE (BM25 Available: {BM25_AVAILABLE})")
        else:
            print(f"🔄 Hybrid search: INACTIVE")
        
        # Web search status with extra debug info
        if enable_web_search:
            if self.tavily_client:
                print(f"🌐 Web search: ENABLED with Tavily API")
                print(f"🌐 Tavily API key present: {bool(self.tavily_api_key)}")
            else:
                print(f"🌐 Web search: REQUESTED but Tavily API not available")
                print(f"🌐 Tavily API key present: {bool(self.tavily_api_key)}")
                print(f"🌐 TAVILY_AVAILABLE global: {TAVILY_AVAILABLE}")
        else:
            print(f"🌐 Web search: DISABLED (param value: {enable_web_search})")
        
        # Model information
        current_model = llm_model_name or self.default_llm_model_name
        print(f"🧠 Using model: {current_model}")
        print(f"{'='*80}")
        
        formatted_chat_history = await self._get_formatted_chat_history(session_id)
        retrieval_query = query
        
        print(f"\n📝 Processing query: '{retrieval_query}'")
        
        all_retrieved_docs: List[Document] = []
        
        # First get user document context with deeper search (higher k-value)
        user_session_retriever = await self._get_user_retriever(session_id)
        user_session_docs = await self._get_retrieved_documents(
            user_session_retriever, 
            retrieval_query, 
            k_val=3,  # Change from 5 to 3
            is_hybrid_search_active=actual_use_hybrid_search,
            is_user_doc=True  # Flag as user doc for deeper search
        )
        if user_session_docs: 
            print(f"📄 Retrieved {len(user_session_docs)} user-specific documents")
            all_retrieved_docs.extend(user_session_docs)
        else:
            print(f"📄 No user-specific documents found")
        
        # Then add knowledge base context
        kb_docs = await self._get_retrieved_documents(
            self.kb_retriever, 
            retrieval_query, 
            k_val=5, 
            is_hybrid_search_active=actual_use_hybrid_search
        )
        if kb_docs: 
            print(f"📚 Retrieved {len(kb_docs)} knowledge base documents")
            all_retrieved_docs.extend(kb_docs)
        else:
            print(f"📚 No knowledge base documents found")
        
        # Add web search results if enabled - MOVED EARLIER in the process
        if enable_web_search and self.tavily_client:
            web_docs = await self._get_web_search_docs(retrieval_query, True, num_results=4)
            if web_docs:
                print(f"🌐 Retrieved {len(web_docs)} web search documents")
                all_retrieved_docs.extend(web_docs)
        
        # Process any adhoc document keys
        if user_r2_document_keys:
            print(f"📎 Processing {len(user_r2_document_keys)} ad-hoc document keys")
            adhoc_load_tasks = [self._download_and_split_one_doc(r2_key) for r2_key in user_r2_document_keys]
            results_list_of_splits = await asyncio.gather(*adhoc_load_tasks)
            adhoc_docs_count = 0
            for splits_from_one_doc in results_list_of_splits:
                adhoc_docs_count += len(splits_from_one_doc)
                all_retrieved_docs.extend(splits_from_one_doc) 
            print(f"📎 Added {adhoc_docs_count} splits from ad-hoc documents")
        
        # Deduplicate the documents
        unique_docs_content = set()
        deduplicated_docs = []
        for doc in all_retrieved_docs:
            if doc.page_content not in unique_docs_content:
                deduplicated_docs.append(doc)
                unique_docs_content.add(doc.page_content)
        all_retrieved_docs = deduplicated_docs
        
        print(f"\n🔍 Retrieved {len(all_retrieved_docs)} total unique documents")
        
        # Count documents by source type
        source_counts = {}
        for doc in all_retrieved_docs:
            source_type = doc.metadata.get("source_type", "unknown")
            if "web" in source_type:
                source_type = "web_search"
            elif "user" in str(doc.metadata.get("source", "")):
                source_type = "user_document"
            else:
                source_type = "knowledge_base"
            
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        for src_type, count in source_counts.items():
            if src_type == "web_search":
                print(f"🌐 Web search documents: {count}")
            elif src_type == "user_document":
                print(f"📄 User documents: {count}")
            elif src_type == "knowledge_base":
                print(f"📚 Knowledge base documents: {count}")
            else:
                print(f"📃 {src_type} documents: {count}")
        
        print("\n🧠 Starting LLM stream generation...")
        llm_stream_generator = await self._generate_llm_response(
            session_id, query, all_retrieved_docs, formatted_chat_history,
            llm_model_name, system_prompt_override, stream=True
        )
        
        print("🔄 LLM stream initialized, beginning content streaming")
        async for content_chunk in llm_stream_generator:
            yield {"type": "content", "data": content_chunk}
        
        print("✅ Stream complete, sending done signal")
        total_time = int((time.time() - start_time) * 1000)
        print(f"⏱️ Total processing time: {total_time}ms")
        yield {"type": "done", "data": {"total_time_ms": total_time}}
        print(f"{'='*80}\n")

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
            print(f"Hybrid search is ACTIVE for this query (session: {session_id}). BM25 Available: {BM25_AVAILABLE}")
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
    print(f"rag.py loaded. Qdrant URL: {os.getenv('QDRANT_URL')}. Tavily available: {TAVILY_AVAILABLE}. BM25 available: {BM25_AVAILABLE}")