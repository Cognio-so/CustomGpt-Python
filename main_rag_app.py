from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import shutil
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
from io import BytesIO

from storage import CloudflareR2Storage
from rag import EnhancedRAG
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

app = FastAPI(title="Enhanced RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

r2_storage = CloudflareR2Storage()

active_rag_sessions: Dict[str, EnhancedRAG] = {}
sessions_lock = asyncio.Lock()

LOCAL_DATA_BASE_PATH = os.getenv("LOCAL_DATA_PATH", "local_rag_data")
LOCAL_KB_INDEX_PATH_TEMPLATE = os.path.join(LOCAL_DATA_BASE_PATH, "kb_indexes", "{gpt_id}")
LOCAL_USER_INDEX_BASE_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "user_indexes")
TEMP_DOWNLOAD_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "temp_downloads")

os.makedirs(os.path.join(LOCAL_DATA_BASE_PATH, "kb_indexes"), exist_ok=True)
os.makedirs(LOCAL_USER_INDEX_BASE_PATH, exist_ok=True)
os.makedirs(TEMP_DOWNLOAD_PATH, exist_ok=True)

# --- Pydantic Models ---
class BaseRAGRequest(BaseModel):
    user_email: str
    gpt_id: str
    gpt_name: Optional[str] = "default_gpt"

class ChatPayload(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    user_document_keys: Optional[List[str]] = Field([], alias="user_documents")
    use_hybrid_search: Optional[bool] = False
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    web_search_enabled: Optional[bool] = False

class ChatStreamRequest(BaseRAGRequest, ChatPayload):
    memory: Optional[List[Dict[str, str]]] = []

class ChatRequest(BaseRAGRequest, ChatPayload):
    pass

class GptContextSetupRequest(BaseRAGRequest):
    kb_document_urls: Optional[List[str]] = []
    default_model: Optional[str] = None
    default_system_prompt: Optional[str] = None
    default_use_hybrid_search: Optional[bool] = False

class FileUploadInfoResponse(BaseModel):
    filename: str
    stored_url_or_key: str
    status: str
    error_message: Optional[str] = None

class GptOpenedRequest(BaseModel):
    user_email: str
    gpt_id: str
    gpt_name: str
    file_urls: List[str] = []
    use_hybrid_search: bool = False
    config_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")  # Renamed to avoid shadowing

# --- Helper Functions ---
def get_session_id(user_email: str, gpt_id: str) -> str:
    email_part = user_email.replace('@', '_').replace('.', '_')
    return f"user_{email_part}_gpt_{gpt_id}"

async def get_or_create_rag_instance(
    user_email: str,
    gpt_id: str,
    gpt_name: Optional[str] = "default_gpt",
    default_model: Optional[str] = None,
    default_system_prompt: Optional[str] = None,
    default_use_hybrid_search: Optional[bool] = False
) -> EnhancedRAG:
    async with sessions_lock:
        if gpt_id not in active_rag_sessions:
            print(f"Creating new EnhancedRAG instance for gpt_id: {gpt_id}")
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment.")
                
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url:
                raise ValueError("QDRANT_URL not set in environment.")

            active_rag_sessions[gpt_id] = EnhancedRAG(
                gpt_id=gpt_id,
                r2_storage_client=r2_storage,
                openai_api_key=openai_api_key,
                default_llm_model_name=default_model or os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o"),
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                temp_processing_path=TEMP_DOWNLOAD_PATH,
                default_system_prompt=default_system_prompt,
                default_use_hybrid_search=default_use_hybrid_search
            )
        else:
            rag_instance = active_rag_sessions[gpt_id]
            if default_model:
                rag_instance.default_llm_model_name = default_model
            if default_system_prompt:
                rag_instance.default_system_prompt = default_system_prompt
            if default_use_hybrid_search is not None:
                rag_instance.default_use_hybrid_search = default_use_hybrid_search
            print(f"Reusing EnhancedRAG instance for gpt_id: {gpt_id}. Updated defaults if provided.")

        return active_rag_sessions[gpt_id]

async def _process_uploaded_file_to_r2(
    file: UploadFile,
    is_user_doc: bool
) -> FileUploadInfoResponse:
    try:
        file_content = await file.read()
        file_bytes_io = BytesIO(file_content)
        
        success, r2_path_or_error = await asyncio.to_thread(
            r2_storage.upload_file,
            file_data=file_bytes_io,
            filename=file.filename,
            is_user_doc=is_user_doc
        )

        if success:
            print(f"File '{file.filename}' (is_user_doc={is_user_doc}) stored at: {r2_path_or_error}")
            return FileUploadInfoResponse(
                filename=file.filename,
                stored_url_or_key=r2_path_or_error,
                status="success"
            )
        else:
            print(f"Failed to store file '{file.filename}'. Error: {r2_path_or_error}")
            return FileUploadInfoResponse(
                filename=file.filename,
                stored_url_or_key="", status="failure", error_message=r2_path_or_error
            )
    except Exception as e:
        print(f"Exception processing file '{file.filename}': {e}")
        return FileUploadInfoResponse(
            filename=file.filename,
            stored_url_or_key="", status="failure", error_message=str(e)
        )

# --- API Endpoints ---

@app.post("/setup-gpt-context", summary="Initialize/update a GPT's knowledge base from URLs")
async def setup_gpt_context_endpoint(request: GptContextSetupRequest, background_tasks: BackgroundTasks):
    rag_instance = await get_or_create_rag_instance(
        user_email=request.user_email,
        gpt_id=request.gpt_id,
        gpt_name=request.gpt_name,
        default_model=request.default_model,
        default_system_prompt=request.default_system_prompt,
        default_use_hybrid_search=request.default_use_hybrid_search
    )

    if request.kb_document_urls:
        async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
            print(f"BG Task: Processing {len(urls)} KB URLs for gpt_id '{rag.gpt_id}'...")
            r2_kb_keys_or_urls_for_indexing = []
            for url in urls:
                if not (url.startswith('http://') or url.startswith('https://')):
                    print(f"Skipping invalid KB URL: {url}")
                    continue
                success, r2_path = await asyncio.to_thread(
                    r2_storage.download_file_from_url, url=url
                )
                if success:
                    r2_kb_keys_or_urls_for_indexing.append(r2_path)
                    print(f"KB URL '{url}' processed to R2: {r2_path}")
                else:
                    print(f"Failed to process KB URL '{url}'. Error: {r2_path}")
            
            if r2_kb_keys_or_urls_for_indexing:
                try:
                    await rag.update_knowledge_base_from_r2(r2_kb_keys_or_urls_for_indexing)
                except Exception as e:
                    print(f"Error indexing KB documents for gpt_id '{rag.gpt_id}': {e}")

        background_tasks.add_task(_process_kb_urls_task, request.kb_document_urls, rag_instance)
        return JSONResponse(status_code=202, content={
            "message": f"KB processing for gpt_id '{request.gpt_id}' initiated for {len(request.kb_document_urls)} URLs.",
            "gpt_id": request.gpt_id
        })
    else:
        return JSONResponse(status_code=200, content={
            "message": f"No KB URLs provided. RAG instance for gpt_id '{request.gpt_id}' is ready.",
            "gpt_id": request.gpt_id
        })

@app.post("/upload-documents", summary="Upload documents (KB or User-specific)")
async def upload_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_email: str = Form(...),
    gpt_id: str = Form(...),
    is_user_document: str = Form("false"),
):
    is_user_doc_bool = is_user_document.lower() == "true"
    processing_results: List[FileUploadInfoResponse] = []
    r2_keys_or_urls_for_indexing: List[str] = []

    for file_upload in files:
        result = await _process_uploaded_file_to_r2(file_upload, is_user_doc_bool)
        processing_results.append(result)
        if result.status == "success" and result.stored_url_or_key:
            r2_keys_or_urls_for_indexing.append(result.stored_url_or_key)

    if not r2_keys_or_urls_for_indexing:
        return JSONResponse(status_code=400, content={
            "message": "No files were successfully uploaded to R2.",
            "upload_results": [r.model_dump() for r in processing_results]
        })

    rag_instance = await get_or_create_rag_instance(user_email=user_email, gpt_id=gpt_id)
    
    async def _index_documents_task(rag: EnhancedRAG, keys_or_urls: List[str], is_user_specific: bool, u_email: str, g_id: str):
        doc_type = "user-specific" if is_user_specific else "knowledge base"
        s_id = get_session_id(u_email, g_id)
        print(f"BG Task: Indexing {len(keys_or_urls)} {doc_type} documents for gpt_id '{rag.gpt_id}' (session '{s_id}')...")
        try:
            if is_user_specific:
                await rag.update_user_documents_from_r2(session_id=s_id, r2_keys_or_urls=keys_or_urls)
            else:
                await rag.update_knowledge_base_from_r2(keys_or_urls)
            print(f"BG Task: Indexing complete for {doc_type} documents.")
        except Exception as e:
            print(f"BG Task: Error indexing {doc_type} documents for gpt_id '{rag.gpt_id}': {e}")

    background_tasks.add_task(_index_documents_task, rag_instance, r2_keys_or_urls_for_indexing, is_user_doc_bool, user_email, gpt_id)

    return JSONResponse(status_code=202, content={
        "message": f"{len(r2_keys_or_urls_for_indexing)} files accepted for {'user-specific' if is_user_doc_bool else 'knowledge base'} indexing. Processing in background.",
        "upload_results": [r.model_dump() for r in processing_results]
    })

@app.post("/chat-stream")
async def chat_stream(request: ChatStreamRequest):
    try:
        # Initialize rag_instance
        rag_instance = await get_or_create_rag_instance(
            user_email=request.user_email,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.model,
            default_system_prompt=request.system_prompt,
            default_use_hybrid_search=request.use_hybrid_search
        )
        
        session_id = get_session_id(request.user_email, request.gpt_id)

        print(f"\n{'='*40}")
        print(f"📝 New chat request from user: {request.user_email}")
        print(f"🔍 GPT ID: {request.gpt_id}")
        print(f"💬 Query: '{request.message}'")
        if request.web_search_enabled:
            print(f"🌐 Web search requested: ENABLED")
        else:
            print(f"🌐 Web search requested: DISABLED")
        print(f"{'='*40}\n")

        # Setup SSE headers
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }

        # Create streaming response generator
        async def generate():
            try:
                async for chunk in rag_instance.query_stream(
                    session_id=session_id,
                    query=request.message,
                    chat_history=request.history,
                    user_r2_document_keys=request.user_document_keys,
                    use_hybrid_search=request.use_hybrid_search,
                    llm_model_name=request.model,
                    system_prompt_override=request.system_prompt,
                    enable_web_search=request.web_search_enabled
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                print(f"❌ Error during streaming in /chat-stream: {e}")
                error_chunk = {
                    "type": "error",
                    "data": {"error": str(e)}
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(generate(), headers=headers)
    
    except Exception as e:
        print(f"❌ Error in /chat-stream endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/chat", summary="Handle non-streaming chat requests")
async def chat_endpoint(request: ChatRequest):
    rag_instance = await get_or_create_rag_instance(
        user_email=request.user_email, gpt_id=request.gpt_id, gpt_name=request.gpt_name,
        default_model=request.model,
        default_system_prompt=request.system_prompt,
        default_use_hybrid_search=request.use_hybrid_search
    )
    session_id = get_session_id(request.user_email, request.gpt_id)
    try:
        response_data = await rag_instance.query(
            session_id=session_id,
            query=request.message,
            chat_history=request.history,
            user_r2_document_keys=request.user_document_keys,
            use_hybrid_search=request.use_hybrid_search,
            llm_model_name=request.model,
            system_prompt_override=request.system_prompt,
            enable_web_search=request.web_search_enabled
        )
        return JSONResponse(content={"success": True, "data": response_data})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/gpt-opened", summary="Notify backend when a GPT is opened")
async def gpt_opened_endpoint(request: GptOpenedRequest, background_tasks: BackgroundTasks):
    try:
        rag_instance = await get_or_create_rag_instance(
            user_email=request.user_email,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.config_schema.get("model") if request.config_schema else None,
            default_system_prompt=request.config_schema.get("instructions") if request.config_schema else None,
            default_use_hybrid_search=request.config_schema.get("capabilities", {}).get("hybridSearch", False) if request.config_schema else request.use_hybrid_search
        )
        
        sanitized_email = request.user_email.replace('@', '_').replace('.', '_')
        sanitized_gpt_name = (request.gpt_name or 'gpt').replace(' ', '_').replace('-', '_')
        collection_name = f"kb_{sanitized_email}_{sanitized_gpt_name}_{request.gpt_id}"
        
        if request.file_urls:
            async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
                r2_kb_keys_or_urls_for_indexing = []
                for url in urls:
                    if url.startswith('http://') or url.startswith('https://'):
                        success, r2_path = await asyncio.to_thread(
                            r2_storage.download_file_from_url, url=url
                        )
                        if success:
                            r2_kb_keys_or_urls_for_indexing.append(r2_path)
                
                if r2_kb_keys_or_urls_for_indexing:
                    try:
                        await rag.update_knowledge_base_from_r2(r2_kb_keys_or_urls_for_indexing)
                    except Exception as e:
                        print(f"Error indexing KB documents for gpt_id '{rag.gpt_id}': {e}")
            
            background_tasks.add_task(_process_kb_urls_task, request.file_urls, rag_instance)
        
        return {"success": True, "collection_name": collection_name}
    except Exception as e:
        print(f"Error in gpt-opened endpoint: {e}")
        return {"success": False, "error": str(e)}

@app.post("/upload-chat-files", summary="Upload files for chat")
async def upload_chat_files_endpoint(
    files: List[UploadFile] = File(...),
    user_email: str = Form(...),
    gpt_id: str = Form(...),
    gpt_name: str = Form(...),
    collection_name: str = Form(...),
    is_user_document: str = Form("true"),
    use_hybrid_search: str = Form("false"),
    optimize_pdfs: str = Form("false"),
):
    is_user_doc_bool = is_user_document.lower() == "true"
    use_hybrid_search_bool = use_hybrid_search.lower() == "true"
    optimize_pdfs_bool = optimize_pdfs.lower() == "true"
    
    processing_results = []
    file_urls = []

    for file_upload in files:
        result = await _process_uploaded_file_to_r2(file_upload, is_user_doc_bool)
        if result.status == "success" and result.stored_url_or_key:
            file_urls.append(result.stored_url_or_key)
        processing_results.append(result)

    rag_instance = await get_or_create_rag_instance(
        user_email=user_email, 
        gpt_id=gpt_id,
        gpt_name=gpt_name
    )
    
    if file_urls:
        session_id = get_session_id(user_email, gpt_id)
        
        try:
            if is_user_doc_bool:
                await rag_instance.update_user_documents_from_r2(session_id=session_id, r2_keys_or_urls=file_urls)
            else:
                await rag_instance.update_knowledge_base_from_r2(file_urls)
            print(f"Indexing complete for {len(file_urls)} {'user-specific' if is_user_doc_bool else 'knowledge base'} documents for session '{session_id}'.")
        except Exception as e:
            print(f"Error indexing chat files for session '{session_id}': {e}")
            return {
                "success": False,
                "message": f"Failed to index {len(file_urls)} files: {str(e)}",
                "file_urls": file_urls,
                "processing": False
            }
    
    return {
        "success": True,
        "message": f"Processed and indexed {len(file_urls)} files",
        "file_urls": file_urls,
        "processing": len(file_urls) > 0
    }

@app.get("/gpt-collection-info/{param1}/{param2}", summary="Get information about a GPT collection")
async def gpt_collection_info(param1: str, param2: str):
    return {
        "status": "available",
        "timestamp": time.time()
    }

@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Health check endpoint", tags=["Monitoring"])
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/dev/reset-gpt-context", summary="DEVELOPMENT ONLY: Clear RAG context for a GPT", tags=["Development"])
async def dev_reset_gpt_context_endpoint(gpt_id: str = Form(...)):
    if os.getenv("ENVIRONMENT_TYPE", "production").lower() != "development":
        return JSONResponse(status_code=403, content={"error": "Endpoint only available in development."})

    async with sessions_lock:
        if gpt_id in active_rag_sessions:
            try:
                rag_instance_to_reset = active_rag_sessions.pop(gpt_id)
                await rag_instance_to_reset.clear_all_context()
                
                kb_index_path_to_delete = LOCAL_KB_INDEX_PATH_TEMPLATE.format(gpt_id=gpt_id)
                if os.path.exists(kb_index_path_to_delete):
                    shutil.rmtree(kb_index_path_to_delete)
                
                print(f"DEV: Cleared in-memory RAG context and local KB index for gpt_id '{gpt_id}'. R2 files not deleted.")
                return {"status": "success", "message": f"RAG context for gpt_id '{gpt_id}' cleared from memory and local disk."}
            except Exception as e:
                print(f"DEV: Error clearing context for gpt_id '{gpt_id}': {e}")
                return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        else:
            return JSONResponse(status_code=404, content={"status": "not_found", "message": f"No active RAG context for gpt_id '{gpt_id}'."})

if __name__ == "__main__":
    print("Starting RAG API server...")
    print(f"Local data base path: {os.path.abspath(LOCAL_DATA_BASE_PATH)}")
    print(f"OpenAI API Key Loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No - Set OPENAI_API_KEY'}")
    print(f"CORS Origins: {os.getenv('CORS_ALLOWED_ORIGINS', '[\"http://localhost:5173\"]')}")
    
    uvicorn.run(
        "main_rag_app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT_TYPE", "").lower() == "development",
        timeout_keep_alive=60
    )

# Note: To fix the LangChainDeprecationWarning in rag.py, update the import as follows:
# from langchain_community.chat_message_histories import ChatMessageHistory
# This should be applied in the rag.py file to avoid the deprecation warning.
