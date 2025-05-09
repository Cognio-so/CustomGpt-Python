import boto3
import requests
import os
from typing import List, Optional, Tuple, Union
from botocore.exceptions import ClientError
import hashlib
from urllib.parse import urlparse
import shutil
from io import BytesIO

class CloudflareR2Storage:
    def __init__(self):
        self.use_local_fallback = False # True if R2 cannot be initialized/used
        self.r2 = None # Initialize r2 client attribute
        
        try:
            # Load Cloudflare R2 credentials from environment
            self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            self.access_key = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
            self.secret_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
            self.bucket_name = os.getenv("CLOUDFLARE_BUCKET_NAME", "rag-documents")
            
            if not all([self.account_id, self.access_key, self.secret_key]):
                print("Missing Cloudflare R2 credentials. R2 services will not be available. Local fallback for KB docs only.")
                self.use_local_fallback = True
            else:
                # Initialize R2 client
                self.r2 = boto3.resource(
                    's3',
                    endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
                self._ensure_bucket_exists() # This might raise an exception if R2 is not reachable
                
        except Exception as e:
            print(f"Error initializing R2 storage or ensuring bucket: {e}")
            print("R2 services will not be available. Falling back to local storage for KB documents only.")
            self.use_local_fallback = True
            self.r2 = None # Ensure r2 client is None if setup failed
            
        # Create local storage directory for knowledge base (kb) files.
        # User documents will strictly try R2 or fail and are not stored locally.
        os.makedirs("local_storage/kb", exist_ok=True)

    def _ensure_bucket_exists(self) -> None:
        """Ensure the R2 bucket exists. Assumes self.r2 is initialized."""
        if not self.r2: # Should not happen if called correctly, but defensive check
            raise ConnectionError("R2 client not initialized.")
        try:
            self.r2.meta.client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404': # Not Found
                 # Create the bucket if it doesn't exist
                print(f"Bucket '{self.bucket_name}' not found. Creating bucket.")
                self.r2.create_bucket(Bucket=self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                # Re-raise other client errors
                raise

    def _upload_local_kb(self, file_data, filename: str) -> Tuple[bool, str]:
        """Upload a knowledge base file to local storage"""
        try:
            folder = "kb" 
            local_path = f"local_storage/{folder}/{filename}"
            
            # Ensure the specific kb subfolder exists (though top-level kb is made in __init__)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if isinstance(file_data, bytes):
                with open(local_path, 'wb') as f:
                    f.write(file_data)
            else: # Assume file-like object
                file_data.seek(0) 
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(file_data, f)
            
            file_url = f"file://{os.path.abspath(local_path)}"
            print(f"KB file '{filename}' saved locally to '{local_path}'.")
            return True, file_url
        except Exception as e:
            print(f"Error saving KB file '{filename}' locally: {e}")
            return False, str(e)

    def upload_file(self, file_data, filename: str, is_user_doc: bool = False, 
                    schedule_deletion_hours: int = 72) -> Tuple[bool, str]:
        """
        Upload a file. User documents (is_user_doc=True) go only to R2.
        Knowledge base files (is_user_doc=False) go to R2 with local fallback.
        Now supports image files as well.
        Automatically schedules deletion after specified hours (default: 72).
        """
        # Get file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Check if it's an image file
        is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        if is_user_doc:
            folder = "user_docs"
            key = f"{folder}/{filename}"

            if self.use_local_fallback or not self.r2:
                print(f"Cannot upload user document '{filename}'. R2 is not available or not initialized. User documents must be uploaded to R2.")
                return False, "R2 not available/initialized; user documents require R2."
            
            try:
                print(f"Uploading user document '{filename}' to R2 key '{key}'...")
                file_obj_to_upload = BytesIO(file_data) if isinstance(file_data, bytes) else file_data
                if not isinstance(file_data, bytes): # If it's a file-like object, ensure it's at the start
                    file_obj_to_upload.seek(0)

                self.r2.meta.client.upload_fileobj(file_obj_to_upload, self.bucket_name, key)
                file_url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{key}"
                
                # Schedule deletion after specified hours
                self.schedule_deletion(key, schedule_deletion_hours)
                
                print(f"User document '{filename}' uploaded successfully to R2: {file_url} (will be deleted after {schedule_deletion_hours} hours)")
                return True, file_url
            except Exception as e:
                print(f"Error uploading user document '{filename}' to R2: {e}. Will not fall back to local storage.")
                return False, f"R2 upload failed for user document: {e}"
        else: # This is for knowledge base files (is_user_doc=False)
            folder = "kb"
            key = f"{folder}/{filename}"

            if self.use_local_fallback or not self.r2:
                print(f"R2 not available or not initialized, falling back to local storage for KB file '{filename}'.")
                return self._upload_local_kb(file_data, filename)
            
            try:
                print(f"Uploading KB file '{filename}' to R2 key '{key}'...")
                file_obj_to_upload = BytesIO(file_data) if isinstance(file_data, bytes) else file_data
                if not isinstance(file_data, bytes):
                    file_obj_to_upload.seek(0)

                self.r2.meta.client.upload_fileobj(file_obj_to_upload, self.bucket_name, key)
                file_url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{key}"
                
                # Schedule deletion after specified hours
                self.schedule_deletion(key, schedule_deletion_hours)
                
                print(f"KB file '{filename}' uploaded successfully to R2: {file_url} (will be deleted after {schedule_deletion_hours} hours)")
                return True, file_url
            except Exception as e:
                print(f"Error uploading KB file '{filename}' to R2: {e}. Falling back to local storage.")
                return self._upload_local_kb(file_data, filename)

    def _download_content_from_url(self, url: str) -> Tuple[bool, Union[bytes, str]]:
        """Downloads content from a URL, returns (success, content_bytes_or_error_string)."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            return True, response.content
        except requests.exceptions.RequestException as e:
            print(f"Error downloading content from {url}: {e}")
            return False, str(e)

    def download_file_from_url(self, url: str, target_filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Download a file from a URL and store it as a knowledge base (kb) document.
        Uses R2 with local fallback for these kb files.
        'target_filename' is the name it will have in storage. If None, it's derived from URL.
        Returns (success, stored_file_url_or_error_message).
        """
        final_filename: str
        if target_filename:
            final_filename = os.path.basename(target_filename)
        else:
            parsed_url = urlparse(url)
            basename = os.path.basename(parsed_url.path)
            if basename:
                final_filename = basename
            else:
                url_hash = hashlib.md5(url.encode()).hexdigest()
                # Attempt to get a reasonable extension
                content_type = requests.head(url, timeout=10).headers.get('content-type')
                ext = ".data" # default extension
                if content_type:
                    if 'pdf' in content_type: ext = '.pdf'
                    elif 'text' in content_type: ext = '.txt'
                    elif 'html' in content_type: ext = '.html'
                    elif 'json' in content_type: ext = '.json'
                final_filename = f"{url_hash}{ext}"
        
        print(f"Attempting to download from URL '{url}' to be stored as '{final_filename}'.")
        success, content_or_error = self._download_content_from_url(url)
        
        if not success:
            return False, content_or_error # content_or_error is the error message

        # Upload the downloaded content as a knowledge base file (is_user_doc=False)
        print(f"Content from '{url}' downloaded successfully. Now uploading as KB file '{final_filename}'.")
        return self.upload_file(content_or_error, final_filename, is_user_doc=False)
            
    def download_file(self, key: str, local_download_path: str) -> bool:
        """
        Download a file to local_download_path.
        'key' is the R2 object key (e.g., "user_docs/file.pdf" or "kb/file.txt").
        User documents are fetched only from R2.
        Knowledge base documents are fetched from R2, with a fallback to local storage
        if they were originally saved there due to R2 issues.
        """
        is_user_doc_key = key.startswith("user_docs/")
        is_kb_doc_key = key.startswith("kb/")

        if not is_user_doc_key and not is_kb_doc_key:
            print(f"Invalid key format: '{key}'. Key must start with 'user_docs/' or 'kb/'.")
            return False

        # Case 1: R2 was not available at initialization (self.use_local_fallback is True)
        if self.use_local_fallback or not self.r2:
            if is_user_doc_key:
                print(f"R2 is unavailable/uninitialized. Cannot download user document '{key}' as it must be on R2.")
                return False
            if is_kb_doc_key:
                local_source_path = f"local_storage/{key}"
                print(f"R2 unavailable/uninitialized. Attempting to download KB file '{key}' from local fallback '{local_source_path}'.")
                if os.path.exists(local_source_path):
                    try:
                        shutil.copy2(local_source_path, local_download_path)
                        print(f"KB file '{key}' downloaded from local fallback storage to '{local_download_path}'.")
                        return True
                    except Exception as e:
                        print(f"Error copying local KB file '{local_source_path}' to '{local_download_path}': {e}")
                        return False
                else:
                    print(f"R2 unavailable/uninitialized and KB file '{key}' not found in local fallback storage.")
                    return False
            return False # Should not be reached given key checks

        # Case 2: R2 is presumed active, try R2 first.
        try:
            print(f"Attempting to download '{key}' from R2 to '{local_download_path}'...")
            os.makedirs(os.path.dirname(local_download_path), exist_ok=True) # Ensure target dir exists
            self.r2.meta.client.download_file(
                self.bucket_name,
                key,
                local_download_path
            )
            print(f"File '{key}' downloaded successfully from R2.")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == '404' or error_code == 'NoSuchKey' or "NotFound" in str(e): # R2 can return different 404 like codes
                print(f"File '{key}' not found in R2 (NoSuchKey/404).")
                if is_kb_doc_key:
                    local_source_path = f"local_storage/{key}"
                    print(f"Checking local fallback for KB file at '{local_source_path}'...")
                    if os.path.exists(local_source_path):
                        try:
                            shutil.copy2(local_source_path, local_download_path)
                            print(f"KB file '{key}' downloaded from local fallback storage to '{local_download_path}'.")
                            return True
                        except Exception as e_copy:
                            print(f"Error copying local fallback KB file '{key}': {e_copy}")
                            return False
                    else:
                        print(f"KB file '{key}' also not found in local fallback storage.")
                        return False
                else: # User doc not found in R2 - it's definitively not found.
                    print(f"User document '{key}' not found in R2 and has no local fallback.")
                    return False
            else: # Other R2 client error
                print(f"R2 ClientError when downloading '{key}': {e}")
                return False
        except Exception as e: # Other general errors
            print(f"General error downloading file '{key}': {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with the given prefix.
        If R2 is active, lists from R2. This won't show KB files that *only* exist locally.
        If R2 initialization failed, lists from local_storage (effectively local_storage/kb/ if prefix allows).
        """
        if self.use_local_fallback or not self.r2:
            print(f"R2 unavailable/uninitialized. Listing files from local_storage with prefix '{prefix}'.")
            try:
                # Normalize prefix for local path construction: remove leading / if any, ensure ends with / if not empty
                local_prefix_dir = prefix
                if local_prefix_dir.startswith('/'): 
                    local_prefix_dir = local_prefix_dir[1:]
                if local_prefix_dir and not local_prefix_dir.endswith('/'):
                    local_prefix_dir += '/'
                
                base_local_dir = "local_storage/"
                # Only allow listing within 'kb/' if prefix specifies it or is empty (implies list all, but we only have kb locally)
                # Or if prefix is 'user_docs/', which should be empty locally.
                
                effective_local_dir = os.path.join(base_local_dir, local_prefix_dir)
                
                # Security/Consistency: If global fallback is on, only list from 'kb' if applicable.
                # User docs should not be listed from local.
                if prefix.startswith("user_docs/"):
                    print("User documents are R2-only; cannot list from local storage.")
                    return []

                # Adjust effective_local_dir if prefix doesn't specify 'kb/' but we are in local fallback for kb
                if not prefix.startswith("kb/") and os.path.exists(os.path.join(base_local_dir, "kb")):
                    # If prefix is generic, list all relevant (i.e. kb) files.
                    # This part of logic might need refinement based on exact desired listing behavior for general prefix.
                    # For now, if prefix is "kb/", it works. If empty, it lists from "local_storage/".
                    pass # current effective_local_dir is okay, or could be forced to "local_storage/kb/"

                listed_files = []
                if os.path.exists(effective_local_dir) and os.path.isdir(effective_local_dir):
                    for f_name in os.listdir(effective_local_dir):
                        if os.path.isfile(os.path.join(effective_local_dir, f_name)):
                            # Return keys relative to bucket (e.g., kb/file.txt)
                            listed_files.append(os.path.join(local_prefix_dir, f_name).replace("\\", "/")) 
                return listed_files
            except Exception as e:
                print(f"Error listing local files with prefix '{prefix}': {e}")
                return []
        
        # R2 is active
        try:
            print(f"Listing files from R2 bucket '{self.bucket_name}' with prefix '{prefix}'.")
            response = self.r2.meta.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"Error listing files from R2 with prefix '{prefix}': {e}")
            return []

    def schedule_deletion(self, key: str, hours: int = 72) -> bool:
        """
        Schedule a file for deletion after specified hours (default: 72 hours)
        This works by setting object metadata with expiration time
        """
        if self.use_local_fallback or not self.r2:
            print(f"R2 unavailable/uninitialized. Cannot schedule deletion for '{key}'.")
            return False
        
        try:
            # First, check if object exists
            try:
                self.r2.meta.client.head_object(Bucket=self.bucket_name, Key=key)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    print(f"File '{key}' not found in R2, cannot schedule deletion.")
                    return False
                raise
            
            # Set object lifecycle metadata
            import datetime
            
            # Calculate expiration time
            expiration_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
            expiration_timestamp = int(expiration_time.timestamp())
            
            # Copy object to itself with new metadata (can't update metadata directly)
            self.r2.meta.client.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': key},
                Key=key,
                Metadata={
                    'expiration_time': str(expiration_timestamp),
                    'auto_delete': 'true'
                },
                MetadataDirective='REPLACE'
            )
            
            print(f"File '{key}' scheduled for deletion after {hours} hours (at {expiration_time}).")
            return True
        except Exception as e:
            print(f"Error scheduling deletion for '{key}': {e}")
            return False

    def check_and_delete_expired_files(self) -> int:
        """
        Check all files and delete those that have passed their expiration time
        Returns count of deleted files
        """
        if self.use_local_fallback or not self.r2:
            print("R2 unavailable/uninitialized. Cannot check for expired files.")
            return 0
        
        import datetime
        deleted_count = 0
        current_time = datetime.datetime.now().timestamp()
        
        try:
            # List all objects in the bucket
            paginator = self.r2.meta.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    try:
                        # Get object metadata
                        response = self.r2.meta.client.head_object(
                            Bucket=self.bucket_name,
                            Key=key
                        )
                        
                        metadata = response.get('Metadata', {})
                        if 'expiration_time' in metadata and metadata.get('auto_delete') == 'true':
                            expiration_time = int(metadata['expiration_time'])
                            
                            # Check if file has expired
                            if current_time > expiration_time:
                                # Delete the expired file
                                self.r2.meta.client.delete_object(
                                    Bucket=self.bucket_name,
                                    Key=key
                                )
                                print(f"Deleted expired file '{key}'")
                                deleted_count += 1
                    except Exception as e_obj:
                        print(f"Error checking metadata for '{key}': {e_obj}")
            
            return deleted_count
        except Exception as e:
            print(f"Error checking for expired files: {e}")
            return 0

    def cleanup_expired_files(self):
        """Run periodic cleanup of expired files"""
        if self.use_local_fallback or not self.r2:
            return
        
        try:
            deleted_count = self.check_and_delete_expired_files()
            if deleted_count > 0:
                print(f"Cleanup completed: deleted {deleted_count} expired files")
        except Exception as e:
            print(f"Error during cleanup of expired files: {e}")