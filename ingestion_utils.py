# --- START OF FILE ingestion_utils.py ---

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client

load_dotenv()

# Initialize Gemini and Supabase clients
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    sys.exit("Error: GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- IMPORTANT ---
# Choose the Gemini models
# For generation (title/summary): gemini-1.5-flash-latest is good and fast
# For embeddings: text-embedding-004 or models/embedding-001 are common choices.
# Ensure your Supabase vector column dimension matches the embedding model's dimension!
# - text-embedding-004: 768 dimensions
# - models/embedding-001: 768 dimensions
# - text-embedding-ada-002 (OpenAI): 1536 dimensions
# - text-embedding-3-small (OpenAI): 1536 dimensions
GENERATION_MODEL_NAME = os.getenv("GEMINI_GENERATION_MODEL", "gemini-1.5-flash-latest")
EMBEDDING_MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001") # Ensure this matches Supabase vector column dimension

generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
embedding_model_string = EMBEDDING_MODEL_NAME # Name used in API call

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
if not supabase_url or not supabase_key:
    sys.exit("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not found.")

supabase: Client = create_client(supabase_url, supabase_key)

# --- Supabase Table Schema Note ---
# Ensure your 'site_pages' table in Supabase has columns like:
# - id (int8, primary key, auto-increment)
# - url (text)
# - chunk_number (int4)
# - title (text)
# - summary (text)
# - content (text)
# - metadata (jsonb)
# - embedding (vector, dimension matching your chosen GEMINI_EMBEDDING_MODEL, e.g., 768 for models/embedding-001)
# - created_at (timestamp with time zone, default now())
# Add appropriate indexes, especially on 'embedding' (e.g., ivfflat or hnsw) and potentially 'url', 'metadata'.

@dataclass
class ProcessedChunk:
    source_identifier: str  # URL for web, filename for pdf
    source_type: str        # 'web' or 'pdf'
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 4000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs.
       Adjusted chunk_size for typical embedding model limits."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk_segment = text[start:end]
        potential_end = end

        # Prioritize code block boundaries
        code_block_end = chunk_segment.rfind('```')
        if code_block_end != -1:
            # Check if it's an *ending* marker by looking ahead slightly
            if text.find('```', start + code_block_end + 3, start + code_block_end + 6) != -1 or end > text_length-3:
                 potential_end = start + code_block_end + 3 # Include the marker

        # If no code block end found, try paragraph break
        elif '\n\n' in chunk_segment:
            last_para_break = chunk_segment.rfind('\n\n')
            if last_para_break > chunk_size * 0.5: # Break if break is past midpoint
                 potential_end = start + last_para_break + 2 # Include the newline chars

        # If no good break found, take the chunk size (might cut mid-sentence)
        # This simple approach avoids complex sentence boundary detection
        final_end = min(potential_end, text_length)

        chunk = text[start:final_end].strip()
        if chunk:
            chunks.append(chunk)
        start = final_end

    return chunks


async def get_title_and_summary(chunk: str, source_identifier: str) -> Dict[str, str]:
    """Extract title and summary using Gemini."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
Return *ONLY* a valid JSON object (no preamble, no explanation) with 'title' and 'summary' keys.
- For the title: If this seems like the start of a document (web page or file), extract its main title. If it's a middle chunk, derive a short, descriptive title for this specific chunk's content.
- For the summary: Create a concise summary (1-2 sentences) of the main points in this specific chunk.
Keep both title and summary concise but informative.
"""
    prompt = f"""System Instruction:
{system_prompt}

Source Identifier (URL or Filename): {source_identifier}

Content Chunk (first 1000 chars):
{chunk[:1000]}...

Respond with ONLY the JSON object."""

    try:
        # Use generate_content with response_mime_type for structured output
        response = await generation_model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        # Gemini might wrap JSON in ```json ... ```, try to extract
        content = response.text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        data = json.loads(content.strip())
        return {
            "title": data.get("title", "Default Title"),
            "summary": data.get("summary", "Default Summary")
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini for title/summary: {e}\nResponse: {response.text}")
        return {"title": "Error: Invalid JSON", "summary": "Could not generate summary due to formatting error."}
    except Exception as e:
        # Catch other potential Gemini API errors
        print(f"Error getting title and summary from Gemini: {e}")
        # Log the response if available and useful
        error_response_text = getattr(response, 'text', 'No response text available')
        print(f"Gemini raw response on error: {error_response_text}")
        return {"title": "Error processing title", "summary": "Error processing summary"}


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Gemini."""
    try:
        # Ensure text is not empty
        if not text.strip():
            print("Warning: Attempted to embed empty text. Returning zero vector.")
            # Match the dimension of your chosen embedding model
            embed_dim = 768 # Default for text-embedding-004 / models/embedding-001
            # Potentially fetch dimension dynamically if needed, but hardcoding is simpler
            return [0.0] * embed_dim

        result = await genai.embed_content_async(
            model=embedding_model_string,
            content=text,
            task_type="retrieval_document" # or "retrieval_query" if embedding a query
        )
        return result['embedding']
    except Exception as e:
        print(f"Error getting embedding from Gemini: {e}")
        # Match the dimension of your chosen embedding model
        embed_dim = 768 # Default for text-embedding-004 / models/embedding-001
        return [0.0] * embed_dim # Return zero vector on error


async def process_chunk(chunk: str, chunk_number: int, source_identifier: str, source_type: str, filename: Optional[str] = None) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, source_identifier)

    # Get embedding
    embedding = await get_embedding(chunk) # Embed the actual chunk content

    # Create metadata
    metadata = {
        "source_type": source_type,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "processing_model": GENERATION_MODEL_NAME,
        "embedding_model": embedding_model_string,
    }
    if source_type == 'web':
         metadata["url_path"] = urlparse(source_identifier).path
    elif source_type == 'pdf' and filename:
         metadata["filename"] = filename

    return ProcessedChunk(
        source_identifier=source_identifier,
        source_type=source_type,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.source_identifier, # Use source_identifier for URL/filename
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata, # Contains source_type, filename etc.
            "embedding": chunk.embedding
        }

        # Use upsert to potentially avoid duplicates if needed, based on url and chunk_number
        # result = await supabase.table("site_pages").upsert(data, on_conflict='url, chunk_number').execute()
        # Or just insert:
        result = await asyncio.to_thread(
            supabase.table("site_pages").insert(data).execute
        )

        print(f"Inserted chunk {chunk.chunk_number} for {chunk.source_type}: {chunk.source_identifier}")
        # print(f"DEBUG: Insert result: {result}") # Optional: for debugging
        return result
    except Exception as e:
        print(f"Error inserting chunk {chunk.chunk_number} for {chunk.source_identifier}: {e}")
        # print(f"Failed data: {data}") # Optional: Log data that failed
        return None

async def process_and_store_document(identifier: str, content: str, source_type: str, filename: Optional[str] = None):
    """Process a document (from web or file) and store its chunks."""
    if not content:
        print(f"Skipping {identifier} - content is empty.")
        return

    # Split into chunks
    chunks = chunk_text(content)
    if not chunks:
        print(f"Skipping {identifier} - no chunks generated.")
        return

    print(f"Processing {identifier} ({source_type}) - {len(chunks)} chunks")

    # Process chunks in parallel (consider rate limits for API calls)
    process_tasks = [
        process_chunk(chunk, i, identifier, source_type, filename)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*process_tasks)

    # Filter out any potential None results from failed processing
    valid_chunks = [chunk for chunk in processed_chunks if chunk is not None]

    # Store chunks sequentially or in smaller parallel batches to manage load/rate limits
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in valid_chunks
    ]
    await asyncio.gather(*insert_tasks)
    print(f"Finished storing chunks for {identifier}")

# --- END OF FILE ingestion_utils.py ---