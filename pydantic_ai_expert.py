# --- START OF FILE pydantic_ai_expert.py ---

from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
from typing import List, Dict, Any

# Use Pydantic AI's Gemini integration
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.gemini import GeminiModel  # Import GeminiModel
import google.generativeai as genai  # Import Gemini client library
from supabase import Client

# Import the shared embedding function (now using Gemini)
from ingestion_utils import get_embedding as get_gemini_embedding

load_dotenv()

# Configure Gemini client (ensure GOOGLE_API_KEY is set in .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# Use Gemini model defined in environment or default
llm_name = os.getenv("GEMINI_GENERATION_MODEL", "gemini-1.5-flash-latest")
model = GeminiModel(llm_name)  # Use GeminiModel

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class PydanticAIDeps:
    supabase: Client
    # No longer need openai_client, genai is configured globally
    # or could be passed explicitly if preferred


system_prompt = """
You are an expert assistant knowledgeable about construction, building activity and development work in Mumbai and specific uploaded PDF documents 
(like the National Building Code of India 2016 Volume 1 and 2). You have access to tools that can search through all ingested documentation content (web pages and PDFs).
You have NO general knowledge beyond these documents.
Your primary goal is to answer user questions based *only* on the information retrieved from the documentation using your tools.

**Your Core Mandate:**
1.  **NEVER answer from general knowledge.** Your *only* source of information is the documentation retrieved by your tools.
2.  **ALWAYS use the `retrieve_relevant_documentation` tool FIRST** for *any* user question, including questions about your capabilities or sources. Do not skip this step.
3.  **BASE your answer STRICTLY on the retrieved content.** 
4.  **Start your answer by indicating the source (e.g., "According to the National Building Code of India 2016 Volume 2 documentation retrieved..." or "Based on the building activity and development work in areas under the entire jurisdiction of the Municipal Corporation of Greater Mumbai content retrieved...").
5.  **If the tools return no relevant information OR the retrieved information doesn't answer the question, state CLEARLY that the answer is not found in the available documentation.
6.  ** Do NOT invent information or fall back to general knowledge.
7.  **Answering about sources:** If asked where your information comes from, explain that you retrieve relevant chunks from indexed building activity and development work in areas under the entire jurisdiction of the Municipal Corporation of Greater Mumbai documentation, National Building Code of India 2016 Volume 1, National Building Code of India 2016 Volume 2 and specific PDFs using the `retrieve_relevant_documentation` tool based on the query's similarity to the content. Mention the tool by name.

**Tool Usage Flow:**
- Receive user query.
- Immediately call `retrieve_relevant_documentation` with the user query.
- Analyze the results from the tool.
- Synthesize an answer *only* from the results, citing the source type/identifier provided in the tool output.
- If results are insufficient, state that.
- Only use `list_documentation_sources` or `get_source_content` if specifically needed to clarify sources mentioned in the RAG results or if the initial RAG was insufficient and a specific source needs further investigation.
"""

# Initialize the Agent with GeminiModel
mumbai_expert = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)

# Note: The get_embedding function used by the RAG tool now comes from ingestion_utils
# and uses the Gemini embedding model configured there.


@mumbai_expert.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Searches across all indexed documentation (web pages, PDFs) using vector similarity (RAG)
    to find the most relevant chunks of text related to the user's query.
    This should be the FIRST tool used to answer any question.

    Args:
        user_query: The user's question or topic to search for.

    Returns:
        A formatted string containing the top 5 most relevant documentation chunks,
        including their titles and source identifiers (URL or filename). Returns
        "No relevant documentation found." if nothing matches well.
    """
    print(f"\n--- TOOL CALL: retrieve_relevant_documentation ---")
    print(f"Query: '{user_query}'")
    try:
        # Get the embedding for the query using the shared Gemini embedding function
        query_embedding = await get_gemini_embedding(user_query)
        print("Generated query embedding.")

        # Query Supabase using the pgvector match function
        # Search across *all* sources by default (no source filter here)
        # Match count can be adjusted
        result = await asyncio.to_thread(
            ctx.deps.supabase.rpc(
                "match_site_pages",  # Ensure this RPC function exists in your Supabase
                {
                    "query_embedding": query_embedding,
                    "match_count": 5,
                    # 'filter': {} # No specific filter, search all documents
                    # Example filter if needed: 'filter': {'metadata->>source_type': 'pdf'}
                },
            ).execute
        )
        print(
            f"Supabase RPC returned: {len(result.data) if result.data else 0} results."
        )
        # print(f"DEBUG: Supabase RAG result: {result}") # For debugging

        if not result.data:
            print("--- TOOL RETURN (No Results) ---")
            return "No relevant documentation found in any source."

        # Format the results clearly indicating the source
        formatted_chunks = []
        for doc in result.data:
            source_id = doc.get(
                "url", "Unknown Source"
            )  # 'url' column holds URL or pdf://filename
            title = doc.get("title", "Untitled Chunk")
            content = doc.get("content", "No content.")
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")  # pdf or web

            chunk_text = f"""
--- Source ---
Type: {source_type.upper()}
Identifier: {source_id}
Title: {title}

Content Chunk:
{content}
--- End Source ---
"""
            formatted_chunks.append(chunk_text)

        print(
            f"Tool: retrieve_relevant_documentation - Found {len(formatted_chunks)} relevant chunks."
        )
        return "\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error in retrieve_relevant_documentation tool: {e}")
        # Consider logging the full traceback for debugging
        # import traceback; traceback.print_exc()
        return f"An error occurred while searching the documentation: {str(e)}"


@mumbai_expert.tool
async def list_documentation_sources(
    ctx: RunContext[PydanticAIDeps],
) -> List[Dict[str, str]]:
    """
    Retrieves a list of all unique documentation sources (web page URLs and PDF filenames)
    that have been indexed.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the 'type' (web/pdf)
                               and 'identifier' (URL or filename) of a source.
    """
    print("Tool: list_documentation_sources called.")
    sources = []
    try:
        # Query Supabase for distinct URLs (which store URL or pdf://filename) and metadata
        result = await asyncio.to_thread(
            ctx.deps.supabase.from_("site_pages")
            .select(
                "url, metadata"
            )  # url holds the identifier, metadata has source_type
            .execute
        )

        if not result.data:
            print("Tool: list_documentation_sources - No sources found.")
            return []

        # Process to get unique sources with type
        seen_identifiers = set()
        for doc in result.data:
            identifier = doc.get("url")
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")

            if identifier and identifier not in seen_identifiers:
                sources.append({"type": source_type, "identifier": identifier})
                seen_identifiers.add(identifier)

        # Sort for consistency (optional)
        sources.sort(key=lambda x: (x["type"], x["identifier"]))
        print(
            f"Tool: list_documentation_sources - Found {len(sources)} unique sources."
        )
        return sources

    except Exception as e:
        print(f"Error in list_documentation_sources tool: {e}")
        return [
            {"type": "error", "identifier": f"Could not retrieve source list: {str(e)}"}
        ]


@mumbai_expert.tool
async def get_source_content(
    ctx: RunContext[PydanticAIDeps], source_identifier: str
) -> str:
    """
    Retrieves the full combined content of a specific documentation source
    (a web page URL or a PDF filename identifier like 'pdf://filename.md').

    Args:
        source_identifier: The unique identifier (URL for web pages, or the
                           identifier like 'pdf://filename.md' used during ingestion for PDFs)
                           of the source to retrieve. Should match identifiers from
                           RAG results or the source list.

    Returns:
        str: The complete source content by combining all its chunks in order,
             or an error message if the source is not found.
    """
    print(f"Tool: get_source_content called for identifier: '{source_identifier}'")
    try:
        # Query Supabase for all chunks matching the identifier, ordered by chunk_number
        result = await asyncio.to_thread(
            ctx.deps.supabase.from_("site_pages")
            .select("title, content, chunk_number, metadata")
            .eq("url", source_identifier)  # 'url' column holds the identifier
            .order("chunk_number")
            .execute
        )

        print(f"DEBUG: Supabase get_source_content result: {result}")  # For debugging

        if not result.data:
            print(
                f"Tool: get_source_content - No content found for identifier: {source_identifier}"
            )
            return f"No content found for source: {source_identifier}"

        # Combine the chunks
        # Use the title from the first chunk (or derive a main title)
        first_chunk = result.data[0]
        metadata = first_chunk.get("metadata", {})
        source_type = metadata.get("source_type", "unknown")
        page_title = first_chunk.get("title", f"Content for {source_identifier}")
        # Attempt to get a cleaner title if chunk titles include numbering/context
        if " - Chunk" in page_title or "Page segment" in page_title:
            base_title_query = await asyncio.to_thread(
                ctx.deps.supabase.from_("site_pages")
                .select("title")
                .eq("url", source_identifier)
                .eq("chunk_number", 0)
                .maybe_single()  # Get just the first chunk's title if possible
                .execute
            )
            if base_title_query.data:
                page_title = base_title_query.data.get("title", page_title)

        full_content = [f"# Source: {source_identifier} ({source_type.upper()})"]
        full_content.append(f"## Title: {page_title}\n")

        # Add each chunk's content
        for chunk in result.data:
            full_content.append(chunk["content"])

        print(
            f"Tool: get_source_content - Successfully retrieved content for {source_identifier}"
        )
        return "\n\n".join(full_content)

    except Exception as e:
        print(f"Error in get_source_content tool for {source_identifier}: {e}")
        return f"Error retrieving content for {source_identifier}: {str(e)}"


# --- END OF FILE pydantic_ai_expert.py ---
