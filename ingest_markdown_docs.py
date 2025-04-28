# --- START OF FILE ingest_markdown_docs.py ---

import os
import asyncio

# import argparse # No longer needed for directory argument
from pathlib import Path
from dotenv import load_dotenv

# Import from the shared utils file
from ingestion_utils import process_and_store_document

load_dotenv()


async def process_markdown_directory(md_dir: Path):  # Takes a Path object
    """Finds all .md files in a directory and processes them."""
    if not md_dir.is_dir():
        print(f"Error: Directory not found: {md_dir}")
        return

    print(f"Searching for markdown files (.md) in: {md_dir}")
    # Use rglob to find .md files recursively within the directory
    md_files = list(md_dir.rglob("*.md"))

    if not md_files:
        print(f"No markdown files (.md) found in {md_dir} or its subdirectories.")
        return

    print(f"Found {len(md_files)} markdown files to process:")
    for f in md_files:
        print(f"  - {f.relative_to(md_dir.parent)}")  # Print relative path for clarity

    # Process each found file
    process_tasks = []
    for md_file in md_files:
        print(f"\nProcessing file: {md_file.name}...")
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                print(f"Skipping empty file: {md_file.name}")
                continue

            # Use filename as the primary identifier for PDF-derived content
            # Create a pseudo-URL identifier
            identifier = f"pdf://{md_file.name}"  # Using pdf:// convention

            # Create a task for processing and storing this document
            # This function call eventually leads to Supabase insertion
            task = process_and_store_document(
                identifier=identifier,
                content=content,
                source_type="pdf",  # Assuming MD files came from PDFs as per original context
                filename=md_file.name,  # Store original filename in metadata
            )
            process_tasks.append(task)

        except Exception as e:
            print(f"Error reading or queuing file {md_file.name} for processing: {e}")
            # Decide whether to continue or stop on error
            # continue

    # Run all processing tasks concurrently
    if process_tasks:
        print(
            f"\nStarting concurrent processing and ingestion for {len(process_tasks)} files..."
        )
        await asyncio.gather(*process_tasks)
        print("Finished processing all queued markdown files.")
    else:
        print("No valid markdown files were queued for processing.")


async def main_markdown_ingest():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    # Define the target directory relative to the script's location
    # Assumes 'data' folder is in the same directory as this script OR
    # if script is in 'scripts/', it looks for 'scripts/data/'
    # Adjust if your 'data' folder is elsewhere relative to the script
    data_dir = script_dir / "data"

    # If 'data' is in the parent directory (project root) instead:
    # data_dir = script_dir.parent / "data"
    # Choose the correct line based on your project structure ^^^

    print(f"Script location: {script_dir}")
    print(f"Target data directory: {data_dir}")

    await process_markdown_directory(data_dir)


if __name__ == "__main__":
    print("--- Starting Markdown Document Ingestion ---")
    # Now runs without needing command-line arguments
    asyncio.run(main_markdown_ingest())
    print("--- Markdown Document Ingestion Script Finished ---")

# --- END OF FILE ingest_markdown_docs.py ---
