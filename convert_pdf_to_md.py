# --- START OF FILE convert_pdf_to_md.py ---

import os
import gc
import time
import PyPDF2
import argparse
from pathlib import Path

# --- IMPORTANT ---
# marker-pdf requires PyTorch and other dependencies.
# Ensure they are installed: pip install marker-pdf torch torchvision torchaudio
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
except ImportError:
    print("Error: marker-pdf or its dependencies not found.")
    print("Please install it: pip install marker-pdf torch torchvision torchaudio")
    print("If you have CUDA, ensure PyTorch is installed with CUDA support.")
    exit()


def convert_pdf_batch(pdf_path_str: str, output_dir_str: str, batch_size: int = 10, max_pages: int = None):
    """Converts a PDF to Markdown in batches using marker-pdf."""
    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)

    if not pdf_path.is_file():
        print(f"Error: PDF file not found at {pdf_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        # Load marker models once
        print("Loading marker models (this may take a while)...")
        model_lst = load_all_models()
        print("Models loaded.")

        # Get total pages
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"Total pages in PDF: {total_pages}")

        # Determine pages to process
        pages_to_process = total_pages
        if max_pages is not None and max_pages < total_pages:
            pages_to_process = max_pages
            print(f"Processing only the first {max_pages} pages.")

        # Process in batches
        for start_page in range(0, pages_to_process, batch_size):
            end_page = min(start_page + batch_size, pages_to_process)
            page_indices = list(range(start_page, end_page)) # marker uses 0-based index list

            batch_num = (start_page // batch_size) + 1
            output_md_filename = f"{pdf_path.stem}_pages_{start_page+1}-{end_page}.md"
            output_md_path = output_dir / output_md_filename

            print(f"\n=== Processing Batch {batch_num}: Pages {start_page+1} to {end_page} ===")
            print(f"Outputting to: {output_md_path}")

            # Force garbage collection before conversion
            gc.collect()
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()


            try:
                # Convert the specified pages of the *original* PDF
                # marker handles page selection internally via `page_indices`
                full_text, out_meta = convert_single_pdf(
                    str(pdf_path),
                    model_lst,
                    page_indices=page_indices,
                    batch_multiplier=1 # Adjust based on GPU VRAM if needed
                )

                # Write output
                with open(output_md_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

                print(f"Successfully converted pages {start_page+1}-{end_page} to '{output_md_path}'")
                # print(f"Metadata for batch: {out_meta}") # Optional: print metadata

            except Exception as e:
                print(f"Error processing batch {batch_num} (pages {start_page+1}-{end_page}): {str(e)}")
                print("Continuing with next batch...")
                # import traceback # Uncomment for detailed error logs
                # traceback.print_exc() # Uncomment for detailed error logs

            # Optional: Wait between batches if memory/GPU issues persist
            # if end_page < pages_to_process:
            #     wait_time = 5
            #     print(f"Waiting {wait_time} seconds before next batch...")
            #     time.sleep(wait_time)


        print("\n=== PDF to Markdown Conversion Complete ===")
        print(f"All markdown files saved to: {output_dir}")

    except Exception as e:
        print(f"A critical error occurred during PDF conversion setup or processing: {str(e)}")
        # import traceback # Uncomment for detailed error logs
        # traceback.print_exc() # Uncomment for detailed error logs
    finally:
         # Clean up models (if possible/needed, depends on marker's structure)
         del model_lst
         gc.collect()
         if torch.cuda.is_available():
              torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown files in batches using marker-pdf.")
    parser.add_argument("pdf_file", help="Path to the input PDF file.")
    parser.add_argument("output_dir", help="Directory to save the output Markdown files.")
    parser.add_argument("-b", "--batch_size", type=int, default=5, help="Number of pages to process per batch (default: 5). Adjust based on system memory/GPU.")
    parser.add_argument("-m", "--max_pages", type=int, default=None, help="Maximum number of pages to convert from the start of the PDF (optional).")

    args = parser.parse_args()

    # Make sure torch is available (marker dependency)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
             print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
             print("CUDA not available, running on CPU (will be slower).")
    except ImportError:
        print("Error: PyTorch not found. Please install it (pip install torch torchvision torchaudio).")
        exit()


    convert_pdf_batch(args.pdf_file, args.output_dir, args.batch_size, args.max_pages)

# --- END OF FILE convert_pdf_to_md.py ---