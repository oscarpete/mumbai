# --- START OF FILE ingest_web_docs.py ---

import asyncio
import requests
from xml.etree import ElementTree
from typing import List
from dotenv import load_dotenv

# Import from the shared utils file
from ingestion_utils import process_and_store_document

# NOTE: crawl4ai might have its own dependencies like playwright, beautifulsoup4
# Make sure they are installed: pip install crawl4ai beautifulsoup4 lxml requests
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
except ImportError:
    print("Error: crawl4ai not found. Please install it: pip install crawl4ai")
    print("You might also need to run: playwright install --with-deps")
    exit()


load_dotenv()

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    # Consider CACHE_ONLY_URL or REFRESH mode depending on needs
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    # Check if crawler needs explicit browser path if playwright isn't global
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                print(f"Attempting to crawl: {url}")
                try:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="pydantic-ai-crawl" # Use a consistent session ID if needed
                    )
                    if result.success and result.markdown_v2:
                        print(f"Successfully crawled: {url}")
                        await process_and_store_document(
                            identifier=url,
                            content=result.markdown_v2.raw_markdown,
                            source_type='web'
                        )
                    elif result.success and not result.markdown_v2:
                         print(f"Crawled but no markdown content: {url}")
                    else:
                        print(f"Failed crawl: {url} - Error: {result.error_message}")
                except Exception as e:
                    print(f"Exception during crawl/process for {url}: {e}")

        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls(sitemap_url: str = "https://ai.pydantic.dev/sitemap.xml") -> List[str]:
    """Get URLs from a sitemap."""
    urls = []
    try:
        response = requests.get(sitemap_url, timeout=15)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        # Namespace might vary, adjust if needed
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        loc_elements = root.findall('.//ns:loc', namespace)
        if not loc_elements: # Try without namespace if not found
             loc_elements = root.findall('.//loc')

        urls = [loc.text for loc in loc_elements if loc.text]
        print(f"Found {len(urls)} URLs in {sitemap_url}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        print(f"Error parsing sitemap XML from {sitemap_url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred getting URLs from {sitemap_url}: {e}")

    # Basic filtering (optional, refine as needed)
    urls = [u for u in urls if u.startswith("http") and 'ai.pydantic.dev' in u]
    return urls

async def main_web_crawl():
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl. Exiting.")
        return

    print(f"Starting crawl for {len(urls)} URLs...")
    await crawl_parallel(urls, max_concurrent=3) # Keep concurrency low initially
    print("Web crawling finished.")

if __name__ == "__main__":
    # Ensure playwright is installed if crawl4ai needs it
    # Run `playwright install --with-deps` in your terminal if you encounter issues
    print("Starting web document ingestion...")
    asyncio.run(main_web_crawl())
    print("Web document ingestion script finished.")

# --- END OF FILE ingest_web_docs.py ---