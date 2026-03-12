from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Load the pdf
paper_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
loader = PyPDFLoader(paper_url)
pdf_document = loader.load()

# Load content from website
web_url = "https://docs.langchain.com/oss/python/integrations/document_loaders"
web_loader = WebBaseLoader(web_url)
web_document = web_loader.load()

# Create 2 different text splitters
splitter_1 = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
splitter_2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Apply both splitters to the PDF document
chunks_1 = splitter_1.split_documents(pdf_document)
chunks_2 = splitter_2.split_documents(pdf_document)


# Function to display chunk statistics
def display_document_stats(chunks, splitter):
    """Display statistics about a list of document chunks"""

    print(f"\n=== {splitter} Statistics ===")
    total_chunks = len(chunks)
    print(f"Total number of chunks: {total_chunks}")

    all_metadata_keys = set()
    for num, chunk in enumerate(chunks):
        print(f"Chunk {num}: {len(chunk.page_content)} characters")
        all_metadata_keys.update(chunk.metadata.keys())

    print(all_metadata_keys)


# Display stats for both chunk sets
display_document_stats(chunks_1, "Splitter 1")
display_document_stats(chunks_2, "Splitter 2")
