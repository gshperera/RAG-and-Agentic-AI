from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --Load the document--
loader = PyPDFLoader(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
)
document = loader.load()
print(f"Length of Documents: {len(document)}")

# --Splitting Document into Chunks--
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")

chunks = text_splitter.split_documents(document)
print(f"Length of Chunks: {len(chunks)}")

chunks[5].page_content
