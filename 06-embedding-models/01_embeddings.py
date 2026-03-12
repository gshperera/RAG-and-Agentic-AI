from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --Load the document--
loader = PyPDFLoader(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
)
document = loader.load()

# --Splitting Document into Chunks--
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)

# --Embedding--
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chunk_contents = [chunk.page_content for chunk in chunks]
print(f"Number of chunks: {len(chunk_contents)}")

# This only gives you vectors
embeddings = embedding_model.encode(chunk_contents)
print(f"Number of vector embeddings: {len(embeddings)}")
print(f"Length of one vector: {len(embeddings[0])}")

print(embeddings)
