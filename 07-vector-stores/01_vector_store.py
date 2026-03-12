from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --Load the document--
loader = PyPDFLoader(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
)
document = loader.load()

# --Splitting Document into Chunks--
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)

"""
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
store = Chroma.from_documents(chunks, embedding_model)

# we cannot directly pass SentenceTransformer object to this
Error: AttributeError: 'SentenceTransformer' object has no attribute 'embed_documents'

We can use HuggingFaceEmbeddings wrapper instead
HuggingFaceEmbeddings is a wrapper inside LangChain.
Internally it still uses SentenceTransformers, but it formats the outputs so they work with:
- Chroma
- FAISS
- Pinecone
- LangChain retrievers
"""
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",  # Optional: save to disk
)

query = "langchain"
similar_docs = store.similarity_search(query)
print(similar_docs)
