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

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",  # Optional: save to disk
)

# query = "langchain"
# similar_docs = store.similarity_search(query)
# print(similar_docs)

# Use the vector store as a retriever
# This converts the vector store into a retriever interface that can fetch relevant documents
retriever = store.as_retriever()

# This will:
# 1. Convert the query text "Langchain" into an embedding vector
# 2. Perform a similarity search in the vector store using this embedding
# 3. Return the most semantically similar documents to the query
docs = retriever.invoke("langchain")

docs[0]
