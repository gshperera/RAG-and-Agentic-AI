from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.stores import InMemoryStore

# --Load the document--
loader = PyPDFLoader(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
)
document = loader.load()

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# Set up two different text splitters for a hierarchical splitting approack:

# 1. Parent splitter creates larger chunks (2000 characters)
# This is used to split documents into larger, more contextually complete sections
parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator="\n")

# 2. Child splitter creates smaller chunks (400 characters)
# This is used to split the parent chunks into smaller pieces for more precise retrieval
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator="\n")

# Create a Chroma vector store with:
# - A specific collection name "split_parents" for organization
# - The previously configured embeddings function
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embedding_model
)

# Set up an in-memory storage layer for the parent documents
# This will store the larger chunks that provide context, but won't be directly embedded
store = InMemoryStore()

# Create a ParentDocumentRetriever instance that implements hierarchical document retrieval
retriever = ParentDocumentRetriever(
    # The vector store where child document embeddings will be stored and searched
    # This Chroma instance will contain the embeddings for the smaller chunks
    vectorstore=vectorstore,
    
    # The document store where parent documents will be stored
    # These larger chunks won't be embedded but will be retrieved by ID when needed
    docstore=store,
    
    # The splitter used to create small chunks (400 chars) for precise vector search
    # These smaller chunks are embedded and used for similarity matching
    child_splitter=child_splitter,
    
    # The splitter used to create larger chunks (2000 chars) for better context
    # These parent chunks provide more complete information when retrieved
    parent_splitter=parent_splitter,
)

# Add documents to the hierarchical retrieval system
retriever.add_documents(document)

# Retrieves and counts the number of parent document IDs stored in the document store
len(list(store.yield_keys()))

# Verify that the underlying vector store still retrieves the small chunks
sub_docs = vectorstore.similarity_search("Langchain")

print(sub_docs[0].page_content)

retrieved_docs = retriever.invoke("Langchain")

print(retrieved_docs[0].page_content)