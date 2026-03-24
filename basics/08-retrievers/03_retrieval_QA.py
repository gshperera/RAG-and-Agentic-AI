from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGroq(model="llama-3.1-8b-instant")

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

# Create a RetrievalQA chain by configuring:
qa = RetrievalQA.from_chain_type(
    # The language model to use for generating answers
    llm=llm,
    # The chain type "stuff" means all retrieved documents are simply concatenated and passed to the LLM
    chain_type="stuff",
    # The retriever component that will fetch relevant documents
    # docsearch.as_retriever() converts the vector store into a retriever interface
    retriever=store.as_retriever(),
    # Whether to include the source documents in the response
    # Set to False to return only the generated answer
    return_source_documents=False,
)

query = "What is this paper discussing?"

# Execute the QA chain with the query
# This will:
# 1. Send the query to the retriever to get relevant documents
# 2. Combine those documents using the "stuff" method
# 3. Send the query and combined documents to the Llama LLM
# 4. Return the generated answer (without source documents)
qa.invoke(query)
