from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "How much did Microseft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# -- Answer Generation Part --

template = f"""Based on the following documents, please answer this questions: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't hve enough information to answer that question based on the provided documents.
"""

llm = ChatGroq(model="llama-3.1-8b-instant")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=template),
]

result = llm.invoke(messages)

print("\n--- Generated Response ---")
print(result)
print("Content only:")
print(result.content)
