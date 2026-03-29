from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
)

llm = ChatGroq(model="llama-3.1-8b-instant")

# Store our conversation as messages
chat_history = []


def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: If there's chat history, rewrite the question to be standalone
    if chat_history:
        messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                ),
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {user_question}")]
        )

        result = llm.invoke(messages)
        search_question = result.content.strip()
        print(f"Rewritten question for retrieval: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve relevant documents based on the rewritten question
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(search_question)

    print(f"\nRetrieved {len(relevant_docs)} relevant documents.")
    for i, doc in enumerate(relevant_docs, 1):
        # Show first 2 lines of each document
        doc_preview = "\n".join(doc.page_content.splitlines()[:2])
        print(f"Document {i} preview:\n{doc_preview}...\n")

    # Step 3: Generate answer using the retrieved documents and the original question
    template = f"""Based on the following documents, please answer this question: {user_question}
    
    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    messages = (
        [
            SystemMessage(
                content="You are a helpful assistant that answers questions based on provided documents."
            )
        ]
        + chat_history
        + [
            HumanMessage(content=template),
        ]
    )

    result = llm.invoke(messages)
    answer = result.content

    # Add the new question to the chat history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\n--- Answer ---\n{answer}\n")
    return answer


# Simple loop to ask multiple questions
def start_chat():
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break
        ask_question(user_input)


if __name__ == "__main__":
    start_chat()
