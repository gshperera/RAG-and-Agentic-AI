# Conversation buffer memory allows for the storage of messages, which you use to extract messages to a variable. Consider using conversation buffer memory in a chain setting verbose=True so that the prompt is visible

from langchain_classic.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory

llm = ChatGroq(model="llama-3.1-8b-instant")

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

conversation.invoke(input="Hello, I am a little cat. Who are you?")

conversation.invoke(input="What can you do?")
conversation.invoke(input="Who am I?.")
