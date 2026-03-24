from langchain_groq import ChatGroq
from langchain_classic.memory import ChatMessageHistory

llm = ChatGroq(model="llama-3.1-8b-instant")

# Create a new conversation history object
# This will store the back-and-forth messages in the conversation
history = ChatMessageHistory()

# Add an initial greeting message from the AI to the history
# This represents a message that would have been sent by the AI assistant
history.add_ai_message("hi!")

# Add a user's question to the conversation history
# This represents a message sent by the user
history.add_user_message("what is the capital of France?")

print(history.messages)

# You can pass these messages in history to the model to generate a response. the code below is retrieving all messages from the ChatMessageHistory object and passing them to the llm to generate a contextually appropriate response based on the conversation history.
result = llm.invoke(history.messages)
print(result)

history.add_ai_message(result)
history.messages
