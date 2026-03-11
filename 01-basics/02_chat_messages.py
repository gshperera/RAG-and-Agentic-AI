from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# -- CHAT MODEL --
llm = ChatGroq(model="llama-3.1-8b-instant")

# -- CHAT MESSAGE --
response = llm.invoke(
    [
        SystemMessage(
            content="You are a helpful AI bot that assists a user in choosing the perfect book to read in one short sentence."
        ),
        HumanMessage(content="I enjoy positive vibe movies, what should I read?"),
    ]
)

print(response)

# You can use these message types to pass an entire chat history along with the AI's responses to the model

response_2 = llm.invoke(
    [
        SystemMessage(
            content="You are a supportive AI bot that suggests fitness activities to a user in one short sentence"
        ),
        HumanMessage(content="I like high-intensity workouts, what should I do?"),
        AIMessage(content="You should try a CrossFit class"),
        HumanMessage(content="How often should I attend?"),
    ]
)

print(response_2)
