from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SequentialChain
from pprint import pprint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatGroq(model="llama-3.1-8b-instant")

template = """Your job is to come up with a classic dish form the area that the useres suggests.
{location}
YOUR RESPONSE:
"""
prompt_template = PromptTemplate(template=template, input_variables=["locaton"])

location_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="meal")

# Create a template for generating a recipe based on a meal
template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.
 YOUR RESPONSE:
"""
# Create a PromptTemplate with 'meal' as the input variable
prompt_template = PromptTemplate(template=template, input_variables=["meal"])

# The output_key='recipe' defines how this chain's output will be referenced in later chains
dish_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="recipe")

# Create a template for estimating cooking time based on a recipe
# This template asks the LLM to analyze a recipe and estimate preparation time
template = """Given the recipe {recipe}, estimate how much time I need to cook it.
 YOUR RESPONSE:
"""

prompt_template = PromptTemplate(template=template, input_variables=["recipe"])

# The output_key='time' defines the key for this chain's output in the final result
recipe_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="time")

overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=["location"],
    output_variables=["meal", "recipe", "time"],
    verbose=True,
)

result = overall_chain.invoke(input={"location": "China"})
pprint(result)
print(result)

# -- MODERN APPROACH: LCEL --

# Define the templates for each step
location_template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}

YOUR RESPONSE:
"""

dish_template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

YOUR RESPONSE:
"""

time_template = """Given the recipe {recipe}, estimate how much time I need to cook it.

YOUR RESPONSE:
"""

# Create the location chain using LCEL (LangChain Expression Language)
# This chain takes a location and returns a classic dish from that region
location_chain_lcel = (
    PromptTemplate.from_template(location_template)  # Format the prompt with location
    | llm  # Send to the LLM
    | StrOutputParser()  # Extract the string response
)

# Create the dish chain using LCEL
# This chain takes a meal name and returns a recipe
dish_chain_lcel = (
    PromptTemplate.from_template(dish_template)  # Format the prompt with meal
    | llm  # Send to the LLM
    | StrOutputParser()  # Extract the string response
)

# Create the time estimation chain using LCEL
# This chain takes a recipe and returns an estimated cooking time
time_chain_lcel = (
    PromptTemplate.from_template(time_template)  # Format the prompt with recipe
    | llm  # Send to the LLM
    | StrOutputParser()  # Extract the string response
)

# Combine all chains into a single workflow using RunnablePassthrough.assign
# RunnablePassthrough.assign adds new keys to the input dictionary without removing existing ones
overall_chain_lcel = (
    # Step 1: Generate a meal based on location and add it to the input dictionary
    RunnablePassthrough.assign(
        meal=lambda x: location_chain_lcel.invoke({"location": x["location"]})
    )
    # Step 2: Generate a recipe based on the meal and add it to the input dictionary
    | RunnablePassthrough.assign(
        recipe=lambda x: dish_chain_lcel.invoke({"meal": x["meal"]})
    )
    # Step 3: Estimate cooking time based on the recipe and add it to the input dictionary
    | RunnablePassthrough.assign(
        time=lambda x: time_chain_lcel.invoke({"recipe": x["recipe"]})
    )
)
# Run the chain
result = overall_chain_lcel.invoke({"location": "China"})
pprint(result)
