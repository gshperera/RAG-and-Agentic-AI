from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Create a template string for generating recommendations of classic dishes from a given location
# The template includes:
# - Instructions for the task (recommending a classic dish)
# - A placeholder {location} that will be replaced with user input
# - A format indicator for the expected response
template = """Your job is to come up with a classic dish form the area that the useres suggests.
{location}
YOUR RESPONSE:
"""

# Create a PromptTemplate object by providing:
# - The template string defined above
# - A list of input variables that will be used to format the template
prompt_template = PromptTemplate(template=template, input_variables=["locaton"])

llm = ChatGroq(model="llama-3.1-8b-instant")

# Create an LLMChain that connects:
# - The Llama language model (llama_llm)
# - The prompt template configured for location-based dish recommendations
# - An output_key 'meal' that specifies the key name for the chain's response in the output dictionary
location_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="meal")

# This will:
# 1. Format the template with {location: 'China'}
# 2. Send the formatted prompt to the Llama LLM
# 3. Return a dictionary with the response under the key 'meal'
location_chain.invoke(input={"location": "China"})

# -- MODERN APPROACH: LCEL --

prompt = PromptTemplate.from_template(template)

# Create a chain using LangChain Expression Language (LCEL) with the pipe operator
# This creates a processing pipeline that:
# 1. Formats the prompt with the input values
# 2. Sends the formatted prompt to the Llama LLM
# 3. Parses the output to extract just the string response
location_chain_lcel = prompt | llm | StrOutputParser()

result = location_chain_lcel.invoke({"location": "Sri Lanka"})
print(result)
