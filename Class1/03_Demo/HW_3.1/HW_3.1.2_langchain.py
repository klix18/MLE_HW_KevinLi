# Example: Using LCEL to reproduce a "Basic Prompting" scenario
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama 

# Few-shot prompt
system_prompt = """You are a helpful AI assistant that provides ONLY the capital name as a one-word answer.

Here are some examples of good responses:

Question: What is the capital of France?
Answer: Paris.

Question: What is the capital of Japan?
Answer: Tokyo.

Question: What is the capital of Brazil?
Answer: Brasília.

Question: What is the capital of India?
Answer: New Delhi.

IMPORTANT: Respond with ONLY the capital name, nothing else. No extra words, no explanations."""

# 2. Define the human message template
human_template = "What is the capital of {topic}?"

# 3. Create the chat prompt template with system and human messages
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_template)
])

# 4. Define the model
model = ChatOllama(model = "llama2")  # Using Ollama 

# 5. Chain the components together using LCEL
chain = (
    # LCEL syntax: use the pipe operator | to connect each step
    {"topic": RunnablePassthrough()}  # Accept user input
    | chat_prompt                     # Transform it into a prompt message
    | model                           # Call the model
    | StrOutputParser()               # Parse the output as a string
)

# 5. Execute - Interactive version
while True:
    user_input = input("Enter a country to know its capital (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    
    result = chain.invoke(user_input)
    print(f"User prompt: 'What is the capital of {user_input}?'")
    print("Model answer:", result)
    print("-" * 50)