import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")
import os
import json
from typing import List
from pydantic import BaseModel, Field

# Updated imports for LangChain v0.2:
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Use the correct agent and tool classes:
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool

# Use updated community imports:
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

# Import your custom CLIPEmbeddings instance from the vector embeddings module.
from embeddings.build_vectorstore_chroma import clip_embedding_model

from config import (
    openai_api_key,
    VECTORSTORE_TEXT_PATH_Chroma,
    VECTORSTORE_IMAGE_PATH_Chroma
)

# -----------------------------------------------------------------------------
# 1) Define the output schema and create an output parser.
# -----------------------------------------------------------------------------
class Recommendations(BaseModel):
    message: str = Field(..., description="A human-friendly response message.")
    products: List[int] = Field(..., description="A list of product IDs recommended to the user.")

parser = PydanticOutputParser(pydantic_object=Recommendations)
format_instructions = parser.get_format_instructions()

# -----------------------------------------------------------------------------
# 2) Load persistent Chroma vector stores.
# -----------------------------------------------------------------------------
def load_text_index():
    """Load the persistent Chroma vector store for text products."""
    text_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return Chroma(
        persist_directory=VECTORSTORE_TEXT_PATH_Chroma,
        embedding_function=text_embedding,
        collection_name="text_products"
    )

def load_image_index():
    """Load the persistent Chroma vector store for image products using the imported CLIPEmbeddings."""
    return Chroma(
        persist_directory=VECTORSTORE_IMAGE_PATH_Chroma,
        embedding_function=clip_embedding_model,
        collection_name="image_products"
    )

# -----------------------------------------------------------------------------
# 3) Define the RAG tool function.
# -----------------------------------------------------------------------------
def retrieve_products_rag(query: str) -> str:
    """
    Implements a retrieval augmented generation (RAG) pipeline:
      - Queries both persistent text and image vector stores (top-5 each).
      - Deduplicates products based on metadata "id".
      - Constructs an augmented context string with product names and IDs.
      - Calls ChatOpenAI to generate a final JSON response conforming to the Recommendations schema.
    """
    text_store = load_text_index()
    image_store = load_image_index()

    text_results = text_store.similarity_search(query, k=5)
    image_results = image_store.similarity_search(query, k=5)

    # Deduplicate results (assumes each doc.metadata contains "id")
    combined = {}
    for doc in text_results + image_results:
        pid = doc.metadata.get("id")
        if pid is None:
            continue
        if pid not in combined:
            name = doc.metadata.get("productDisplayName", doc.page_content)
            combined[pid] = name

    context_lines = [f"Product ID: {pid}, Name: {name}" for pid, name in combined.items()]
    context_str = "\n".join(context_lines)

    augmented_prompt = f"""
You are a shopping assistant. Use the following retrieved product context to answer the query.
Retrieved Products:
{context_str}

Query: {query}

Based on the above, provide a final answer in JSON format with exactly two keys:
"message" - a human-friendly response,
"products" - a list of product IDs (integers) recommended.
Do not include any extra text.
"""
    llm_for_rag = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4-turbo",
        temperature=0
    )
    response = llm_for_rag.invoke([{"role": "system", "content": augmented_prompt}])
    return response.content

# -----------------------------------------------------------------------------
# 4) Create tools and memory.
# -----------------------------------------------------------------------------
def create_tools_and_memory():
    rag_tool = Tool(
        name="RAG-Product-Retrieval",
        func=retrieve_products_rag,
        description=(
            "Retrieves top-5 similar products from both text and image collections, "
            "deduplicates them, augments the context, and returns a JSON string with keys 'message' and 'products'."
        )
    )
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return [rag_tool], memory

# -----------------------------------------------------------------------------
# 5) Create the agent and executor 
# -----------------------------------------------------------------------------
def create_agent_executor():
    tools, memory = create_tools_and_memory()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4-turbo",
        temperature=0
    )

    # Define the system message
    system_message = SystemMessagePromptTemplate.from_template(
        """
        You are a helpful shopping assistant. Answer customer queries ONLY by outputting a valid JSON object with exactly two keys: "message" and "products". Do not output any internal reasoning or extra text.

        Format Instructions:
        {{format_instructions}}

        Example:
        User: "I want a black shoe with yellow lines"
        AI: {{"message": "Here are some suggestions: ...", "products": [12345, 67890]}}
        """
    )

    # Define the human message
    human_message = HumanMessagePromptTemplate.from_template("{input}")

    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message,
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    ).partial(format_instructions=format_instructions)

    # Create the agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    # Wrap in AgentExecutor to handle memory and execution
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)


def run_interactive():
    agent_executor = create_agent_executor()
    print("\n=== Welcome to the RAG Shopping Assistant! ===\n")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        raw_output = agent_executor.invoke({"input": user_input})
        raw_output = raw_output["output"]

        print("\nDEBUG OUTPUT:\n", raw_output)
        try:
            structured_output = parser.parse(raw_output)
        except OutputParserException:
            structured_output = {"message": raw_output, "products": []}
        print(f"\nAI: {structured_output}\n")

from langchain_openai import ChatOpenAI
from langchain.output_parsers import RetryOutputParser

retry_llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4-turbo",
    temperature=0
)

retry_parser = RetryOutputParser.from_llm(
    parser=parser, 
    llm=retry_llm,
    max_retries=3
)

def process_query(agent: AgentExecutor, query: str) -> dict:

    try:
        raw_output_dict = agent.invoke({"input": query})
        raw_output = raw_output_dict.get("output", "")
    except Exception as agent_exception:
        return {"message": f"Agent invocation failed: {agent_exception}", "products": []}
    
    try:

        structured_output = retry_parser.parse_with_prompt(
            completion=raw_output, 
            prompt_value=query  # Using the query as context for the retry correction.
        )
        return structured_output.dict()

    except Exception as retry_exception:
        try:
            structured_output = parser.parse(raw_output)
            return structured_output.dict()
        except Exception as parse_exception:
            # If all parsing attempts fail, return a fallback response.
            return {"message": raw_output, "products": []}


if __name__ == "__main__":
    run_interactive()
