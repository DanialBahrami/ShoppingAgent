import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")
import os
import json
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.memory import ConversationBufferMemory
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from config import openai_api_key, VECTORSTORE_TEXT_PATH, VECTORSTORE_IMAGE_PATH
from config import (
    openai_api_key,
    VECTORSTORE_TEXT_PATH_Chroma,
    VECTORSTORE_IMAGE_PATH_Chroma
)

# 1) Pydantic Model for final JSON output
class Recommendations(BaseModel):
    message: str = Field(..., description="A human-friendly response message.")
    products: List[int] = Field(..., description="A list of product IDs recommended to the user.")

parser = PydanticOutputParser(pydantic_object=Recommendations)
format_instructions = parser.get_format_instructions()

# 2) Define functions to load persistent Chroma vector stores.
def load_text_index():
    """Load the persistent Chroma vector store for text products."""
    text_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_store = Chroma(
        persist_directory=VECTORSTORE_TEXT_PATH_Chroma,
        embedding_function=text_embedding,
        collection_name="text_products"
    )
    return text_store

class CLIPEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def embed_image(self, image: Image.Image) -> list:
        inputs = self.processor(images=image, return_tensors="pt", truncation=True)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features[0].cpu().numpy().tolist()

    def embed_text(self, text: str) -> list:
        inputs = self.processor(text=[text], return_tensors="pt", truncation=True)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features[0].cpu().numpy().tolist()

    # For compatibility with Chroma's API:
    def embed_documents(self, items: list) -> list:
        # Here we assume items are strings (for queries) when used for retrieval.
        return [self.embed_text(item) for item in items]

    def embed_query(self, query: str) -> list:
        return self.embed_text(query)

clip_embedding_model = CLIPEmbeddings()
def load_image_index():
    """Load the persistent Chroma vector store for image products.
       Uses the imported clip_embedding_model.
    """
    image_store = Chroma(
        persist_directory=VECTORSTORE_IMAGE_PATH_Chroma,
        embedding_function=clip_embedding_model, 
        collection_name="image_products"
    )
    return image_store

# 3) Define the RAG tool function
def retrieve_products_rag(query: str) -> str:
    """
    Perform retrieval augmented generation:
      - Query both text and image Chroma vector stores for top-5 similar products.
      - Deduplicate products (using metadata 'id').
      - Construct an augmented context string from the retrieved product info.
      - Use ChatOpenAI to produce a final JSON output (with keys 'message' and 'products').
    """
    text_store = load_text_index()
    image_store = load_image_index()
    # Retrieve top-5 products from both stores.
    text_results = text_store.similarity_search(query, k=5)
    image_results = image_store.similarity_search(query, k=5)
    
    # Combine results and deduplicate by product ID (assuming each doc.metadata contains an "id").
    combined = {}
    for doc in text_results + image_results:
        pid = doc.metadata.get("id")
        if pid is None:
            continue
        # Avoid duplicates: if already present, skip.
        if pid not in combined:
            # Use productDisplayName if available, or page_content.
            name = doc.metadata.get("productDisplayName", doc.page_content)
            combined[pid] = name
    
    # Build the context string.
    context_lines = []
    for pid, name in combined.items():
        context_lines.append(f"Product ID: {pid}, Name: {name}")
    context_str = "\n".join(context_lines)
    
    # Create an augmented prompt for the LLM.
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
    # Use ChatOpenAI to generate the answer.
    llm_for_rag = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="chatgpt-4o-latest",
        temperature=0
    )
    response = llm_for_rag([{"role": "system", "content": augmented_prompt}])
    final_output = response["output"]
    return final_output

# 3) Create Tools and Memory for the agent.
def create_tools_and_memory():
    rag_tool = Tool(
        name="RAG Product Retrieval",
        func=retrieve_products_rag,
        description=(
            "Use this tool to perform a retrieval augmented generation: "
            "It retrieves top-5 similar products from both text and image collections, "
            "deduplicates them, augments the context, and returns a JSON string with 'message' and 'products'."
        )
    )
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return [rag_tool], memory

# 4) Prompt Template for the Agent
def create_agent():
    tools, memory = create_tools_and_memory()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="chatgpt-4o-latest",
        temperature=0
    )
    raw_system_prompt = f"""
    You are a helpful shopping assistant. Answer customer queries ONLY by outputting a valid JSON object with exactly two keys: "message" and "products". Do not output any internal reasoning or extra text.

    Format Instructions:
    {format_instructions}

    Example:
    Input: "I want a black shoe with yellow lines"
    Output: {{"message": "Here are some suggestions: ...", "products": [12345, 67890]}}
    """
    suffix = """
    IMPORTANT:
    - Use the provided tool to perform retrieval.
    - Your final answer MUST be only a valid JSON object with exactly two keys: "message" and "products".
    - Do not output any extra text.
    """
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": raw_system_prompt,
            "suffix": suffix,
            "input_variables": ["input"]
        }
    )
    return agent

# 5) Main Interactive Loop
def run_interactive():
    agent = create_agent()
    print("\n=== Welcome to the RAG Shopping Assistant! ===\n")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Use our RAG tool by invoking the agent with the query.
        raw_output_dict = agent.invoke({"input": user_input})
        print("\nDEBUG OUTPUT:\n", raw_output_dict)
        raw_output = raw_output_dict["output"]
        
        try:
            structured_output = parser.parse(raw_output)
        except OutputParserException:
            structured_output = {"message": raw_output, "products": []}
        
        print(f"\nAI: {structured_output}\n")

def process_query(agent: AgentExecutor, parser: PydanticOutputParser, user_input: str) -> dict:
    """
    1. Call agent.run(user_input) to get the final text.
    2. Parse it via the PydanticOutputParser into {message, products}.
    3. If parsing fails, return {message: raw_output, products: []}.
    """
    raw_output_dict = agent.invoke({"input": user_input})
    raw_output = raw_output_dict["output"]
    try:
        structured_output = parser.parse(raw_output)
        return structured_output.dict()  
    except OutputParserException:
        # If the LLM doesn't comply with the format instructions,
        # fallback to the raw text as "message" with an empty products list.
        return {
            "message": raw_output,
            "products": []
        }


if __name__ == "__main__":
    run_interactive()
