import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")

import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from transformers import CLIPModel, CLIPTokenizer
import torch

from config import (
    openai_api_key,
    VECTORSTORE_TEXT_PATH,
    VECTORSTORE_IMAGE_PATH
)

# 1) Pydantic Model and Output Parser
class Recommendations(BaseModel):
    """Schema for structured JSON output."""
    message: str = Field(..., description="A human-friendly response message.")
    products: List[int] = Field(..., description="A list of product IDs recommended to the user.")


parser = PydanticOutputParser(pydantic_object=Recommendations)

format_instructions = parser.get_format_instructions()

# 2) Load FAISS Indexes
def load_text_index():
    """Load the FAISS index for text-based product retrieval."""
    text_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_store = FAISS.load_local(
        folder_path=VECTORSTORE_TEXT_PATH,
        embeddings=text_embedding,
        allow_dangerous_deserialization=True
    )
    return text_store


class CLIPTextEmbeddings(Embeddings):
    """CLIP Text Embeddings for image retrieval."""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts: list) -> list:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list:
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features[0].cpu().numpy().tolist()


def load_image_index():
    """Load the FAISS index for image-based (CLIP) product retrieval."""
    clip_text_embed = CLIPTextEmbeddings("openai/clip-vit-base-patch32")
    image_store = FAISS.load_local(
        folder_path=VECTORSTORE_IMAGE_PATH,
        embeddings=clip_text_embed,
        allow_dangerous_deserialization=True
    )
    return image_store

# 3) Define Tool Functions That Return JSON
def retrieve_text_products(query: str, text_store: FAISS) -> str:
    """Search text-based FAISS index, return JSON string with product IDs & message."""
    docs = text_store.similarity_search(query, k=5)
    if not docs:
        rec = Recommendations(
            message="I couldn't find any matching products. Could you provide more details?",
            products=[]
        )
        return rec.json()

    product_ids = []
    lines = []
    for doc in docs:
        meta = doc.metadata
        pid = meta.get("id")
        if not pid:
            continue
        try:
            pid = int(pid)
        except:
            continue
        name = meta.get("productDisplayName", "Unnamed Product")
        product_ids.append(pid)
        lines.append(f"- {name} (Product ID: {pid})")

    message = "Based on your query, here are some suggestions:\n" + "\n".join(lines)
    rec = Recommendations(message=message, products=product_ids)
    return rec.json()

def retrieve_image_products(query: str, image_store: FAISS) -> str:
    """Use CLIP for image-based retrieval, return JSON string with product IDs & message."""
    docs = image_store.similarity_search(query, k=5)
    if not docs:
        rec = Recommendations(
            message="I couldn't find any images matching that description. Could you clarify?",
            products=[]
        )
        return rec.json()

    product_ids = []
    lines = []
    for doc in docs:
        meta = doc.metadata
        pid = meta.get("id")
        if not pid:
            continue
        try:
            pid = int(pid)
        except:
            continue
        product_ids.append(pid)
        lines.append(f"- Product ID: {pid}")

    message = "Here are the best matches based on your description:\n" + "\n".join(lines)
    rec = Recommendations(message=message, products=product_ids)
    return rec.json()

# 4) Create Tools and Memory
def create_tools_and_memory():
    text_store = load_text_index()
    image_store = load_image_index()

    text_tool = Tool(
        name="Product Search",
        func=lambda q: retrieve_text_products(q, text_store),
        description=(
            "Use this tool for queries about products' categories, names, usage..."
            " The tool returns a JSON string with 'message' and 'products'."
            " Simply return that JSON string verbatim as your final answer if you call this tool."
        )
    )
    image_tool = Tool(
        name="Appearance Product Search",
        func=lambda q: retrieve_image_products(q, image_store),
        description=(
            "Use this tool for queries about product appearance, color, or pattern."
            " The tool returns a JSON string with 'message' and 'products'."
            " Simply return that JSON string verbatim as your final answer if you call this tool."
        )
    )
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    return [text_tool, image_tool], memory

# 5) Prompt Template for the Agent (with Format Instructions)
def create_agent():
    tools, memory = create_tools_and_memory()
    
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="chatgpt-4o-latest",
        temperature=0
    )
    raw_system_prompt = """
    You are a helpful shopping assistant. Your task is to answer customer queries ONLY by outputting a valid JSON object with exactly two keys: "message" and "products". Do not output any internal reasoning or extra text.

    You have access to two tools:
    1. "Product Search" – for queries about product names, categories, or usage.
    2. "Appearance Product Search" – for queries about product appearance (color, pattern, style).

    If you need to call a tool, return its JSON output exactly.
    If you do NOT use a tool, output a JSON object exactly like this:
    {{"message": "<your answer>", "products": []}}

    Format Instructions:
    {format_instructions}

    Examples:
    Example 1:
    User: Hi
    AI: {{"message": "Hi, how can I help you today?", "products": []}}

    Example 2:
    User: I need hiking shoes.
    AI: {{"message": "Sure, here are some hiking shoes. Do you have a preferred brand or usage in mind?", "products": [7753, 37241, 39263]}}

    Example 3:
    User: I'm looking for a black and white pair for cold days.
    AI: {{"message": "Great! Here are some black and white options that are perfect for cold days:", "products": [12345, 67890, 23456, 78901]}}

    Remember: Your FINAL output must be ONLY the JSON object.
    """
    # Inject format_instructions into the system prompt:
    system_prompt = raw_system_prompt.format(format_instructions=format_instructions)
    suffix = """
            IMPORTANT:
            - Use Provided Tools
            - Your final answer MUST be only a valid JSON object with exactly two keys: "message" and "products".
            - Do not output any internal reasoning or extra text.
            """

    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        #agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "suffix": suffix,
            "system_message": system_prompt,
            "input_variables": ["input"]
            }
    )
    
    return agent, system_prompt

# 6) Main Interactive Loop with Structured Parser
def run_interactive():
    agent, system_message_text = create_agent()
    
    print("\n=== Welcome to the AI Shopping Assistant! ===\n")
    print("Type 'exit' or 'quit' to stop.\n")

    #conversation_history = [SystemMessage(content=system_message_text)]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        #conversation_history.append(HumanMessage(content=user_input))

        raw_output_dict = agent.invoke({"input": user_input}, eturn_intermediate_steps=True)
        print("\nDEBUG OUTPUT:\n", raw_output_dict)
        raw_output = raw_output_dict["output"]

        # 7) Parse final text with the Pydantic parser
        try:
            structured_output = parser.parse(raw_output)
        except OutputParserException:

            structured_output = {"message": raw_output, "products": []}
        
        # Display final structured output to user
        print(f"\nAI: {structured_output}\n")

        #conversation_history.append(AIMessage(content=raw_output))


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
