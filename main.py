import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")

import os
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.llms import OpenAI  
from langchain_openai import OpenAIEmbeddings  
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from transformers import CLIPModel, CLIPTokenizer
import torch

from config import (
    openai_api_key,
    VECTORSTORE_TEXT_PATH,
    VECTORSTORE_IMAGE_PATH
)

#########################################################################
# 1) Load Vector Stores
#########################################################################
def load_text_index():
    """Load the FAISS index for text-based product retrieval."""
    text_embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_store = FAISS.load_local(
        folder_path=VECTORSTORE_TEXT_PATH,
        embeddings=text_embedding,
        allow_dangerous_deserialization=True
    )
    return text_store

# CLIP Text Embedding Class for Image Retrieval
class CLIPTextEmbeddings(Embeddings):
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
    """Load the FAISS index for cross-modal image retrieval using CLIPTextEmbeddings."""
    clip_text_embed = CLIPTextEmbeddings("openai/clip-vit-base-patch32")
    image_store = FAISS.load_local(
        folder_path=VECTORSTORE_IMAGE_PATH,
        embeddings=clip_text_embed,
        allow_dangerous_deserialization=True
    )
    return image_store

#########################################################################
# 2) Tools: Retrieval + Returning Formatted Product IDs
#########################################################################

def retrieve_text_products(query: str, text_store: FAISS) -> str:
    """
    Search the text-based FAISS index for top matches.
    Returns a conversationally formatted response with product IDs & descriptions.
    """
    docs = text_store.similarity_search(query, k=5)

    if not docs:
        return "I couldn't find any matching products. Could you provide more details?"

    # Extract product details
    product_recommendations = []
    for doc in docs:
        meta = doc.metadata
        pid = meta.get("id", "UnknownID")
        name = meta.get("productDisplayName", "Unnamed Product")
        product_recommendations.append(f"- **{name}** (Product ID: {pid})")

    return f"Based on your query, I recommend the following products:\n" + "\n".join(product_recommendations)

def retrieve_image_products(query: str, image_store: FAISS) -> str:
    """
    Uses CLIP's text-to-image retrieval to find the best image matches.
    Returns a formatted response with product IDs.
    """
    docs = image_store.similarity_search(query, k=5)

    if not docs:
        return "I couldn't find any images that match your description. Could you describe it differently?"

    product_recommendations = []
    for doc in docs:
        meta = doc.metadata
        pid = meta.get("id", "UnknownID")
        product_recommendations.append(f"- Product ID: {pid}")

    return f"Here are the best matches based on your description:\n" + "\n".join(product_recommendations)

#########################################################################
# 3) Create a Conversational Agent with Memory
#########################################################################

def create_conversational_agent(text_store: FAISS, image_store: FAISS):
    """
    An AI assistant that:
    - Keeps track of conversation history (multi-turn dialog).
    - Uses tools for text-based & image-based product recommendations.
    """

    # Tool #1: Text-based retrieval
    text_tool = Tool(
        name="Product Search",
        func=lambda query: retrieve_text_products(query, text_store),
        description=(
           "Use this tool when the user is looking for products based on NAME, CATEGORY, FUNCTION, OR BRAND. "
            "For example, use this tool for queries like 'I need hiking shoes, I need jean from Zara "
            " or 'What are the best waterproof jackets?'. "
        )
    )

    # Tool #2: Image-based retrieval
    image_tool = Tool(
        name="Appearance Product Search",
        func=lambda query: retrieve_image_products(query, image_store),
        description=(
            "Use this tool when the user describes a product's VISUAL APPEARANCE, COLORS, OR PATTERNS. "
            "For example, use this tool for queries like 'Show me a black T-shirt with white stripes', "
            "'I need shoes with a floral print', 'Find me a zebra pattern dress', or "
            "'I want a blue and gold ethnic outfit'. "
        )
    )

    # Memory to store conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # Initialize agent with conversational memory
    agent = initialize_agent(
        tools=[text_tool, image_tool],
        llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    return agent

#########################################################################
# 4) Run Interactive Session
#########################################################################

if __name__ == "__main__":
    # Load indexes
    text_store = load_text_index()
    image_store = load_image_index()

    # Create an AI shopping assistant
    agent = create_conversational_agent(text_store, image_store)

    print("\n=== Welcome to the AI Shopping Assistant! ===")
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye! Happy shopping. 🛍️")
            break

        # Get response from agent
        response = agent.run(user_query)
        print("\nAI: ", response)
