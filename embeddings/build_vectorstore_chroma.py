import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")
import os
import pandas as pd
from PIL import Image
import torch
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from transformers import CLIPModel, CLIPProcessor
from config import DATA_PATH, IMAGES_FOLDER, openai_api_key, VECTORSTORE_TEXT_PATH_Chroma, VECTORSTORE_IMAGE_PATH_Chroma

text_embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Revised CLIP embeddings class to support both image and text queries
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

def create_text_embedding(row):
    text = (
        f"Gender: {row['gender']}, Master Category: {row['masterCategory']}, "
        f"Subcategory: {row['subCategory']}, Article Type: {row['articleType']}, "
        f"Base Colour: {row['baseColour']}, Season: {row['season']}, "
        f"Usage: {row['usage']}, Product Name: {row['productDisplayName']}"
    )
    return text_embedding_model.embed_query(text)

def create_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    return clip_embedding_model.embed_image(image)

def create_vectordb():
    data = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    data['image_path'] = data['id'].apply(lambda x: os.path.join(IMAGES_FOLDER, f"{x}.jpg"))
    data = data[data['image_path'].apply(os.path.exists)].reset_index(drop=True)
    text_docs = []
    image_docs = []

    for idx, row in data.iterrows():
        product_name = row["productDisplayName"] if pd.notna(row["productDisplayName"]) else "Unnamed Product"
        t_doc = Document(page_content=product_name, metadata=row.to_dict())
        text_docs.append(t_doc)
        i_doc = Document(page_content=f"Image for {product_name}", metadata={"id": row["id"], "productDisplayName": product_name, "image_path": row["image_path"]})
        image_docs.append(i_doc)

    os.makedirs(VECTORSTORE_TEXT_PATH_Chroma, exist_ok=True)
    text_vector_store = Chroma.from_documents(
        documents=text_docs,
        embedding=text_embedding_model,
        persist_directory=VECTORSTORE_TEXT_PATH_Chroma,
        collection_name="text_products"
    )
    text_vector_store.persist()

    os.makedirs(VECTORSTORE_IMAGE_PATH_Chroma, exist_ok=True)
    image_vector_store = Chroma.from_documents(
        documents=image_docs,
        embedding=clip_embedding_model,  # Use CLIP for images; note embed_query returns text embeddings
        persist_directory=VECTORSTORE_IMAGE_PATH_Chroma,
        collection_name="image_products"
    )
    image_vector_store.persist()

    print("Chroma vector stores for text and image embeddings built and saved successfully.")


if __name__ == "__main__":
    create_vectordb()