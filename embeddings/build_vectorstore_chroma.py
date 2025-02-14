import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")
import os
import pandas as pd
from PIL import Image
import torch
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration
from config import DATA_PATH, IMAGES_FOLDER, openai_api_key, VECTORSTORE_TEXT_PATH_Chroma, VECTORSTORE_IMAGE_PATH_Chroma, VECTORSTORE_IMAGE_DESC_PATH_Chroma

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

# Initialize BLIP for image captioning (to generate image descriptions)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()

def generate_image_description(image_path):
    """Generate a caption for an image using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

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

def load_data():
    """Load and preprocess data from CSV."""
    data = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    data['image_path'] = data['id'].apply(lambda x: os.path.join(IMAGES_FOLDER, f"{x}.jpg"))
    data = data[data['image_path'].apply(os.path.exists)].reset_index(drop=True)
    return data

def create_text_index():
    """Create the text index using product names with OpenAI embeddings."""
    data = load_data()
    text_docs = []
    for idx, row in data.iterrows():
        product_name = row["productDisplayName"] if pd.notna(row["productDisplayName"]) else "Unnamed Product"
        doc = Document(page_content=product_name, metadata=row.to_dict())
        text_docs.append(doc)
    
    # Remove existing collection if it exists
    if os.path.exists(VECTORSTORE_TEXT_PATH_Chroma):
        store = Chroma(
            persist_directory=VECTORSTORE_TEXT_PATH_Chroma,
            embedding_function=text_embedding_model,
            collection_name="text_products"
        )
        store.delete_collection()
    os.makedirs(VECTORSTORE_TEXT_PATH_Chroma, exist_ok=True)
    text_store = Chroma.from_documents(
        documents=text_docs,
        embedding=text_embedding_model,
        persist_directory=VECTORSTORE_TEXT_PATH_Chroma,
        collection_name="text_products"
    )
    text_store.persist()
    print("Text index created and persisted.")

def create_image_index():
    """Create the image index using CLIP embeddings."""
    data = load_data()
    image_docs = []
    for idx, row in data.iterrows():
        product_name = row["productDisplayName"] if pd.notna(row["productDisplayName"]) else "Unnamed Product"
        doc = Document(
            page_content=f"Image for {product_name}",
            metadata={"id": row["id"], "productDisplayName": product_name, "image_path": row["image_path"]}
        )
        image_docs.append(doc)
    
    if os.path.exists(VECTORSTORE_IMAGE_PATH_Chroma):
        store = Chroma(
            persist_directory=VECTORSTORE_IMAGE_PATH_Chroma,
            embedding_function=clip_embedding_model,
            collection_name="image_products"
        )
        store.delete_collection()
    os.makedirs(VECTORSTORE_IMAGE_PATH_Chroma, exist_ok=True)
    image_store = Chroma.from_documents(
        documents=image_docs,
        embedding=clip_embedding_model,
        persist_directory=VECTORSTORE_IMAGE_PATH_Chroma,
        collection_name="image_products"
    )
    image_store.persist()
    print("Image index created and persisted.")

def create_image_caption_index():
    """Create an index of image captions (descriptions) using BLIP and text embeddings."""
    data = load_data()
    caption_docs = []
    captions_list = []  # For storing id and caption for CSV output
    for idx, row in data.iterrows():
        product_name = row["productDisplayName"] if pd.notna(row["productDisplayName"]) else "Unnamed Product"
        caption = generate_image_description(row["image_path"])
        doc = Document(
            page_content=caption,
            metadata={"id": row["id"], "productDisplayName": product_name, "image_path": row["image_path"], "caption": caption}
        )
        caption_docs.append(doc)
        captions_list.append({"id": row["id"], "caption": caption})
        
        if idx%50==0:
            print(idx, caption)
    
    # Save the captions to a CSV file for further use
    captions_df = pd.DataFrame(captions_list)
    csv_path = os.path.join(os.path.dirname(VECTORSTORE_IMAGE_DESC_PATH_Chroma), "image_captions.csv")
    captions_df.to_csv(csv_path, index=False)
    print(f"Image captions saved to {csv_path}")
    print(len(caption_docs))

    if os.path.exists(VECTORSTORE_IMAGE_DESC_PATH_Chroma):
        store = Chroma(
            persist_directory=VECTORSTORE_IMAGE_DESC_PATH_Chroma,
            embedding_function=text_embedding_model,
            collection_name="image_description"
        )
        store.delete_collection()
    os.makedirs(VECTORSTORE_IMAGE_DESC_PATH_Chroma, exist_ok=True)
    caption_store = Chroma.from_documents(
        documents=caption_docs,
        embedding=text_embedding_model,  # Use text embeddings for captions
        persist_directory=VECTORSTORE_IMAGE_DESC_PATH_Chroma,
        collection_name="image_description"
    )
    caption_store.persist()
    print("Image caption index created and persisted.")

from concurrent.futures import ThreadPoolExecutor, as_completed

def create_image_caption_index_con():
    data = load_data()
    caption_docs = []
    captions_list = []

    # Use ThreadPoolExecutor to generate captions concurrently.
    with ThreadPoolExecutor() as executor:
        # Submit caption generation tasks for each image path.
        future_to_row = {executor.submit(generate_image_description, row["image_path"]): row 
                         for idx, row in data.iterrows()}
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                caption = future.result()
            except Exception as exc:
                print(f"Caption generation failed for id {row['id']} with exception: {exc}")
                caption = ""
            product_name = row["productDisplayName"] if pd.notna(row["productDisplayName"]) else "Unnamed Product"
            doc = Document(
                page_content=caption,
                metadata={
                    "id": row["id"],
                    "productDisplayName": product_name,
                    "image_path": row["image_path"],
                    "caption": caption
                }
            )
            caption_docs.append(doc)
            captions_list.append({"id": row["id"], "caption": caption})
            
            if int(row["id"])%100==0:
                print(row["id"], caption)
    
    # Save the captions to a CSV file for further use.
    captions_df = pd.DataFrame(captions_list)
    csv_path = os.path.join(os.path.dirname(VECTORSTORE_IMAGE_DESC_PATH_Chroma), "image_captions.csv")
    captions_df.to_csv(csv_path, index=False)
    print(f"Image captions saved to {csv_path}")

    # Delete the existing image caption collection if it exists.
    if os.path.exists(VECTORSTORE_IMAGE_DESC_PATH_Chroma):
        store = Chroma(
            persist_directory=VECTORSTORE_IMAGE_DESC_PATH_Chroma,
            embedding_function=text_embedding_model,
            collection_name="image_description"
        )
        store.delete_collection()
    os.makedirs(VECTORSTORE_IMAGE_DESC_PATH_Chroma, exist_ok=True)
    caption_store = Chroma.from_documents(
        documents=caption_docs,
        embedding=text_embedding_model,  # Use text embeddings for captions
        persist_directory=VECTORSTORE_IMAGE_DESC_PATH_Chroma,
        collection_name="image_description"
    )
    caption_store.persist()
    print("Image caption index created and persisted.")



if __name__ == "__main__":
    #create_text_index()
    #create_image_index()
    create_image_caption_index()
    #create_image_caption_index_con()