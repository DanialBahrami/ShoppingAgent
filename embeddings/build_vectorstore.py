import sys
sys.path.append("/Users/daniel/Documents/ShoppingAgent/")

import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from PIL import Image
import torch
from config import (
    DATA_PATH,
    IMAGES_FOLDER,
    openai_api_key,
    VECTORSTORE_TEXT_PATH,
    VECTORSTORE_IMAGE_PATH
)

# ============================
# 1. LOAD AND PREPROCESS DATA
# ============================
data = pd.read_csv(DATA_PATH, on_bad_lines='skip')
data['image_path'] = data['id'].apply(lambda x: os.path.join(IMAGES_FOLDER, f"{x}.jpg"))
data = data[data['image_path'].apply(os.path.exists)].reset_index(drop=True)

# ============================
# 2. INITIALIZE MODELS
# ============================
text_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ============================
# 3. EMBEDDING FUNCTIONS
# ============================
def create_text_embedding(row):
    """
    Combine text fields and embed using OpenAIEmbeddings.
    Returns a 1D numpy array or list[float].
    """
    text = (
        f"Gender: {row['gender']}, Master Category: {row['masterCategory']}, "
        f"Subcategory: {row['subCategory']}, Article Type: {row['articleType']}, "
        f"Base Colour: {row['baseColour']}, Season: {row['season']}, "
        f"Usage: {row['usage']}, Product Name: {row['productDisplayName']}"
    )
    return text_model.embed_query(text)


def create_image_embedding(image_path):
    """
    Load an image and embed it using CLIP.
    Returns a 1D numpy array.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

# ============================
# 4. LOAD OR INIT FAISS STORES
# ============================
# We'll create or load two separate FAISS indexes (text vs. image).

text_faiss = None
if os.path.isdir(VECTORSTORE_TEXT_PATH):
    text_faiss = FAISS.load_local(VECTORSTORE_TEXT_PATH, text_model, allow_dangerous_deserialization=True)

image_faiss = None
if os.path.isdir(VECTORSTORE_IMAGE_PATH):
    image_faiss = FAISS.load_local(VECTORSTORE_IMAGE_PATH, text_model, allow_dangerous_deserialization=True)

# ============================
# 5. CHUNK-BASED PARTIAL BUILD
# ============================
CHUNK_SIZE = 1000  

text_docs_buffer = []
text_emb_buffer = []

image_docs_buffer = []
image_emb_buffer = []

def flush_buffers():
    global text_faiss, image_faiss
    global text_docs_buffer, text_emb_buffer
    global image_docs_buffer, image_emb_buffer
    
    # --- TEXT ---
    if text_emb_buffer:
        text_metadatas = [doc.metadata for doc in text_docs_buffer]
        # Build (text, embedding) pairs
        text_pairs = [
            (doc.page_content, emb)
            for doc, emb in zip(text_docs_buffer, text_emb_buffer)
        ]
        
        if text_faiss is None:
            text_faiss = FAISS.from_embeddings(
                text_pairs,   # a list of (string, embedding)
                text_model,   # second positional arg (the Embeddings instance)
                metadatas=text_metadatas
            )
        else:
            # The add_embeddings expects the same (text, embedding) format
            text_faiss.add_embeddings(
                text_pairs,
                text_metadatas
            )
        text_faiss.save_local(VECTORSTORE_TEXT_PATH)
    
    # --- IMAGE ---
    if image_emb_buffer:
        image_metadatas = [doc.metadata for doc in image_docs_buffer]
        # Build (text, embedding) pairs. The "text" here can be doc.page_content, or any label.
        image_pairs = [
            (doc.page_content, emb)
            for doc, emb in zip(image_docs_buffer, image_emb_buffer)
        ]
        
        if image_faiss is None:
            image_faiss = FAISS.from_embeddings(
                image_pairs,
                text_model,
                metadatas=image_metadatas
            )
        else:
            image_faiss.add_embeddings(
                image_pairs,
                image_metadatas
            )
        image_faiss.save_local(VECTORSTORE_IMAGE_PATH)

    # Clear buffers
    text_docs_buffer.clear()
    text_emb_buffer.clear()
    image_docs_buffer.clear()
    image_emb_buffer.clear()

# ============================
# 6. ITERATE AND BUILD INDEXES
# ============================

for idx, row in data.iterrows():
    # Handling NaN values
    product_display_name = row["productDisplayName"]
    if pd.isna(product_display_name):
        product_display_name = "No product display name available"
    
    # Create text embedding
    t_emb = create_text_embedding(row)
    t_doc = Document(
        page_content=product_display_name,
        metadata=row.to_dict()
    )
    text_docs_buffer.append(t_doc)
    text_emb_buffer.append(t_emb)
    
    # Create image embedding
    i_emb = create_image_embedding(row["image_path"])
    i_doc = Document(
        page_content=f"Image for {product_display_name}",
        metadata={"id": row["id"], "image_path": row["image_path"]}
    )
    image_docs_buffer.append(i_doc)
    image_emb_buffer.append(i_emb)
    
    # Chunk-based partial save
    if (idx + 1) % CHUNK_SIZE == 0:
        print(f"Processed and saving chunk up to row {idx + 1} of {len(data)} ...")
        flush_buffers()

# Flush any remaining
if text_docs_buffer or image_docs_buffer:
    print(f"Final flush for remaining {len(text_docs_buffer)} text items and "
          f"{len(image_docs_buffer)} image items.")
    flush_buffers()

print("FAISS indexes for text and image embeddings built and saved successfully.")
